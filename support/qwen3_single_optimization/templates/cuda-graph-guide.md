# CUDA Graph：跨层捕获 post-attention 与下一层 QKV

本指南记录一条具体实现路线：把当前层 attention 之后的固定计算，与下一层 attention 之前的固定计算，捕获到同一个 CUDA Graph。计算角色和状态语义是实施依据，原始文件、基线和历史成绩不是实施前提。

## 1. 优化身份与边界

优化对象是单请求、单 token decode 中重复执行的固定形状计算。CUDA Graph 在初始化时记录这些 GPU 操作，运行时用当前输入重放，减少逐个提交操作的 host 调度工作。

这条路线不是捕获整个模型，也不是只给 MLP 套一个 graph：

- 每个 decoder 层有一个独立 graph 和独立的 graph-private 内存池。
- 每个 graph 包含本层 attention 之后的 residual add、post-attention norm、MLP、residual add。
- 除最后一层外，还包含下一层 input norm、原来的独立 Q/K/V projections，以及原模型规定的 Q/K head norm 和张量布局变换。
- 捕获止于下一层 RoPE 之前。
- RoPE、请求位置、mask、KV 更新、attention 和输出投影继续走当前模型的正常路径。
- 第一个 decoder 层的 input norm 和 QKV preparation 没有前驱 graph，仍正常执行。
- prefill 和多 token 输入不走这条 graph 路径。

本优化只改变上述操作的调度和跨层结果交接，不新增投影融合，不替换 attention 算法，不改变权重、精度、算术顺序或 KV 表示。

## 2. 核心机制

典型 decoder 层可以拆成：

```text
层输入 h
  -> input norm
  -> Q/K/V projections 与模型原有的 head norm、布局变换
  -> RoPE、KV 更新、attention、输出投影
  -> 加回 h
  -> post-attention norm
  -> MLP
  -> 再加回进入 post-attention norm 前的 residual
  -> 下一层
```

其中，attention 会访问随请求长度增长的 KV。其前后的 norm、MLP、projection 在单 token decode 下则具有固定输入形状。

因此，选择以下跨层捕获边界：

```text
graph_i 输入：
  本层原始 residual h_i
  本层 attention 输出 a_i

graph_i 内部：
  s_i = h_i + a_i
  h_next = s_i + MLP_i(post_norm_i(s_i))

  如果不是最后一层：
    n_next = input_norm_next(h_next)
    q_next, k_next, v_next = 下一层原有的 QKV preparation(n_next)

graph_i 输出：
  h_next
  如果存在下一层：n_next 与 RoPE 前的 q_next / k_next / v_next
```

下一层直接消费这些输出，再执行 RoPE 之后的正常 attention 路径。关键不是“提前计算后再照常计算一遍”，而是让被捕获的 input norm 和 QKV preparation 替代它们原来的 eager 执行位置。

这里的 replay 会用当前输入重新执行 GPU 计算；它不复用旧请求的答案或旧输入的 QKV。

## 3. 接入所需能力

先在当前工程中定位以下语义角色，不要求它们对应特定类或文件：

1. **可拆分的 decoder 数据流**：能分开 input norm、QKV preparation、attention 后半段和 post-attention block。
2. **固定形状的单 token 输入**：能判断一次 forward 是否属于 continuation decode，并识别 batch、token 数、dtype、device 和布局。
3. **稳定的 GPU 存储**：能创建长期存活的输入 staging buffer，并保留 graph 捕获产生的输出 buffer。
4. **原模型运算的复用**：可以使用同一批参数和原有模块执行捕获边界，而不是创建另一套模型计算结果。
5. **串行的 buffer 使用生命周期**：同一组 graph buffer 不会被两个请求或两个 forward 并发覆盖。
6. **forward 级交接状态**：能标记本次 forward、生产层、消费层，以及尚未消费的跨层输出。

该路线适配的是具有上述 norm / attention / MLP / residual 数据流的模型。其他结构不能只按模块名字替换，必须先找到实际等价的计算边界。

## 4. 推荐实现路径

### 步骤 1：拆开 attention 的 preparation 与 suffix

把 attention 分成两段，但不改变两段的原始算术：

- preparation：独立 Q/K/V projection，原有 Q/K head norm，以及对应 view、transpose。
- suffix：接收 RoPE 前 Q/K/V，执行原来的 RoPE、KV 更新、attention、输出投影。

suffix 继续使用当前 attention 的参数、缩放、mask 和缓存规则。它不能先调用完整 attention，再覆盖其中的 QKV；否则被捕获的计算会重复执行。

保留完整正常路径，供第一个 decoder 层、prefill 和多 token forward 使用。

### 步骤 2：定义每层的捕获函数

捕获函数接收两份输入：本层 attention 前保留的 residual，以及 attention 输出。

严格按原顺序完成：

```text
第一次 residual add
原 post-attention norm
原 MLP
第二次 residual add
下一层原 input norm
下一层原 QKV preparation
```

最后一层只执行前四项。不要把原有 BF16 中间舍入、norm accumulation 或独立 projection 改写成另一套算术。

下一层模块的引用只是执行链接。保留原来的参数归属，不重复注册模块或复制权重。

### 步骤 3：创建固定 buffer，并初始化 graph

设模型有 `L` 层，hidden size 为 `H`。每层两份 staging input 的形状为 `[1, 1, H]`；`L`、`H` 从当前模型读取，不硬编码历史模型的值。

初始化过程：

1. 等待模型权重和正常初始化工作完成。
2. 建立一个用于此执行路径的 CUDA stream。
3. 为每层分配固定地址的 residual 和 attention-output staging buffer。
4. 在该 stream 上用相同形状执行捕获函数的 warmup。
5. 每层创建独立 CUDA Graph；不跨层共享 graph-private pool。
6. 捕获完整边界，并持有捕获产生的 residual、normalized、Q/K/V 输出。
7. 初始化完成后，保证捕获工作结束再允许生产 replay。

warmup 和 capture 不应写入真实请求 KV。固定的是存储地址与计算形状，不是运行时输入数值。

### 步骤 4：在 forward 入口统一决定路径

创建本次 forward 的身份标记，并判定是否使用跨层 graph 路径：

- 已经过 prefill，当前是 continuation decode；
- batch 为 1，当前输入 token 数为 1；
- 中间输入与固定 buffer 的形状、dtype、device、布局兼容；
- 使用声明的串行 CUDA stream。

prefill 即使只有一个 token，也不因为形状相同而进入此路径。

不符合正常适用条件的 forward 使用完整正常路径，不生产或消费跨层 bundle。不要先在前面几层启用交接，再让后面几层无约束地改走另一条路径。

### 步骤 5：从第一层开始执行与交接

第一层没有前驱，正常执行 input norm、QKV preparation 和 attention suffix。

本层 attention 完成后：

1. 把本次 residual 复制进固定 residual buffer。
2. 把本次 attention 输出复制进固定 attention buffer。
3. replay 本层 graph。
4. 返回本层输出 residual。
5. 如果存在下一层，把 normalized 和 Q/K/V 输出标记为待消费 bundle。

bundle 绑定当前 forward 和确定的相邻层关系，不是一个可被任意层读取的全局缓存。

### 步骤 6：下一层消费 bundle，而不是重复 preparation

下一层入口取得前驱的 residual、normalized 和 Q/K/V：

- residual 用于下一层 attention 后的 residual add；
- normalized 是该层原 input norm 的结果；
- Q/K/V 是该层原 preparation 的结果，尚未做 RoPE。

消费前确认 bundle 属于当前 forward、正确的前驱与当前层，且尚未被消费。消费一次后清除待消费标记。

跳过该层原来的 eager input norm 和完整 QKV preparation，直接把 Q/K/V 交给 attention suffix。随后继续执行本层 graph，生成再下一层的 bundle。

最后一层没有后继 bundle，输出继续交给模型原有的 final norm 和输出头。

### 步骤 7：结束本次 forward

成功结束时不应遗留任何待消费 bundle。发生异常时清除 active-forward、pending-producer 和该 forward 的交接标记，并传播错误。

请求完成、失败或断开时，释放该请求自己的 KV。长期 graph buffer 可以保留，但不能保留一个仍被视为有效的旧请求交接。

## 5. 正确性与边界条件

- **同一运算、同一参数**：本优化改变调度，不改变 residual 的加法位置、norm/MLP/projection 的计算顺序或精度。
- **不可重复 preparation**：已消费前驱 QKV 的层，不能再执行原 Q/K/V projection 或 head norm。
- **保留未归一化 residual**：下一层不能把 normalized tensor 当作 residual 加回。
- **RoPE 与 KV 保持请求语义**：每步位置、mask 和 KV 长度仍由本请求决定，不进入固定 capture。
- **输出存储会被 replay 覆盖**：同一 stream 上的下游运算必须先消费本次输出，才允许后续 replay 覆盖它；不能把这些引用当作永久不变的结果。
- **缓存不能引用将被覆盖的暂存值**：本次 K/V 必须按原正常路径完成位置处理与缓存更新，不能让请求 KV 依赖下一次 replay 会改写的 graph buffer。
- **一次生产、一次消费**：缺失、重复、错层或旧 forward 的 bundle 是错误，不应静默回退。
- **正常不适用与执行失败分开**：prefill、多 token 等已声明情形使用正常路径；capture/replay 失败或交接损坏不能被包装成 eager 成功。
- **共享存储必须串行使用**：这份路线不提供同一组固定 buffer 的并发请求执行。
- **固定模型生命周期**：权重、模块或捕获形状发生变化后，不能继续使用旧 graph。

上述限制是这条实现路线的状态契约，不是性能门槛。初始化内存、输入复制和动态 attention 的开销仍然存在，不由 CUDA Graph 自动消除。
