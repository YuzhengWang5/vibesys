# 投机解码：prediction 长后缀对齐与接受率门控草稿

本指南记录一条具体实现路线：从当前请求的 prediction 提取草稿，优先使用 8-token 后缀定位，并由真实 target 一次验证多个 token；草稿长度从 4 开始，按连续完整接受情况升到 8。

## 1. 优化身份与边界

这是基于输入 prediction 的贪心投机解码，不是用另一个小模型生成草稿。

- 草稿只来自当前请求的 prediction 文本。
- 输出的决定权始终在真实生产 target 模型。
- target 的权重、精度、attention、位置处理和 KV 表示不因这项优化而改变。
- 不复用已完成请求的答案，不从 prompt 构建另一种 drafter。
- 没有 prediction 或暂时找不到可用片段时，继续普通自回归生成。

记录的完整路线包括：初始精确前缀推进、分歧后优先最早 8-token 后缀匹配、4-token 匹配兜底、4/8 草稿长度门控、最长贪心前缀接受、pending token 和事务式 KV 提交。

它不包含分歧后沿已恢复位置持续推进的 sticky cursor，也不包含“只允许恢复一次、搜索 32 token”等其他尝试的规则。

适用场景是 prediction 与待生成文本存在可复用片段，例如基于已有代码内容生成编辑后的代码。它描述可实施的机制，不承诺所有输入都会产生收益。

## 2. 核心机制

每次调用 target，决定一个新 token。若 prediction 给出了可能连续正确的草稿，可以一次把这些候选送入 target，再逐个判断哪些与 target 的贪心选择一致。

只接受从草稿开头连续匹配的最长前缀：

```text
草稿：             A B C D
target 对齐的选择：A B X ...

接受：A B
纠正：X
丢弃：C D 及其未接受的 KV
```

不能跳过被拒绝的 C 再接受 D，因为后面的预测依赖包含错误草稿的上下文。

本路线维护一个 pending token：它已经由 target 决定并交给生成输出流程，但还没有写入 KV。下次把 pending 与草稿一起 forward，就能同时完成 pending 的计算和草稿验证。

因此，一个验证块通常推进“接受的草稿 token + 一个新的纠正或 bonus token”，而不是无条件输出 prediction。

## 3. 接入所需能力

在当前工程里找到以下语义角色：

1. **输入与 tokenization**：能取得请求 prediction，并用生产 tokenizer 编成 token ID，不添加另一套模板或额外 special token。
2. **贪心生成循环**：能从 target logits 选择 argmax，并按 token 顺序驱动停止判断和流式输出。
3. **多 token 因果 forward**：能对输入块返回各个位置的下一 token logits，而不只是最后一行。
4. **请求级 KV 状态**：知道所有层缓存的已提交长度、位置计数和绝对位置。
5. **可撤销的 KV 更新**：一次候选 forward 后，能保留输入块的前缀，截去未提交后缀，并同步位置状态。
6. **已确认 token 与文本输出的分离**：UTF-8、stop-prefix 或 SSE 缓冲不能改变模型 token 的确认顺序。

不要求使用特定 KV 类或事务类，但必须能表达相同的提交、回滚语义。

## 4. 推荐实现路径

### 步骤 1：为 prediction 建立两个位置索引

把 prediction 编成 token 序列 `P`，为连续 4-token 和 8-token 窗口分别建立索引：

```text
窗口 token tuple -> 该窗口最早出现位置的结束下标
```

同一窗口出现多次时，只保留最早的一次。结束下标指向窗口之后的第一个 token，草稿从这里开始取。

索引只建立一次。每个请求独立持有：

- prediction token 序列和两份索引；
- `prefix_mode = true`，初始 prediction cursor 为 0；
- 最多 8 个已确认生成 token 组成的 tail；
- 当前草稿长度 `width = 4`；
- 连续完整接受的 4-token 验证块计数，初始为 0。

tail 来自本请求已经由 target 确认并交给输出处理的生成 token，不是 prompt，也不是未接受草稿或尚未发送的文本片段。

### 步骤 2：处理初始精确前缀与后续对齐

每确认一个非 EOS 输出 token，就更新 tail，并按以下顺序处理：

**仍处于初始 prefix mode：**

- 若该 token 等于 `P[cursor]`，cursor 前进一个位置。
- 此时可以直接从 prediction 当前 cursor 继续提出草稿。
- 第一次不匹配，或 prediction 已没有对应 token 时，永久离开初始 prefix mode。

**已经离开初始 prefix mode：**

1. 如果已有 8 个 tail token，在 8-token 索引查找整个 tail。
2. 找到时使用这个窗口最早出现位置的结束下标。
3. 否则，若至少已有 4 个 tail token，查询最后 4 个 token 的最早匹配。
4. 两者都找不到时，对齐失效。

分歧后的每个已确认输出 token 都重新执行这一查询。重新匹配成功后，也不转为 sticky cursor。

8-token anchor 用于确定“在 prediction 的哪里取草稿”，不是授权一次接受 8 个 token。对齐长度与草稿长度是两个独立概念。

### 步骤 3：prefill，并产生第一个 pending token

对完整 prompt 做一次真实 target forward，提交 prompt KV。

从最后一行 logits 的 argmax 取得第一个生成 token `p`，交给正常输出处理，并更新 prediction 对齐状态。

此时：

```text
KV：已经计算的 prompt
生成流程：已经处理 p
pending：p，尚未进入 KV
```

如果 p 是 EOS、触发停止条件或已经用完输出预算，就结束，不再做验证块。

### 步骤 4：按对齐位置和预算提取草稿

从当前 cursor 提取至多 `width` 个 prediction token。

设：

- `N`：当前已提交的 KV 长度；
- `G`：生成循环当前已处理的模型 token 数，包含当前 pending；
- `M`：输出 token 上限；
- `C`：上下文长度上限。

实际草稿长度受以下上限共同限制：

```text
k = min(
  width,
  prediction 在 cursor 后的剩余 token 数,
  M - G - 1,
  C - N - 1
)
```

输出预算预留的 1 个位置用于纠正或 bonus token；forward 输入预算预留的 1 个位置用于旧 pending。

没有对齐、没有可用后继 token，或预算只允许普通一步时，使用空草稿并重置长度门控。仍然可以 forward 旧 pending，产生下一个普通 AR token。

正常请求入口应保证长度参数有效，不应把负预算当作合法草稿长度。

### 步骤 5：把 pending 与草稿放进一次 target forward

草稿为 `D = [d_1, ..., d_k]`。本次输入：

```text
X = [p, d_1, ..., d_k]
```

已有 KV 是 X 之前的上下文。一次因果 forward 得到 `k + 1` 行 logits，并取每行 argmax 为 `winner[0..k]`。

对齐关系必须是：

- `winner[0]` 是看过 pending 后对 `d_1` 的预测。
- `winner[i]` 用于比较 `D[i]`，其中 i 从 0 开始。
- `winner[k]` 是所有草稿被接受后，块末尾的下一 token 候选。

不要拿草稿自身输入位置之后的错误一行 logits 验证它，也不要把 forward 的所有输入 token 直接当作已接受输出。

### 步骤 6：决定输出，并提交 KV 前缀

从 i=0 开始比较 `winner[i] == D[i]`，在第一处不一致停止。设连续接受了 a 个草稿 token：

```text
本轮待输出 selected = D[:a] + [winner[a]]
本轮先提交的输入数 = 1 + a
提交后的 KV 长度 = N + 1 + a
```

其中的 1 是旧 pending。若 a<k，最后一个 selected token 是纠正 token；若 a=k，它是 bonus token。它尚未作为本次 forward 输入计算，因此成为下一轮 pending。

示例：

```text
旧 KV 长度：N
旧 pending：p
输入草稿：A B C D
对齐 winners：A B X ...

保留 KV：旧 KV + p + A + B
本轮输出：A B X
新 pending：X
```

空草稿时，同一公式退化为普通 AR：只 forward p，提交 p，输出 `winner[0]`，将它设为新 pending。

### 步骤 7：以事务方式维护所有 KV 与位置

一次块 forward 开始前保存旧位置 N。forward 可以暂时写入完整 X，但必须显式提交保留的输入前缀：

1. 所有层 KV 截到统一的目标长度。
2. 同步缓存自身长度追踪和生产 forward 的位置计数。
3. 下一次输入的绝对位置从该目标长度开始。
4. 没有提交、forward 异常或输出处理中断时，撤销当前事务的写入并传播错误。

只更新一个 Python position 变量，而不截去真实 K/V，或只裁剪部分层，都会使后续上下文错误。

### 步骤 8：逐 token 处理输出和停止条件

按 selected 的顺序驱动原有 token 输出流程，而不是一次把整个块转成字符串后发送：

- 非 EOS token 更新 prediction 对齐。
- 原有 UTF-8、stop-string、SSE token-ID 和 usage 规则保持不变。
- 接受的草稿后面如果遇到 EOS、stop 或输出预算边界，后面的 selected token 不再处理。

若在 selected 中间结束，需要再次裁剪本轮 KV，保持“最后处理的 token 仍是 pending”的语义。

若本轮实际处理 selected 的 token 数为 b，结束时保留的输入数为：

```text
1 + min(a, b - 1)
```

也就是旧 pending，加上最后处理 token 之前确实采用的草稿前缀。不要因提前做过完整验证，就把没有实际采用的后续草稿留在 KV 中。

### 步骤 9：更新 4/8 草稿长度门控

本路线的门控是连续完整接受规则，不是累计接受率阈值：

- 初始 width 为 4。
- 一个块实际提出完整 4 个草稿、全部接受、请求没有结束，记一次完整周期。
- 连续两次这样的周期后，width 升为 8，清空 4-token 周期计数。
- width 为 8 时，完整 8-token 草稿全部接受且未结束，保持 8。
- 部分拒绝、实际草稿被预算或 prediction 尾部截短、没有草稿、或者请求结束，都重置 width=4 和完整周期计数。

是否使用 8-token anchor 不直接改变 width；即使使用长 anchor 定位，下一块仍可能只提出 4 个 token。

## 5. 正确性与边界条件

- **prediction 没有输出决定权**：匹配只能决定提案位置，只有 target 的对齐 argmax 才能决定接受和纠正。
- **最长前缀才有效**：首个不匹配之后的草稿不能跳跃接受。
- **只用已确认 token 更新对齐**：被拒绝的草稿不进入 tail，文本缓冲也不改变 tail 的 token 顺序。
- **首次分歧后的规则固定**：优先最早 8-token 匹配，否则最早 4-token 匹配，每个已确认 token 重新定位，不继承 sticky 行为。
- **无对齐不等于永久关闭投机**：继续普通 AR 产生新 token；后续 tail 再次匹配时可以继续提出草稿。
- **pending 不能重复输出或漏算**：旧 pending 已经过输出处理，但下一次必须恰好计算一次；纠正/bonus token 不能被误认为已在 KV 中。
- **所有层状态一致**：提交后各层 KV 长度、cache length 与绝对位置计数必须一致。
- **未采用后缀必须撤销**：拒绝、提前停止和异常都不能留下对下一次 forward 可见的草稿污染。
- **索引与缓存属于当前请求**：请求结束后清理，不能借用另一个请求的 prediction、tail 或 KV。
- **错误与正常 AR 路径分开**：未匹配、prediction 用尽是正常路径；错位 logits、KV 不一致或生产模型异常必须显式失败，不能静默伪装成成功回退。
- **这是贪心路线**：不能直接把 argmax 接受规则用于随机采样而声称保持采样分布。
- **算法上的接受规则不是浮点一致性保证**：多 token forward 与逐 token forward 可能有数值差异，不能仅凭算法声称任意输入的 BF16 bitwise parity。

这条路线保留了明确的对齐、门控和提交规则，但不绑定某一份基线、具体 KV 类型、工程文件或性能提升要求。
