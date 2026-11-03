# 三组后续实验工具

本工具只改操作者计划及objective模板，不改框架/runtime/评测/provider。
`followup_slots`为纯计划；`followup prepare`通过既有`prepare_run`公开控制器接口，
将full-decode/guided模板和固定gate独立导出到初始候选并记录真实pin。
`followup dispatch`记录intent后用原gpuctl launcher/worker，失败显式保留。
prompt由模板包含指南，Python只传variant，不拼接agent-bound内容。

```bash
bash support/qwen3_single_optimization/followup.sh prepare --experiment /absolute/experiment
bash support/qwen3_single_optimization/followup.sh dispatch --run /absolute/run
```

plan中graph repeat1/2为full-decode，graph repeat3/4/5为guided；
speculation repeat1/2/3为guided。repeat是本组唯一内部标识，不把宏观分组编号写入候选。
三组的研究比较仅由操作者记录；候选不能见历史实现或其他候选。
同一run的dispatch锁及worker锁防重跑；运行中不得改source checkpoint/脚本。
依赖uv/Python设置复用现有controller，脚本不sync主仓venv。
既有`prepare pilot/batch`与默认generic模板仍兼容；本新计划不调用旧20slot队列。

固定源码稳定性复测使用`remeasure.sh`、`run_manifest.py`，保存每次真实执行源码、
固定候选archive及原始gate/benchmark JSON。A/B同卡串行，重建服务与容器，
交替AB/BA次序，五对均保留；不从重复结果只挑最高值。
这是固定测量，不给候选优化agent旧实现，也不占用新实现的五轮预算。
