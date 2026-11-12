# 原 prompt n-gram 投机解码

只能使用本次 HTTP 请求原 prompt 的 token 建立 n-gram 查找索引。
使用已经由 target 确认的生成后缀查找原 prompt 中的匹配，提出后续 token 草稿，
经同一生产 target forward 一次验证并接受最长匹配前缀。
找不到匹配时正常自回归，拒绝草稿时正确回滚 KV；保持 EOS、生成预算和请求状态隔离。

不得依赖独立草稿字段、读取数据集 gold、参考 gate 数据或重放历史请求输出。
不要假定草稿是答案；每个输出 token 必须由真实 target 计算决定。
保持当前权重、精度和其他机制不变，使用固定多并发 benchmark 比较真实吞吐，
保留草稿尝试、接受量及生产验证路径证据。
