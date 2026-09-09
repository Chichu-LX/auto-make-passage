# Download Paper Agent

该 Agent 分三阶段检索：权威会议/期刊、近三年论文、arXiv 兜底。DeepSeek 只抽取候选
元数据；确定性策略负责 venue 归一化、70% 权威 + 30% 近期配额以及最终排序。受限下载器
逐跳检查公网地址、端口、重定向、MIME、体积和 `%PDF-` 签名。缺少直链或没有真实 PDF
的候选直接拒绝，只有已经落盘并记录 SHA-256 的文件能标记为 `downloaded`。

权威源优先覆盖 NeurIPS、ICML、ICLR、ACL、EMNLP、AAAI、Nature Communications、
Bioinformatics、Briefings in Bioinformatics、Journal of Cheminformatics 等。正式出版版本优先，
arXiv 只作为合法全文镜像或末级补充。

每篇成功下载的论文会再执行标题限定的 GitHub 检索。Manifest 区分
`search_matched` 与带许可证的 `verified_open_source`，同时保留证据 URL，避免把模型猜测
写成代码开源事实。

主要产物位于 `reference/papers/` 和 `reference/manifests/`。`policy.py` 定义不可由模型修改
的下载限制，`tools/collect_dta_corpus.py` 提供可重复的 DTA 文献整理工具。
