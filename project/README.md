# Auto Make Passage

仅支持 Python 3.12 的科研多 Agent 运行系统：DeepSeek/OpenAI 兼容 API 负责主体任务，
Tavily 负责受权威域名白名单约束的检索，Claude Agent SDK 可选用于最终审稿与 Feedback，
LangGraph 负责执行条件路由守卫并校验每次状态迁移。

下文所有 Shell 命令均假设已进入仓库的 `project/` 目录；正文中以 `project/` 开头的路径
则相对于仓库根目录书写。克隆后先执行 `cd auto-make-passage/project`。

## Python 3.12 安装

不支持 Python 3.10。建议直接使用项目虚拟环境，所有命令均由 3.12 执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -i https://pypi.org/simple -e .
.venv/bin/python --version
```

## 已实现

- 14 个顶层工作流 Agent 角色（Lead、12 个 Specialist、Feedback）；从 Download Paper 到
  Experiment Code Run，以及最终的 Write Paper 与 Check Paper and Code，均已具备可审计实现；
- Lead 明确 `allowed_tools=()`，负责返回 `DispatchDecision` 和
  `LeadTransitionDecision`：前者派发任务，后者在策略层编译的合法候选中决定每次业务状态变化；
- Download Paper 使用 Tavily + DeepSeek 提取元数据，只下载通过公网 URL、MIME、体积、重定向与 PDF 签名检查的文件；
- Download Data 额外执行许可、格式、预计大小、JSON/Parquet/ZIP 安全检查；
- Make Idea 只读取带 SHA-256 的论文证据，输出强类型、可证伪且排序后的创新候选；
- Idea Design 使用 `deepseek-v4-pro` 共享 Provider（高推理强度），依次运行方法、代码复用、
  数据、实验和风险 5 个只读内部顾问；这些顾问不是独立状态节点或顶层工作流角色，再由
  Idea Design 主 Agent 汇总为方法组件、ADR 决策、代码链路、许可约束的 GitHub 模块复用、
  数据审计、统计预注册、阶段门、论文对比、预期结果和纯计算实验；
- Lead 观察 Specialist 结果、Feedback 控制、用户评价、工件类型、重试次数和修订历史后决定状态如何变化；Runtime 仍是唯一原子写入者，拒绝过期、篡改候选或越权迁移；
- 每个 Specialist 执行完成后进入 `AWAITING_USER_REVIEW`，默认不会一次跑完；
- 用户满意才接受阶段结果并继续；不满意则根据 gate 白名单回退，并将意见传入重试任务；
- 主流程固定为 `download-paper -> make-idea -> idea-design -> download-data -> code-make -> code-review -> server-schedule -> check-result -> experiment-code-run -> write-paper -> check-paper-and-code -> paper-review`；
- Server Schedule 只运行 `main_benchmark`；只有 Check Result 通过主指标门后，Experiment Code Run 才审计并运行所有非主实验与声明 seed；
- 补充实验缺口回到 Idea Design，实现缺口回到 Code Make，经 Code Review 和主结果门后再返回实验 Agent；运行/结果/补充实验循环不设次数上限，但每轮都由用户评价 Hook 控制；
- Write Paper 只引用已下载且重新验证签名/哈希的本地 PDF；公式必须绑定经 Code Review 的 Python AST 文件、符号、行号和摘要，实验数字只能由主实验与补充实验的持久化观测确定性生成；
- Check Paper and Code 会独立重建 PDF、AST、方法、Claim 与结果证据，逐字节重渲染 Markdown/LaTeX/BibTeX；任何篡改、虚构引用、公式不一致、结果不一致或越界结论都会回退到 Write Paper；
- Paper Review 以 NeurIPS/ICML 或 Bioinformatics/BIB 等权威审稿标准终审；只有
  `blocking=0` 且 `major=0` 才能通过，否则按问题类型回退并持续修改，不设置虚假的固定轮数上限；
- 论文质量门禁不绑定 DTA/DTI：所有监督学习主题都必须完成最近邻工作结构化比较、
  标识符与真实模型表征区分、重复输入冲突结果与测量语义审计、领域适配的泄漏检查、
  功效/精度与稳健性分析、主张—观测结果绑定和可复现归档；DTA 规则仅作为按主题触发的适配器；
- 每阶段后的 Feedback 独立观察并落盘全局进展、风险、后续动作和调控报告；
- 每个业务 Specialist 与随后执行的 Feedback 分别拥有一次用户评分，任一不满意都会按白名单回退；
- LangGraph 编译 18 个状态节点和 1 个条件路由守卫；Lead 选择的每个候选都必须通过实际图调用，Runtime 才能写入；
- Lead 决策、Specialist 任务和 Feedback 监控均写入按 Agent/运行隔离的 JSONL 过程日志；
- Pydantic `DispatchDecision`、`LeadTransitionOption`、`LeadTransitionDecision`、`TaskEnvelope`、`AgentResult`；
- SSRF、私网跳转、路径穿越、越权目录、符号链接及压缩包路径穿越防护；
- 事件日志、运行快照与 Agent 结果持久化；
- DeepSeek OpenAI 兼容客户端和 Tavily 搜索客户端；
- Lead 与核心 Specialist 均采用配置、提示词、策略、Schema、工具和 Skill 注册表分离的可审计包结构；
- 每个 Agent 注册 5–10 个可选 GitHub Skill 来源，但公开仓库不分发第三方正文或脚本；本地显式同步后的副本仍为 `downloaded_unreviewed`、`activated=false`，Runtime 不会执行；
- 离线端到端测试与真实服务冒烟命令。

## Agent 包结构

```text
project/agents/<agent>/
├── agent.py
├── policy.py
├── README.md
├── config/agent.json
├── prompts/system.md
├── schemas/*.schema.json
├── tools/                  # 有专用工具时
└── skills/
    └── registry.json       # 只记录来源、用途和已观察 commit
```

`agent.py` 会实际加载 `config/agent.json` 和 `prompts/system.md`。第三方 Skill 默认仅处于
`reserved` 状态，不随项目再分发。需要检查上游内容时，可显式同步为 Git 忽略的本地副本；
同步不会激活或执行脚本，也不会改写公开注册表：

```bash
.venv/bin/python scripts/sync_agent_skills.py --workers 4
```

更新 Pydantic JSON Schema：

```bash
.venv/bin/python scripts/export_agent_schemas.py
```

## 按用户评价分步运行

默认只执行一个 Agent，然后持久化快照并暂停：

```bash
.venv/bin/python -m research_agents demo \
  --topic "可复现科研多智能体工作流"
```

每个业务阶段先评价 Specialist，再单独评价 Feedback。查看当前待评价对象、队列数量、
结果摘要、Feedback 建议与合法回退目标：

```bash
.venv/bin/python -m research_agents status --run-id <run_id>
```

满意时，接受当次结果并且最多再执行一个 Agent；下一个 Agent 结束后会再次暂停：

```bash
.venv/bin/python -m research_agents review \
  --run-id <run_id> \
  --verdict satisfied \
  --rating 5 \
  --comment "当次结果满意，继续。"
```

不满意时必须提供可执行意见。不指定 `target-agent` 时默认重试当前 Agent；指定时必须在
`status` 返回的 `allowed_revision_targets` 中：

```bash
.venv/bin/python -m research_agents review \
  --run-id <run_id> \
  --verdict needs-revision \
  --rating 2 \
  --comment "论文质量不足，请补充权威会议开源 PDF 并重新排序。" \
  --target-agent download-paper-agent
```

只应用评价但不立即执行回退/下一个 Agent，可加 `--apply-only`。

只在单元测试或 CI 中显式跳过人工评价：

```bash
.venv/bin/python -m research_agents demo \
  --topic "CI workflow verification" \
  --auto-approve
```

运行单元与端到端测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## 过程日志与 Feedback 调控

日志独立放在 `project/logging/`，不写入任何 Agent 代码包：

```text
project/logging/
├── _workflow/<run_id>/timeline.jsonl
├── lead-agent/<run_id>/process.jsonl
├── download-paper-agent/<run_id>/process.jsonl
├── make-idea-agent/<run_id>/process.jsonl
├── idea-design-agent/<run_id>/process.jsonl
├── download-data-agent/<run_id>/process.jsonl
├── ...
└── feedback-agent/<run_id>/process.jsonl
```

每条记录包含时间、Agent、任务、事件、状态、工作流状态/版本、尝试次数、耗时和受限元数据。
Runtime 在 Lead 决策前后、每个 Specialist 任务前后/异常时以及 Feedback 监控前后记录事件，
用户评价 Hook 还会写入 `user_review_requested`、`user_evaluation_received` 和 Hook 异常事件，
并同步写入全局时间线。日志不保存 Prompt、用户意见正文、PDF 正文、数据行、运行结果正文或密钥；
凭据字段和常见 Token 形式会在落盘前脱敏。

Feedback Agent 每个 Specialist 阶段后读取最近的脱敏日志摘要，只能返回四种类型化建议：

- `continue`：继续工作流；
- `retry`：请求回到 Runtime 白名单允许的 Agent；
- `request_human`：转入 `NEEDS_HUMAN`；
- `abort`：终止并转入 `ABORTED`。

Feedback 不能派发任务或直接改状态。Runtime 是唯一控制执行者，会校验来源状态、
回退目标白名单和每个 Feedback gate 最多 3 次的预算；越权或超额都会失败关闭到
`NEEDS_HUMAN`。即使 Feedback 建议 `continue`，它也必须经过自己独立的用户评价门，不能代替用户批准。
详细合同见 `project/logging/README.md`。

## 测试 DeepSeek 与 Tavily

推荐使用根目录 Git 忽略的 `.env`、Shell 或密钥管理器，禁止把真实 Key 写入代码。
OpenAI 兼容变量优先，DeepSeek 别名仅作后向兼容：

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-v4-pro"
export OPENAI_REASONING_EFFORT="high"
export TAVILY_API_KEY="..."
export TAVILY_SEARCH_DEPTH="advanced"
.venv/bin/python -m research_agents smoke
```

也可通过隐藏式终端提示输入，Key 不落盘：

```bash
.venv/bin/python -m research_agents smoke --prompt-keys
```

## 启动前四个 Agent 链路

该命令会启动真实 provider，但只执行 Download Paper 就暂停。之后每次通过 `review`
批准一步，再分别进入 Make Idea、Idea Design 和 Download Data：

```bash
.venv/bin/python -m research_agents run-first-four \
  --topic "你的研究主题" \
  --server-backend disabled \
  --prompt-keys
```

如果评价已经原子提交，但下一 Agent 因终端或网络中断没有完成，可以从同一个 run 安全续跑
一个业务步骤；该命令不会跳过用户评价门：

```bash
.venv/bin/python -m research_agents continue-live \
  --run-id <run_id> \
  --review-provider deepseek \
  --server-backend local
```

`NEEDS_HUMAN` 是另一类显式断点，必须说明原因后选择重试当前阶段或接受已有成功证据继续：

```bash
.venv/bin/python -m research_agents resolve-human \
  --run-id <run_id> \
  --action retry \
  --reason "已修复阻断条件，重试当前 Agent" \
  --server-backend local
```

主要产物：

- `reference/<project_slug>/papers/`：通过验证的 PDF；
- `reference/<project_slug>/manifests/`：论文来源、状态、哈希和拒绝原因；
- `data/<project_slug>/raw/`：通过许可与安全检查的数据文件；
- `data/<project_slug>/manifests/` 与 `reports/`：实际字节、行数、实体、分布与缺失报告；
- `design/<project_slug>/ideas/`：强类型研究方案或证据缺口报告；
- `design/<project_slug>/research-designs/`：代码、数据、实验和论文比较的完整研究设计；
- `code/<project_slug>/`：Code Make 生成并等待 Code Review 的科研代码。
- `project/paper/<project_slug>/`：证据约束的 Markdown、LaTeX、BibTeX 与论文 Manifest；
- `project/reviews/consistency/<project_slug>/`：论文、原始 PDF、代码、公式与实验结果的一致性复核报告。

`project/` 只保存可复用 Agent/Runtime 代码、运行状态和脱敏日志。启动新研究项目时传入
`--project <slug>`；省略时会生成唯一的 `research-<id>`，不会与其他项目共用目录。

## DTA 100 篇论文语料

重新生成药物–靶点/蛋白质–配体亲和力预测文献集，并尝试下载合规开放 PDF：

```bash
.venv/bin/python \
  agents/download_paper_agent/tools/collect_quality_dta_corpus.py \
  --workers 8
```

结果位于 `reference/assaycal-dta/literature/`：

- `papers.json`、`papers.csv`、`records/`：100 篇去重论文的完整元数据、质量 bucket 与摘要；
- `references.bib`：BibTeX 引用；
- `innovation_points.md`：逐篇创新概括和摘要证据句；
- `papers/`、`pdf_downloads.json`：100 份实际 PDF 及逐篇 SHA-256 下载审计；
- `github_repositories.md`：强匹配 GitHub 仓库、验证状态和 SPDX 许可证；
- `corpus_manifest.json`：70/30 组合、查询、覆盖率、证据级别和文件校验和。

质量组合严格为 70 篇权威/领域主流会议期刊、30 篇 2024–2026 近期论文；其中 DTA 与
structure-based PLBA 各 50 篇。只有 PDF 真实落盘并通过 `%PDF-` 签名后，论文才进入最终清单。
正式出版源优先，Europe PMC 与 arXiv 只作为合法全文镜像或末级补充。

离线复核论文数、DOI/标题去重、分层比例、逐篇创新证据、引用条目及 PDF
签名/哈希：

```bash
.venv/bin/python scripts/validate_dta_corpus.py
```

用真实 Agent 合同调用本地 100 篇语料，依次运行 Make Idea 与多顾问 Idea Design，并在
Idea Design 结束后停在 `AWAITING_USER_REVIEW`：

```bash
.venv/bin/python scripts/run_dta_idea_design.py
```

该离线命令明确不调用或冒充 DeepSeek，适合在无密钥环境验证合同。当前推荐方案为
AssayCal-DTA，当前产物位于 `design/assaycal-dta/ideas/64ec08851757.json` 和
`design/assaycal-dta/research-designs/734f7b3ed678.{json,md}`。辅助 Agent 只记录阶段、耗时、建议数与
issue code，不记录提示词和报告正文。

当前数据阶段已下载 Davis、KIBA 与 BindingDB 202608 staff-curated articles 子集。压缩总大小
20,115,371 bytes，数据本体 445,500,316 bytes，共 241,851 行。三个数据文件已持久化到
`data/assaycal-dta/datasets/`；Manifest 和数据规模/分布报告位于
`data/assaycal-dta/manifests/900a4f355753.json` 与
`data/assaycal-dta/reports/900a4f355753.md`。预计超过 10 GiB 的候选不会在本地自动下载，只在
`data/<project_slug>/scripts/` 生成服务器 Shell 脚本并等待人工执行。

以上 `assaycal-dta` 说明记录的是较新的数据准备工作，不等同于仓库展示页中的固定论文
快照。展示工件 `dta-live-pipeline-check-v2/strictcoldcal_ieee` 只执行了 Davis 本地 CPU pilot；
其中 KIBA 与 Metz 已物化但未执行，因此论文不把它们写成实验结果。

使用已轮换并由环境变量注入的 DeepSeek Key 对同一语料在线重跑：

```bash
export DEEPSEEK_API_KEY="..."
.venv/bin/python scripts/run_dta_idea_design.py --deepseek
```

旧版五候选启发式报告仍可通过以下命令重新生成：

```bash
.venv/bin/python \
  agents/make_idea_agent/tools/synthesize_dta_ideas.py
```

推荐方案为 AU-DTA：测定语境感知、不确定性校准的多模态 DTA，并配套严格 OOD
评测。报告位于 `reference/assaycal-dta/literature/research_ideas.md`，机器可读版本位于
`design/assaycal-dta/ideas/dta_research_ideas_v1.json`。

面向人工审阅的固定命名 Idea Design 文档为
`design/assaycal-dta/assaycal-dta-idea-design-v1.md`，包含方法组件、代码链路、GitHub 模块复用、Davis/KIBA/BindingDB
数据方案、纯计算实验、对比基线、预期结果和失败阈值。

## 实验性 Live 全流程

```bash
.venv/bin/python -m research_agents run-live \
  --topic "你的研究主题" \
  --review-provider deepseek \
  --server-backend disabled \
  --prompt-keys
```

Live 模式默认只产生当前 Agent 所需 API 调用，然后等待用户评价，不会一次跑到末尾。
无法验证的链接只会登记为
`discovered` 或 `requires_review`，不会被冒充为已下载文件。服务器执行和论文文件写入仍是后续阶段。

## Server Schedule、Check Result 与 Experiment Code Run

真实服务器仅通过 Git 忽略的 `project/configs/servers.local.json` 启用。先复制
`project/configs/servers.example.json` 并填写 `credential_prefix`、专用远端根目录、项目训练 argv
及主指标阈值；配置禁止出现主机密码、Token、私钥路径等凭据字段。SSH 主机、端口、
用户名与密码/私钥只写入 `project/agents/server_schedule_agent/.env`：

```bash
cp agents/server_schedule_agent/.env.example agents/server_schedule_agent/.env
chmod 600 agents/server_schedule_agent/.env
```

Agent 不会从主体 `.env` 或进程环境回退读取 SSH 凭据；密码/私钥口令只进入 SSH/rsync
子进程的临时 askpass 环境，不进入 argv、Manifest 或日志。`.env` 必须是非符号链接的
`600` 常规文件，私钥也必须禁止 group/world 访问。SSH 强制严格 host-key 校验，
所以经独立核验的服务器指纹需事先存在 `known_hosts`。只有 `execution_enabled: true` 的
服务器才可进入调度。

Server Schedule 会在启动前检查 CPU load、主机可用内存、专用目录磁盘空间、Conda、GPU
利用率和显存，然后以预计最短 makespan 只分配 `main_benchmark` 的独立 seed。远端目录固定为
`<workspace_root>/projects/<project_slug>/runs/<run_id>/<attempt>/`；先传输 Manifest 验证过的
数据、代码与模型源码，再创建本次尝试专属的新 Conda 环境、安装本地代码并启动 GPU 任务。
安装后会在新环境执行 PyTorch CUDA 可见性预检，然后在每台入选服务器上前台运行 Code Make 已审核的
`minimal_demo`，并受 `server_demo_timeout_seconds` 限制。任一 demo 失败/超时时不启动任何长任务。
全部通过后，训练以 `nohup` 脱离 SSH 连接，持久化 PID/status、bootstrap log、job log 和 JSONL 指标。
不同 GPU 并行工作，同一 GPU 的多个 seed
通过独立文件锁排队，避免显存过量并发。所有训练命令均以 argv 交给受管 Python Runner，禁止 Shell 拼接。任何目录删除命令都被禁止，
运行目录会完整保留。

每个 job 有独立日志、JSONL 指标、`status.json` 与 `best/` checkpoint 目录。Check Result 在
预计完成时间检查，并保证最长每 43,200 秒生成一次用户报告。仍在运行时返回
`in_progress`，不计为失败且不推进论文阶段；完成后按所有 seed 的每-job 最优指标均值验证
阈值。服务器、数据、代码或指标问题分别回退到 Runtime 白名单中的最小修复阶段。
服务器/结果循环没有次数上限，但每轮都持久化并经过用户评价钩子。

`next_report_at` 会注册成 `project/var/runs/monitoring/index.json` 中的持久化任务。两次用户
评价都满意后，CLI 自动启动本地后台守护进程；它每 1–60 秒检查到期任务，到期才唤醒
Check Result 或 Experiment Code Run，并再次停在双评价门。守护进程不会保存任何 API/SSH
凭据，任务历史只标记为 `fired/failed/cancelled`，不会删除。状态与单次手动巡检：

```bash
.venv/bin/python -m research_agents monitor-status
.venv/bin/python -m research_agents monitor-daemon
```

Experiment Code Run 只接受 `verdict=passed` 的主结果。它先以结构化
`ExperimentCoverageDraft` 审计论文主张是否还缺纯计算实验；缺实验时使用固定
`experiment-gap-*` ID 回到 Idea Design，新 Research Design 必须包含同 ID 的
`ExperimentPlan` 与 `ExpectedResult`。然后交给 Download Data、Code Make、Code Review、
Server Schedule 和 Check Result 重新过门，不允许未设计/未审核实验直接上服务器。

审计通过后，Agent 复用通过主实验的不可变代码、数据、Conda 环境与服务器工作区，
重新探测当前 GPU，每台服务器先前台运行 `minimal_demo`，再用现有受管 `nohup`
Runner 启动所有非主实验/seed。它在预计完成时间或最长 43,200 秒间隔读取真实
JSONL，持久化每个 job 的状态、日志、最佳 checkpoint 和有限主指标，并确定性计算
每个实验的 mean、sample SD、min 和 max。缺 job、seed 或指标都是失败，
DeepSeek 不生成、补齐或修改任何实验数值。

## Claude Agent SDK 与 LangGraph 边界

项目已实际依赖并导入 `claude-agent-sdk==0.2.135`。选择
`--review-provider claude-sdk` 时，Paper Review 与 Feedback 通过真实 `query()`、
`ClaudeAgentOptions` 和结构化输出运行；默认模型是 `Codex-opus-4-6` 且使用 adaptive
thinking。SDK 会固定项目 cwd、turn/费用上限和空的只读工具白名单，并显式禁止
Write、Edit、Bash、WebSearch、WebFetch 与 Agent，因此不能修改论文、执行 Shell 或改变状态。

DeepSeek 仍是默认 Provider。Claude SDK 与 DeepSeek 不能互相冒充：只在显式选择时构造
SDK adapter。`langgraph==1.2.11` 编译并实际调用条件路由守卫。StateMachine 只生成完整、
不可篡改的候选快照；Lead 结合全局上下文选择 advance、retry、wait、request-human、
complete 或 abort；Runtime 校验版本、选项完整性与 LangGraph edge 后才原子写入快照。
可审计当前图：

```bash
.venv/bin/python -m research_agents graph
```

## 密钥安全

- 主模型按 `OPENAI_API_KEY/BASE_URL/MODEL` 读取，缺失时才回退到 `DEEPSEEK_*`；
- Tavily 默认 `search_depth=advanced`，论文、GitHub 与数据查询分别使用权威域名白名单；
- `ANTHROPIC_API_KEY` 只在显式选择 Claude Agent SDK 审稿/反馈时使用；
- `.env` 已被 Git 忽略，但仍建议使用系统密钥管理器；
- 日志、事件、快照和异常输出不会记录 Key；
- 已在聊天、截图或日志中暴露的 Key 应立即轮换。

验证每个 Agent 的 Skill 注册数量、来源和保留状态（每个必须 5–10 项）：

```bash
.venv/bin/python -m research_agents skills
```

公开设计说明见 [系统架构](../docs/architecture.md)、[状态机](../docs/state-machine.md)、
[安全模型](../docs/security-model.md) 与 [项目展示](../docs/showcase.md)。
