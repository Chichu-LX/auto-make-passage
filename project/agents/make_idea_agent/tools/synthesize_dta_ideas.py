#!/usr/bin/env python3
"""Create evidence-linked DTA research ideas from the local 100-paper corpus."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
CORPUS_PATH = WORKSPACE_ROOT / "reference" / "assaycal-dta" / "literature" / "papers.json"
OUTPUT_JSON = (
    WORKSPACE_ROOT / "project" / "artifacts" / "ideas" / "dta_research_ideas_v1.json"
)
OUTPUT_MD = WORKSPACE_ROOT / "reference" / "assaycal-dta" / "literature" / "research_ideas.md"


def find_evidence(
    papers: list[dict[str, Any]], phrases: list[str], *, limit: int = 5
) -> list[dict[str, Any]]:
    scored: list[tuple[int, dict[str, Any]]] = []
    for paper in papers:
        title = paper["title"].lower()
        abstract = paper.get("abstract", "").lower()
        score = sum(4 for phrase in phrases if phrase in title) + sum(
            1 for phrase in phrases if phrase in abstract
        )
        if score:
            scored.append((score, paper))
    scored.sort(
        key=lambda item: (item[0], item[1].get("cited_by_count", 0)), reverse=True
    )
    return [
        {
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "doi": paper.get("doi"),
            "innovation_basis": paper["innovation_basis"],
        }
        for _, paper in scored[:limit]
    ]


def build_ideas(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    idea_specs = [
        {
            "idea_id": "IDEA-DTA-01",
            "priority": 1,
            "title": "AU-DTA：测定语境感知且具不确定性校准的多模态亲和力模型",
            "problem": (
                "现有 DTA 模型通常把 Kd、Ki、IC50 及不同实验条件转换成单一 pAffinity 标签，"
                "忽略测量类型、条件差异、重复实验方差和删失值，导致跨数据集迁移及置信度失真。"
            ),
            "hypothesis": (
                "显式编码 assay 类型与条件，并同时学习绝对亲和力、同批次相对排序和异方差"
                "预测分布，可在冷药物、冷靶点及跨数据集测试中提升准确性与校准度。"
            ),
            "method": [
                "药物分支融合分子图编码器与 SMILES 预训练模型；蛋白分支采用序列预训练模型。",
                "加入 assay-context token：Kd/Ki/IC50、温度、pH、来源实验室和测量上下界。",
                "以删失高斯/Student-t 似然建模 aleatoric uncertainty，以深度集成估计 epistemic uncertainty。",
                "联合优化绝对 pAffinity 回归与同 assay 内 pairwise ranking，降低单位和批次偏差。",
                "结构可用时启用 3D 交互分支；缺失结构时用门控机制退化到序列/2D 分支。",
            ],
            "datasets": [
                "BindingDB 原始测定记录（保留 assay 类型、单位、关系符和条件字段）",
                "Davis 与 KIBA（用于和经典 DeepDTA/GraphDTA 工作可比）",
                "PDBbind（用于结构可用子集和跨模态实验）",
            ],
            "evaluation": [
                "random、cold-drug、Bemis–Murcko scaffold、cold-target、protein-family 和 temporal split",
                "RMSE、MAE、Concordance Index、Spearman；NLL、CRPS、ECE 与 90% 区间覆盖率",
                "报告 in-domain/OOD 性能差及选择性预测 risk–coverage 曲线",
            ],
            "ablations": [
                "移除 assay-context token",
                "移除 ranking loss",
                "同方差对比异方差似然",
                "仅序列、仅结构及自适应缺失模态门控",
                "随机切分与严格 OOD 切分的性能差异",
            ],
            "risks": [
                "BindingDB 条件字段缺失较多：先定义字段覆盖率门槛并加入 missingness mask。",
                "IC50 不等价于 Kd/Ki：不做无条件合并，保留测定类型并进行分层敏感性分析。",
                "模型复杂度过高：先实现序列/2D 最小版本，再加入可选 3D 分支。",
            ],
            "evidence": find_evidence(
                papers,
                [
                    "experimental uncertainty",
                    "data quality",
                    "uncertainty",
                    "multimodal contrastive",
                    "bioassay pre-training",
                    "generalizability",
                ],
            ),
        },
        {
            "idea_id": "IDEA-DTA-02",
            "priority": 2,
            "title": "DTA-OODBench：去泄漏的多轴外推评测与置信度基准",
            "problem": "随机拆分会泄漏相似骨架和同源蛋白，掩盖模型在新化学空间和新靶点上的失败。",
            "hypothesis": "统一 scaffold、target-family、pair-cluster 与 temporal 切分，可显著改变模型排名并揭示真实泛化瓶颈。",
            "method": [
                "基于药物骨架、序列同源性和 pair 相似度构造四类互斥 OOD split。",
                "对所有 split 固化哈希、去重规则和相似度阈值，提供泄漏审计报告。",
                "同时评估点预测、排序和不确定性校准，输出模型适用域。",
            ],
            "datasets": ["Davis", "KIBA", "BindingDB 时间快照"],
            "evaluation": ["RMSE/CI/Spearman", "OOD gap", "校准误差", "split 泄漏率"],
            "ablations": ["不同同源性阈值", "是否去除重复 pair", "随机切分对比时间切分"],
            "risks": ["严格 split 后样本过少：发布多难度阈值并报告置信区间。"],
            "evidence": find_evidence(
                papers,
                ["generalizability", "homology", "data scarcity", "benchmark", "cold-start"],
            ),
        },
        {
            "idea_id": "IDEA-DTA-03",
            "priority": 3,
            "title": "结构缺失自适应的序列→口袋→复合物级联 DTA",
            "problem": "3D 模型依赖可靠复合物结构，而大规模 DTA 数据通常只有序列与 SMILES。",
            "hypothesis": "用结构教师蒸馏口袋交互知识，并按结构置信度门控，可兼顾规模、准确率和缺失模态鲁棒性。",
            "method": [
                "序列/2D 学生覆盖全部样本，预测结构或实验结构分支作为教师。",
                "以口袋残基–原子接触图和亲和力分布共同蒸馏。",
                "根据 pLDDT、口袋置信度和 docking ensemble 方差进行动态门控。",
            ],
            "datasets": ["BindingDB", "PDBbind", "AlphaFold/公开结构映射"],
            "evaluation": ["结构覆盖率分层 RMSE", "缺失结构鲁棒性", "推理成本"],
            "ablations": ["无蒸馏", "硬门控对比软门控", "实验结构对比预测结构"],
            "risks": ["结构误差可能传递：将结构置信度作为显式输入并做拒绝预测。"],
            "evidence": find_evidence(
                papers,
                ["pretrained transformers", "folding-docking-affinity", "synthetic structural data", "point cloud", "structure-aware"],
            ),
        },
        {
            "idea_id": "IDEA-DTA-04",
            "priority": 4,
            "title": "构象/对接姿势分布上的多实例物理约束亲和力预测",
            "problem": "单一 docking pose 会把构象不确定性当成确定输入，并可能放大错误姿势。",
            "hypothesis": "对 pose ensemble 做能量感知的多实例聚合，比单姿势 3D GNN 更稳定且更易校准。",
            "method": [
                "为每个 pair 生成受限数量的 ligand conformer 与 docking pose。",
                "用 SE(3) 等变交互网络编码实例，以物理能量和几何冲突作为先验权重。",
                "输出 pose-level 与 pair-level 不确定性并识别错误姿势。",
            ],
            "datasets": ["PDBbind refined/general set", "具有多构象或 MD 轨迹的公开子集"],
            "evaluation": ["RMSE/MAE", "跨 pose 稳定性", "置信区间覆盖率", "计算成本"],
            "ablations": ["单 pose", "平均池化", "无物理先验", "不同 pose 数量"],
            "risks": ["计算量较高：预计算 pose，并以早停和 top-k 门控控制成本。"],
            "evidence": find_evidence(
                papers,
                ["multi-instance", "docking structures", "spatio-temporal", "molecular dynamics", "water network", "equivariant"],
            ),
        },
        {
            "idea_id": "IDEA-DTA-05",
            "priority": 5,
            "title": "跨数据集亲和力语义对齐：绝对值、排序与测定类型的联合学习",
            "problem": "Davis、KIBA、BindingDB 和 PDBbind 的标签定义与分布不同，直接拼接会造成负迁移。",
            "hypothesis": "共享表征配合数据集特定校准头和统一排序目标，可改善跨域迁移而不强迫标签完全同尺度。",
            "method": [
                "共享药物/蛋白编码器，每个数据集使用单调校准头。",
                "同域使用绝对回归，跨域使用一致性和 pairwise ranking。",
                "用 domain-adversarial 或 optimal-transport 正则对齐表征，同时保留 assay token。",
            ],
            "datasets": ["Davis", "KIBA", "BindingDB", "PDBbind"],
            "evaluation": ["leave-one-dataset-out transfer", "zero-shot target family", "校准斜率"],
            "ablations": ["直接合并", "无数据集特定头", "无 ranking", "无域对齐"],
            "risks": ["域对齐可能抹去有用差异：使用条件式对齐并检查各测定类型性能。"],
            "evidence": find_evidence(
                papers,
                ["multi-task", "bioassay", "data quality", "fusion", "transfer", "contrastive"],
            ),
        },
    ]
    return idea_specs


def write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# DTA 研究想法（基于本地 100 篇论文）",
        "",
        "## 推荐结论",
        "",
        "优先建议 `IDEA-DTA-01`，并把 `IDEA-DTA-02` 的严格 OOD split 作为其评测协议。"
        "该组合的创新重点不是再叠加一种编码器，而是修正 DTA 中被普遍忽略的标签语义、"
        "实验不确定性和评测泄漏问题，可验证性更强。",
        "",
    ]
    for idea in payload["ideas"]:
        lines.extend(
            [
                f"## {idea['idea_id']} · {idea['title']}",
                "",
                f"**问题：** {idea['problem']}",
                "",
                f"**假设：** {idea['hypothesis']}",
                "",
                "### 方法",
                "",
                *[f"- {item}" for item in idea["method"]],
                "",
                "### 数据与评测",
                "",
                *[f"- 数据：{item}" for item in idea["datasets"]],
                *[f"- 评测：{item}" for item in idea["evaluation"]],
                "",
                "### 消融与风险",
                "",
                *[f"- 消融：{item}" for item in idea["ablations"]],
                *[f"- 风险：{item}" for item in idea["risks"]],
                "",
                "### 论文证据",
                "",
                *[
                    f"- `{item['paper_id']}` {item['title']}"
                    + (f" — https://doi.org/{item['doi']}" if item.get("doi") else "")
                    for item in idea["evidence"]
                ],
                "",
            ]
        )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    papers = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    if len(papers) != 100:
        raise SystemExit("DTA corpus must contain exactly 100 papers")
    payload = {
        "schema_version": "1.0",
        "topic": "drug-target / protein-ligand binding affinity prediction",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_uri": "reference/assaycal-dta/literature/papers.json",
        "corpus_paper_count": len(papers),
        "recommended_idea_id": "IDEA-DTA-01",
        "recommended_evaluation_companion": "IDEA-DTA-02",
        "ideas": build_ideas(papers),
        "limitations": [
            "Ideas are grounded in indexed titles and abstracts; full-text novelty review is still required.",
            "Dataset licenses and assay-field coverage must be verified before implementation.",
            "No experimental performance is claimed at the ideation stage.",
        ],
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload)
    print(json.dumps({"ideas": len(payload["ideas"]), "recommended": payload["recommended_idea_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
