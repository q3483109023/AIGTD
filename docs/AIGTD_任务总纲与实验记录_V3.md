# AIGTD 期末课程项目：任务总纲、已完成实验记录与交付计划（V3）

> 更新日期：2026-11-22  
> 项目：Detect AI-Generated Sentences for Human-AI Hybrid Articles  
> 项目目录：`/home/zhuangyr/dl-workspace/projects/AIGTD_project_starter/AIGTD`  
> GitHub：`q3483109023/AIGTD`  
> **本文角色：统一的项目任务总纲、实验台账及当前交付依据。** 本文件在原 V2 上修订，保留研究背景、方法候选、已发现的问题；原有《AIGTD_project_handover_compact_2026-11-21(1).md》仍是更早的技术交接记录。  
> 证据等级：**老师 PDF 明示要求**、**历史实验记录**、**GitHub CI 成功记录**、**2026-11-22 用户服务器实测**、**尚待验证/交付**严格区分。**第 12 节为正式实验记录台账**，严禁把开发验证成绩写成无标签测试成绩。

---

## 0. 项目决策摘要：现在要做什么、不做什么

**项目总目标：** 交付一个能够对人机混合文章逐句检测 Human / Machine 的可运行系统，使用合规且可复现的验证证明其性能，并通过**一个明确的问题、一组有说服力的统计分析和一项有对照的改进实验**呈现相对于普通分类作业的独特思考。

**当前决策（2026-11-22）：阶段一的五折 OOF Stacking 已完成完整无标签测试推理与通用结构验收；E2「基于长句分歧的选择性上下文」已完成配对 OOF 实验、五折稳定性检验、来源分组 Bootstrap 和固定规则 Novel 复核。暂停新增训练/调参，转入课程报告、关键图表、代码交付与正式提交格式核实。**

**阶段性结果（两种不同协议，不可直接排名）：** Stacking 在 *Fold-matched Novel* 的 Macro-F1 = **0.9174**（研究候选与已冻结工程推理基线）；E2 在 *严格窗口清理的配对 OOF* 上 Context **0.8998 → 0.9067**，并在重复使用的 Novel 两种预测协议下继续观察到同协议收益。E2 **没有替换**最终 Stacking 的测试预测。

### 成功的三个层次与现状

| 层次 | 必须回答的问题 | 产物 |
|---|---|---|
| A. 课程任务完成（不可缺） | 给定一篇文章，能否按原顺序给每句输出 Human / Machine？ | **推理与通用格式验收已通过；正式提交 schema、完整课程报告尚待确认** |
| B. 有说服力的实验（优先保障） | 哪些方法有增益，在哪些样本上有效或失效？差异是否来自公平比较？ | **同协议比较、配对统计和 Bootstrap 已完成；课程展示图与写作尚待整理** |
| C. 有依据的“小创新”（在 A、B 稳固后做） | 能否利用上下文/风格/长度信息，在不增加明显风险的情况下改善判别？ | **E2 长句 Human-veto 已完成 OOF 与 Novel 开发复核，形成可复核的小改进；不是盲测证明** |

**项目不是**：复现老师给出的全部论文、追求极复杂架构、不断搜索验证集最优阈值、因单折异常无限延长诊断、以接近 1.0 的泄漏指标包装成创新。

---

## 1. 老师实际提出了什么要求

### 1.1 PDF 明示的任务与评价（原始要求）

资料：《1. 数据挖掘与知识发现-课程project(2).pdf》：

- **第 2–3 页：** 输入由多句组成的人机混合文章，识别每句话是人工编写，还是 **GPT-3.5-turbo** 生成；检测粒度为**句子**，不是给整篇文章赋一个标签。
- **第 4 页：** 展示训练数据的文章 `id`、`sent_and_label`、`domain` 等结构。模型必须能保持输入文章和句子顺序。
- **第 5 页：** 评价指标为 **Accuracy、Precision、Recall、Macro-F1、Micro-F1**；展示结果保留小数点后 **4 位**。PDF 未明确 Precision/Recall 的平均方式，正式报告需明确机器类、人工类和宏平均的定义。
- **第 6 页：** LSTM/GloVe、RoBERTa 序列模型、BERT+LLM+SVM 等为**方法示例**，并非必须复现的清单。
- **第 7–9 页：** 鼓励通过统计与实验“说点不一样”：比如类间句长分布差异、阈值重叠和错误来源、方法比较及可解释结论。图中的发现和成绩属于老师展示的示例，**不能直接作为本项目已验证的规律或同协议排行榜**。
- **第 10 页：** 提供 AIGTD 综述链接作为参考。

### 1.2 PDF 没有明确的内容（不要擅自当作硬约束）

该 PDF 未明确规定：精确提交截止时间、指定测试预测文件格式/命名、代码仓库提交方式、报告/PPT 具体篇幅、评分权重、是否必须实现论文创新模块。这些应以老师其他正式通知为准；在信息缺失时做**易检查的工程交付**，但不声称是 PDF 的明示要求。

### 1.3 本项目采用的工程性补充交付（建议）

1. **源码与运行说明：** README、依赖、配置、随机种子、数据准备、训练与推理命令。
2. **方法与实验报告：** 任务与数据、评估协议、基线、特色观察、改进与消融、误差/风险分析、局限。
3. **模型和推理协议：** 明确最后选择的 checkpoint、阈值、tokenizer、Fold/归一化策略；大模型权重以本地保留/获取说明方式管理。
4. **测试预测：** 对无标签测试数据输出逐句结果，进行条数、顺序、标签取值、重复键检查；按老师要求的 schema 最终落盘。

> **完成底线：** 即使改进实验不显著优于基线，也必须按时形成完整、可复现、诚实说明限制的课程交付。

---

## 2. 数据及不可破坏的评估约束

### 2.1 数据口径（来自原交接报告，待最终代码/文件复核）

| 数据 | 文章记录数 | 句子数 | 用途 |
|---|---:|---:|---|
| 原始训练 JSON | 5,000 | 66,471 | 有标签数据来源 |
| Train split | 3,983 | 52,864 | 模型训练与 OOF |
| Full Validation | 1,017 | 13,607 | 已使用过的开发验证资源 |
| Novel Validation | 1,017 个 article_key / 619 个来源 ID | 11,682 | Full Validation 子集，**已反复用于开发** |
| 无标签 Test | 500 | 4,264 | 最终预测；不计算真实 Accuracy/F1 |

原始 5,000 篇有标签训练来源的类别统计：Human = 0（22,418），Machine = 1（44,053），Machine 占约 66.27%；**这一统计对应 66,471 句原始来源，不应误写成 52,864 句 OOF 训练划分类别分布。**数据记录可能含 `domain`（如 academic/news）；需要分域审计，但不能据此默认获得真实泛化能力。

### 2.2 数据键与输入输出约束

- `article_id` / 来源 ID 不保证唯一对应一条文章记录；`article_key` 唯一标识文章**记录**，句子行中可重复。
- `sentence_index` 用于恢复文章内顺序；`oof_row_id` / `novel_row_id` 用于**句子级预测对齐**。
- 训练 JSON 使用 `sent_and_label`；测试 JSON 使用 `sent`；无标签测试输出必须保留文章顺序和句子顺序。
- 标签映射固定为 `human=0`、`machine=1`；任何新脚本须验证映射一致。
- 训练/验证按来源分组，同一来源组不跨折；折内对重复的规范化文本进行 purge。Context 窗口训练需检查 previous/target/next **任一组件**的重叠，而非只检查目标句。
- 标签冲突、重复文本不自动删除或重标；相同来源 ID 不等于精确文本泄漏。

### 2.3 三个评估层次，不能混成一个排行榜

1. **Pooled OOF：** 每个训练样本只由未见其标签的基础模型产生 OOF 预测；用于开发内部比较，需检查 Fold 分数尺度、source 分组和 purge。
2. **Fold-matched Novel：** 既有开发验证集，曾参与早停、阈值选择、融合开发；只能称**开发结果**，不得称独立测试分数。
3. **无标签 Test：** 用于按指定格式推理/提交；没有真实标签，不得报告 Accuracy/F1。后续若老师提供独立评测分数，才可单列。

补充：对融合器/门控器使用 GroupKFold 只避免其直接训练行与验证行的文章重叠；**不自动保证整个基础模型训练→OOF 特征→元模型训练流水线达到严格嵌套外层验证**。所有未做外层独立验证的性能结果应标识为开发评估。

---

## 3. 现有工作：哪些已经完成，哪些只是候选

### 3.1 工程与实验最新完成情况（2026-11-22）

| 模块 | 已有工作 | 状态/限制 |
|---|---|---|
| 数据工程 | JSON 解析、文章/句子键、split、Novel 标记、数据检查 | **已完成；500 篇通用输出结构/文本/顺序校验 PASS；官方提交字段待核对** |
| 数据完整性 | source 分组、exact-text purge、Context 组件级独立审计、重复/冲突分析 | 已有通过记录；不等于标签完全正确 |
| 基础模型 | TF-IDF + LinearSVC、RoBERTa Sentence-only | 已训练并有开发验证预测 |
| 集成 | 五折 Purged OOF；Logistic Stacking、SVM Veto | **Stacking 修正版聚合统计本地 PASS、CI PASS；保存的元模型 11,682 句复现 PASS** |
| Context 实验 | Sentence-matched RoBERTa 与前/目标/后句 Context RoBERTa 配对五折 | **已运行 52,864 条配对 OOF；Fold 4 退化已描述，随后完成 E2 固定规则改善实验** |
| Context 目标句保留率 | 近期对 tokenizer/截断做过审计 | 此前分析显示目标句基本完整保留；将审计输出归档，不再把“目标被截断”当主解释 |
| Fold 4 伤害模式 | 分层、文章聚集、Brier/log-loss 变化、GroupKFold 风险预测 | **分层统计可描述；原风险预测存在目标泄漏，不能用于证明门控泛化** |
| 最终测试推理 | Stacking 单折匹配、五折 SVM+RoBERTa、Meta Logistic，阈值 0.585 | **500 篇/4,264 句 CUDA 推理 PASS；五折均调用；通用 JSON/CSV 校验 PASS；老师正式 schema 待确认** |
| E2 选择性上下文 | 20 词长句 + Sentence/Context 分歧触发 Human-veto | **52,864 OOF + 11,682 Novel 两种协议复核均 PASS；开发证据，不是无标签 Test 的最终模型** |

### 3.2 已有最重要的模型结果（严格注明协议）

**Fold-matched Novel，11,682 句，开发验证（Stacking 修正版聚合统计已复核 PASS）：**

| 模型 | Accuracy | Macro-F1 | 说明 |
|---|---:|---:|---|
| SVM | 0.8857 | 0.8800 | 旧 OOF/匹配协议 |
| RoBERTa | 0.9147 | 0.9093 | 同上 |
| **Stacking** | **0.9217** | **0.9174** | **阶段一已冻结、已完成全量测试推理的工程模型** |
| SVM Veto | 0.9191 | 0.9144 | 简单规则候选 |

**Sentence-matched vs Context，Pooled OOF、固定阈值 0.5：** Sentence Macro-F1 0.8760；Context 0.8998，增益 **约 +2.38 个百分点**。在同协议配对 OOF 上实施 E2 后提升至 0.9067，详情见第 12.7–12.8 节。但 Context 只在 5 折中的 3 折提高 Macro-F1，Fold 4 AUC/AP 明显下降。此处 Sentence-matched 的训练子集和前述 Stacking 所依赖的 RoBERTa 不是相同训练协议，**不得直接用两张表的数字宣称 Context 或 Stacking 的绝对优势**。

**Fold 4，10,579 句，实测：**

| 指标 | Sentence-matched | Context |
|---|---:|---:|
| Accuracy | 0.8905 | 0.8910 |
| Macro-F1 | 0.8682 | 0.8668 |
| ROC-AUC | 0.9664 | 0.9129 |
| AP | 0.9824 | 0.9004 |
| Human→Machine 错误（FP） | 1,015 | 1,091 |
| Machine→Human 错误（FN） | 143 | 62 |

**解释：** Context 在 Fold 4 多纠正了 Machine，却增加了更多 Human 的误判；因此 Accuracy 小幅提高不代表 Macro-F1、排名性能或风险质量提高。

### 3.3 Fold 4 深入诊断的新证据（本次服务器运行，代码 PR #4）

来自 `results/context_harm_patterns/`，采用两模型相同 `oof_row_id` 配对：

- 9,051 条两模型都对；783 条都错；375 条仅 Context 正确；370 条仅 Sentence 正确。
- Context **伤害**的 370 条中 Human 343（92.7%）、Machine 27；**帮助**的 375 条中 Human 267、Machine 108。
- **Human 伤害率 9.67%，帮助率 7.53%；Machine 伤害率 0.38%，帮助率 1.54%。** 这给出了明显的类别非对称性证据。
- 目标长度 `<=70` 字符：帮助 168、伤害 110，净帮助 58；`>130` 字符：帮助 38、伤害 95，净伤害 57。**这是关联，不能直接当因果机制。**
- 按 `article_key`，伤害数量最高的前 10% 文章组覆盖全部伤害的约 55.95%；尚未对文章长度和 Human 占比做受控置换检验，不得声称显著文章聚集。
- 在 370 条 Context 伤害样本上，Context−Sentence 平均 Brier 增量约 **+0.921**，平均 Log-loss 增量约 **+5.791**；相反在 375 条帮助样本上约为 -0.883 和 -4.099。提示需要关注高置信度误判与概率可靠性。
- **已发现风险预测中的目标泄漏：** 脚本只在 Sentence 正确样本上定义伤害 `harmed`，而输入又包含 `sentence_p_machine`、`context_p_machine` 和差分。该子集上 `harmed` 等价于模型是否预测分歧，几乎可直接由输入判断。ExtraTrees AP≈0.9999、AUC≈1.0 **无效于证明真实门控能力**；不得将其写入成果表作为有效增益。

相关代码：`scripts/analyze_context_harm_patterns.py`；最近 PR #4 修复提交 `2e64862`，诊断脚本可以运行，但**下一步不是继续优化其高分**，而是修正研究问题和验证设计。

### 3.4 当前结论与剩余研究边界

- **已回答：** Stacking 相对同协议 RoBERTa 有 266 条仅 Stacking 正确、184 条仅 RoBERTa 正确，Macro-F1 提升约 0.0080；修正版统计和一致性验证已在本地与 CI 通过。
- Context 为什么在一些样本帮助、在另一些样本伤害？**已知现象**：Human FP、长度相关性、概率过度自信；**未知机制**：风格失配、邻句噪声、跨文章来源、校准偏移等。
- **已得到有限肯定：** 在同一严格窗口清理训练协议内，E2 固定 Human-veto 减少 Human FP 并改善 OOF、Novel Macro-F1；**未实现**与不同训练/清理协议的 Stacking 的严格配对胜负比较。
- **工程输出已解决：** Stacking 全量无标签测试可从原始 JSON 复现，500/4,264 校验 PASS；**剩余**老师正式提交字段、课程报告和图表的整理。

---

## 4. 本课程项目要呈现的“不同”：明确选择一个问题

### 4.1 已选定并完成最小实验的主问题

> **在人机混合文章的句子级检测中，上下文信息何时有帮助、何时会干扰判别？能否在已有风格与语义基线之上，通过简洁、可验证的条件决策，在降低 Human 误判风险的同时维持检测性能？**

这与老师要求的句子级分类一致；与现有 SVM/RoBERTa/Stacking/Context 产物可衔接；重点是**提出观察、验证假设、给出对照方法**，而不是创造一套没有证据的大架构。

研究故事（已验证观察用“发现”，未验证机制仍用“假设”）：

`现有分类器 → 风格/语义互补 → Context 对短句可能有价值 → Context 对部分 Human 和长句造成伤害 → 分析可观测的可靠性信号 → 检验简单条件使用能否降低此类错误`

### 4.2 只保留一条已完成的创新路线

**已完成 E2：选择性上下文融合（Selective Context Use）。**

- 输入：已有 Sentence-matched 和 Context 的 OOF 预测分数及推理时可得的文本特征（如句长、邻句可用性）；SVM 风格分数只在确认样本/训练协议一致后加入。
- **已执行原型：** 固定概率融合 → 长度条件选择 → **长句（≥20 词）且两个模型分歧时的 Human-veto**；没有重新训练 RoBERTa，也没有实施可学习门控。
- 关键比较：同一批样本、相同 base 模型、相同阈值/调参预算，公平比较 Sentence-only、Context-only、简单平均/线性融合、长度规则及学习式选择器。
- **实测判断：** OOF 五折均有 Macro-F1 增益、Human FP 下降，但 Machine FN 上升；Novel 固定规则复核支持继续保留该课程特色实验。鉴于已有充分证据，**停止继续叠加复杂 gate**，而不是因为探索闸门允许 E3 就必须训练。

**保底替代 B：Human 误判风险诊断与可靠决策。**

- 若学习式门控不成立，保留强 Stacking 或 Context 基线，围绕 Human FP、阈值—误检率/召回率权衡、概率校准和条件错误分析完成可解释的“小创新”。
- 不能凭现有数据声称 conformal prediction 的严格 FPR 保证；需专用校准集、可验证的假设和独立评估协议。

**不启动的路线：** 同时做 MoE、对比学习、重写/扰动、大模型 logits 分布、DNA 修复、零样本判别、对抗训练。这些是研究储备而非课程必做内容；除非前述低成本路径被证实不适用且课程交付已保障。

---

## 5. 老师给出的论文，如何从“列表”变成“可使用的机制”

下面只保留**与当前问题直接关联、已查到正式来源**的精读优先项；其他论文保留为备选。论文的原任务、数据与当前 GPT-3.5-turbo 句子级检测不同，**借鉴机制不等于照搬论文指标或复现整套框架**。

| 顺序 | 论文与核验来源 | 它真正解决什么 | 在本项目中的用途 | 决策 |
|---|---|---|---|---|
| **1** | **SenDetEX**, EMNLP 2025；https://aclanthology.org/2025.emnlp-main.268/ | 混合文本句子级检测，将风格与上下文结合；也提出数据构造方案 | 主问题和风格/上下文互补的**最直接参照**；先核实其输入、融合粒度、数据协议 | **必读方法/实验相关部分** |
| **2** | **MoSEs**, EMNLP 2025；https://aclanthology.org/2025.emnlp-main.294/ | 风格感知的专家路由、条件阈值与不确定性 | 借鉴“样本条件不同，阈值/专家不应完全固定”的思想；先做规则/小模型，不复刻整个参考仓库 | **重点选读** |
| **3** | **Reliably Bounding False Positives via Multiscaled Conformal Prediction**, ACL 2025；https://aclanthology.org/2025.acl-long.601/ | 高误报代价下的 FPR 控制与检测性能折中 | 用于构造 Human FP 与检测效用的评价讨论；**不直接移植统计保证** | **作为可靠性参考** |
| **4** | **LAMCL**, ACL 2026；https://aclanthology.org/2026.acl-long.1118/ | 机器修订文本的长度感知、多尺度对比学习 | 为长度异质性提供机制灵感；原任务不同，且需要新训练，课程阶段不优先 | **储备，不先复现** |
| **5** | **Identifying Bias in Machine-generated Text Detection**, ACL 2026；https://aclanthology.org/2026.acl-long.109/ | 英文生成文本检测中的群体差异与偏误风险 | 提醒关注把 Human 判成 Machine 的错误代价；当前数据没有相应人口属性标签，**不得套用群体偏见结论** | **讨论/风险边界参考** |

论文采用前做一张简短对照卡：**研究问题 → 输入与监督 → 核心机制 → 评估协议 → 适配点 → 最小改动 → 成本/风险 → 是否值得动手**。阅读终点是产出一个可验证假设，而不是堆阅读笔记。

其他候选（如 DETree、FAID、HACo-Det、Dual-Role Context、AIDER、PRDetect、Exons-Detect 等）在现阶段不进入必须实施清单；如需纳入，先核对正式论文及代码，确认其检测粒度、训练数据与成本。

---

## 6. 实验设计：先设立公平基线，再检验创新

### 6.1 评估指标与必报内容

**课程主指标（四位小数）：** Accuracy、Precision、Recall、Macro-F1、Micro-F1。明确二分类 positive class = Machine（1）；由于老师未指定 Precision/Recall 的单值 averaging，报告 Machine 类、Human 类及 macro 版本，避免口径争议。单标签二分类中 Micro-F1=Accuracy。

**研究辅助指标：** Human→Machine FP 个数与 Human FPR、Machine Recall、每类 F1、混淆矩阵；ROC-AUC、AP、Brier score、log-loss、可靠性图；不同长度、domain、文章来源分组表现。概率指标必须标注是否经过校准。对多折结果同时展示 per-Fold，必要时配合 source/文章级 bootstrap CI。

**最关键的决定标准不是“指标多”，而是：**

1. 主指标 Macro-F1 在相同协议下相对冻结基线是否有可重复的改善。
2. Human FPR 是否同时恶化；若发生权衡，说明改善带来的代价。
3. 改善是否只来自一个 Fold、某一长度桶或反复选阈值。
4. 创新模块带来的增加成本是否配得上增益。

### 6.2 门控诊断的正确标签定义（修复目标泄漏）

旧风险实验只取 Sentence 正确样本，然后把伤害定义为 `Context 错 & Sentence 对`，但预测输入包括双方类别分数。该子集的伤害标签可以由模型**是否分歧**推算，故 AP≈1.0 没有意义。

**真正需要学习的是：两个模型意见不同的时候，应相信谁？**

对于所有 OOF 行，令 `s_i = I(p_sentence >= 0.5)`、`c_i = I(p_context >= 0.5)`。在 `s_i != c_i` 的分歧子集里，定义 `z_i = I(c_i == y_i)`：`z=1` 表示选 Context，`z=0` 表示选 Sentence。这个标签由真值监督训练，但**样本是否进入分歧子集只由推理时可见的预测决定**，不按某模型是否正确筛选。

- 特征只使用推理时可获得的分数、长度、上下文存在性等；严禁把 `y_true`、`sentence_correct`、`context_correct`、`harmed`、文章 ID 编码直接输入。
- 分歧选择器需对比 **Always-Sentence / Always-Context / 更高置信度者 / 长度规则 / Logistic** 等简单方法；不直接跳向复杂网络。
- 按 `article_key` 或更严格的 `source_article_id` 分组验证；如果按 source ID 隔离才符合总体协议，优先用 source 级分组。分折训练中拟合特征标准化和超参数；调参/选模型必须与报告验证分开。
- **最后必须在全部样本上计算整体指标**。只报告约 745 条 Fold 4 分歧样本的选择准确率，不构成整体检测改善。
- Fold 4 已被反复诊断，不能再包装成未见过的独立测试。已有 OOF 和 Novel 都属于开发资源；严格独立效果仍要靠将来的未触碰标注数据/真实隐藏测试结果。

### 6.3 最小实验矩阵与截至 2026-11-22 的状态

| ID | 实验 | 使用什么 | 新训练成本 | 必须输出什么 | 是否进入下一实验 |
|---|---|---|---|---|---|
| E0 | 统一基线与协议 | 已有 SVM、RoBERTa、Stacking、Veto、Matched Sentence/Context | 无 | **完成**：分协议表、修正统计 PASS、复现协议说明 | 已收口 |
| E1 | 解释已有改善/退化 | 已有 OOF、corrected Stacking、Fold 4 诊断 | 无 | **分析完成大部**：错误转移、句长/类别分层、置信度风险、统计表；**课程图表与受控混杂分析尚待整理** | 报告阶段补图，不继续扩模型 |
| E2 | 简单条件融合 | 同协议配对预测 | 无新神经网络训练 | **完成**：7 项合成测试、52,864 OOF、五折改进、Bootstrap、11,682 Novel 固定规则复核（6 项合成测试） | **保留为课程创新；停止额外调参** |
| E3 | 轻量学习式分歧选择器（可选） | 无泄漏 OOF 选择标签 + 可用特征 | 低；只训逻辑回归等元模型 | **未执行、明确暂不启动**；旧 Fold 4 风险学习因目标泄漏作废 | 不作为本次课程完成前置条件 |
| E4 | 与最终强基线公平比较 | 经过协议核对的 Stacking 与创新候选 | 低/视对齐情况 | **未完成严格同协议直接比较**；已明确两者不可跨协议排行 | 阶段一选 Stacking 为工程模型；E2 单独报告 |
| E5 | 最终测试推理与报告 | 冻结的模型与配置 | 推理 | **推理与通用输出检查完成**；500/4,264、阈值 0.585、五折 CUDA PASS；**正式 schema/报告未完成** | 优先完成报告与提交手续 |

**不能把旧 Stacking 与 Matched Context 的“不同训练数据/不同 Fold/不同阈值”的分数直接融合或排名。** 若要扩展到三专家，先按 `oof_row_id`、fold、source、训练样本集合、推理配置全面审计特征兼容性；否则将其列为课后研究，不作为当前主线。

---

## 7. 分阶段实施计划与停止条件（更新为执行状态）

| 阶段 | 计划核心 | 2026-11-22 进度 | 是否仍需执行 |
|---|---|---|---|
| 0 任务边界 | 核对 PDF、标签、输出粒度、指标 | **完成**：明确任务及 PDF 未规定官方 schema | 仅在老师补充通知后更新 |
| 1 基线收口与工程验收 | 模型盘点、Stacking 修正版核对、五折推理 | **完成**：工件 15 项存在，8 项测试 PASS，修正统计 PASS；Stacking Novel replay、16 句 GPU 干跑、500/4,264 全量 GPU 推理均 PASS | 不重训，不变更冻结阈值 |
| 2 现象/错误分析 | Context/Sentence 错误转移、句长、FP/FN、概率风险 | **实验与聚合表大部分完成**，包括 Fold 4 及 E2 分层；**正式报告图仍待绘制** | 制作 2–3 张有根据的图，不把相关性说成因果 |
| 3 最小方法改进 | E2 固定/条件融合、公平配对 OOF、Novel 复核 | **完成**：20 词规则，52,864 OOF 五折与 11,682 Novel 两口径；2,000 次来源 Bootstrap | 冻结规则，E3 暂不执行 |
| 4 测试推理 | 冻结 checkpoint、标准化/阈值、全量无标签预测 | **完成通用工程验收**：500 篇、4,264 句、全五折成功；**未确认官方提交 schema** | 核对教师通知后必要时新增安全格式转换 |
| 5 报告与提交 | 课程主指标、实验图表、创新对照、局限及复现 | **尚未最终交付**，本 V3 提供报告可用的实验台账 | 当前最优先任务 |

**阶段 1 已满足的冻结内容：** 以 `scripts/predict_oof_stacking.py` 对原始测试 JSON 执行五折来源组匹配推理；每折使用原 SVM 和 RoBERTa、保存的 OOF 折标准化参数及对应 Logistic 元模型；阈值 **0.585**。配置与权重保留服务器，禁止无提示覆盖。

**阶段 3 已满足的完成边界：** 固定 ≥20 空白分词的规则仅在 Context 判 Machine、Sentence-matched 判 Human 的情况下否决 Context；OOF/Novel 证据为开发验证，不能据此替换 Stage-1 Stacking 的无标签测试预测，也不能宣称模型端到端已通过公平的 E4 对比。

**当前执行顺序（一次只做一项）：** ① 核实老师是否有正式提交文件 schema、截止时间和报告格式 → ② 根据第 12 节生成课程报告结果表及 2–3 张真实数据图 → ③ 对照验收清单补齐 README/环境/命令与交付材料 → ④ 若正式通知要求预测文件，按其字段从本地已通过的 Stacking 输出转换并重新校验。**E3、额外阈值搜索、重训 RoBERTa 均不在当前计划内。**

---

## 8. 最终交付验收清单（以已验证证据为准）

### 课程任务与工程

- [x] 任务说明、`human=0`/`machine=1`、文章级输入与句子级输出一致。
- [x] 使用正确标签/指标定义；E2 同协议五项主指标已整理为四位小数（第 12.7 节）。
- [x] Stacking 训练/开发结果与其他评估协议分开报告。
- [x] Stacking 推理入口、五折权重、标准化、元模型、阈值 **0.585** 的冻结记录已建立。
- [x] 500 篇/4,264 句无标签推理、顺序/原文/标签/数量/阈值通用 schema 自动校验 PASS。
- [ ] 获取并核对**老师正式提交 schema、文件名、提交入口/日期**（PDF 本身未明确）；如需要，转换并再次验收。
- [ ] 汇总正式课程报告、图表、必要代码与依赖说明，并确认提交成功。

### 实验与研究质量

- [x] Stacking 修正版聚合统计一致性验证；Novel 历史元模型 11,682 句复现通过。
- [x] Sentence-matched / Context OOF 52,864 条配对、来源分组及五折检查。
- [x] E2 固定融合、长度规则、长句 Human-veto 对照、五折指标、按来源组 Bootstrap。
- [x] 不调参复用 E2 的 20 词规则完成 Novel 的两种协议各 11,682 句复核，分别报告结论。
- [x] Human FP/FPR 与 Machine Recall/FN 共同报告，并注明改善代价。
- [x] Fold 4 高性能旧 harm predictor 的目标泄漏识别并**作废**。
- [x] 开发验证与无标签测试明确区分；不同 purge/训练协议不直接排名。
- [ ] 用已验证原始聚合产物绘制核心 2–3 张统计图，补全报告图注与来源。
- [ ] 如最终报告要声称文章伤害“异常集中”，需按文章大小/标签比例进一步控制；否则只作描述。
- [ ] 如声称 E2 优于 Stacking，须另外完成严格同协议 E4（**当前没有此证据，不作该声明**）。

---

## 9. 下一批任务（2026-11-22 起，以交付为中心）

| 优先级 | 动作 | 交付物 | 停止/约束 |
|---|---|---|---|
| P0 | 核对老师其余课程通知/平台提交要求 | 官方提交格式、日期、报告/PPT要求核对单 | 不能凭 PDF 示例擅自设定预测 schema |
| P1 | 把第 12 节实验台账转化为报告的实验设置与结果 | 五项课程指标表、差异表、错误转移表、限制说明 | 只报对应评估协议，Precision/Recall 必须写正类 |
| P1 | 完成有问题导向的核心图 | 句长/类别分布、E2 Human FP/FN 代价、长度/分折净收益 | 从本地真实聚合/预测生成；不上传敏感行级文本 |
| P1 | 冻结运行与复现说明 | README、依赖、模型目录/权重说明、完整预测校验命令 | 不重训、不开新调参循环 |
| P2 | 如官方提交要求不同，做只读格式转换与二次校验 | 从已通过的 Stacking 本地预测生成最终提交文件 | 不改变预测标签、顺序、阈值；提交前先验证 |

**暂不执行：** E3 新门控训练、把 E2 融入旧 Stacking 模型、无标签测试性能估计、跨协议夸大式比较。完整实验细节见第 12 节。

---

## 10. 版本、隐私与证据记录规则

### 10.1 仓库分支与 PR（均为独立渐进分支，不代表已合并到 `main`）

| PR | 分支 | 主要内容 | 本轮状态 |
|---|---|---|---|
| [#5](https://github.com/q3483109023/AIGTD/pull/5) | `handoff/stage1-baseline-freeze` | 工件盘点、Stacking 修正版统计与验证、测试 schema/SVM 干跑 | 本地与 CI 已验收；PR 为 Draft，未据此声称主分支已合并 |
| [#6](https://github.com/q3483109023/AIGTD/pull/6) | `inference/oof-stacking-replay` | 五折 OOF Stacking 的 Novel replay、真实测试推理入口与回归测试 | 服务器 16 句及 500/4,264 全量推理通过；PR 为 Draft |
| [#7](https://github.com/q3483109023/AIGTD/pull/7) | `experiment/e2-selective-context` | E2 同协议配对 OOF 的固定/长度/否决规则、Bootstrap | 服务器 52,864 句、2,000 Bootstrap PASS；PR 为 Draft |
| [#8](https://github.com/q3483109023/AIGTD/pull/8) | `validation/e2-fixed-veto-novel` | 固定 20 词 E2 规则在 Novel 的两种协议下复核 | 服务器 11,682 句、2,000 Bootstrap PASS；PR 为 Draft |

**说明：** PR #5–#8 为层级依赖分支：#8 基于 #7、#7 基于 #6、#6 基于 #5。最新提交与本地是否有未提交修改，始终以服务器 `git status --short --branch` 与 `git log -1 --oneline` 为准；本文不声称这些 PR 已合并 `main`。

### 10.2 证据优先级与数据保护

- 课程 PDF 约束 > 用户私有服务器实际日志 > 已通过 GitHub CI 的公开聚合运行 > 历史交接文档记录 > 尚待验证假设。对于不同模型或协议，不因某个统计显著就跳过可比性审计。
- 所有实验至少记录：目的、样本/文章/来源组数、数据协议和 Fold、模型版本、参数/阈值的来源、运行命令、主要指标、FP/FN/Recall 代价、复核状态、局限、产物目录；详细字段见第 12 节。
- 公共仓库只上传代码、测试与非敏感聚合表；**不要上传**原始训练/测试文本、来源文章 ID、逐句预测、检查点权重和含 row-level 内容的 CSV。`results/stage1_local_audit/`、`results/e2_selective_context/` 与 `results/e2_novel_fixed_rule/` 都应保持本地/被 Git 忽略。
- 不执行 `git add .`、`git clean` 或强制覆盖未知文件；先检查 `git status --short`。课程材料若后续提供正式 schema、截止时间和评分口径，再更新本文第 1、8、9 节，保留实验历史。

---

## 11. 本文依据与使用方式

1. **课程任务源文件：** `1. 数据挖掘与知识发现-课程project(2).pdf`（10 页）；第 2–5 页为任务/评价硬要求，第 6–9 页为方法与分析示例。
2. **历史技术基线：** `AIGTD_project_handover_compact_2026-11-21(1).md`，含 5 折 OOF、数据审计、模型版本、指标、代码路径、历史问题等完整明细；本文没有删除其中的工程历史。
3. **2026-11-22 用户服务器实测：** 15 项工件在位、8 项 Stage-1 回归测试、Stacking corrected 统计 PASS、Novel replay、2/16 句 SVM/Stacking 干跑、500/4,264 句 Stacking 全量测试、E2 52,864 OOF/2,000 source bootstrap、E2 11,682 Novel/2,000 source bootstrap，均有本轮对话的实际终端输出。
4. **论文官方参考：** 本文第 5 节 ACL Anthology 链接；只对已核验的五篇作具体归类，其余列表是待查储备。

**执行原则：任何新的工作必须先指出它要回答本文哪一个问题、解决哪一个交付缺口、采用什么公平对照、满足什么停止条件。当前优先级是报告/格式/提交，而不是继续刷开发集分数。第 12 节为实验记录正本。**

---

## 12. 正式实验记录台账（2026-11-22，报告引用优先依据）

> **统一说明：** 以下记录是原 V2 未包含的增量实验与可核对的历史结论，实验结果以服务器终端实际输出和指定 CI 日志为依据；所有“模型效果”来自开发数据，500 篇无标签 Test 的 `PASS` 只证明推理/结构正确。各小节的阈值、Fold、样本覆盖、来源组数不允许跨协议挪用。`human=0`，`machine=1`，FP = Human→Machine，FN = Machine→Human。

### 12.0 实验编号、进展与证据入口

| 编号 | 实验/验收 | 日期 | 研究/工程目标 | 结论与状态 |
|---|---|---|---|---|
| EXP-00 | 数据与协议核对 | 2026-11-21—22 | 保证样本、文章记录、来源组、训练/验证口径不混乱 | **完成核对**，数据与限制见第 2 节 |
| EXP-01 | Stacking corrected 统计重算 | 2026-11-22 | 验证 AUC/AP 口径、成对改善、Bootstrap | **服务器 PASS + CI PASS**，11,682 句 |
| EXP-02 | Sentence/Context OOF 与 Fold 4 错误分析 | 2026-11-21—22 | 找出上下文收益的非均匀性与风险 | **既有结果已审阅**；高分 harm 风险预测作废 |
| ENG-01 | Stage-1 工件盘点和测试脚本 | 2026-11-22 | 验证权重/入口存在性和静态回归 | **8/8 测试 PASS，15 项工件全存在** |
| ENG-02 | 单 SVM 2/16 句推理干跑 | 2026-11-22 | 先验证无标签输出结构端到端 | **推理与通用 schema PASS**，阈值 0.52 |
| ENG-03 | Stacking Novel 元模型 replay | 2026-11-22 | 验证元模型、折归一化与阈值无漂移 | **PASS**，11,682 句，5 折 |
| ENG-04 | Stacking 2/16 句真实 GPU 干跑 | 2026-11-22 | 测试真实模型权重读取与顺序 | **CUDA/结构/阈值 PASS**，Fold 0/1 |
| ENG-05 | Stacking 500/4,264 句全量推理 | 2026-11-22 | 验证全部五折、正式冻结通用输出 | **全五折 CUDA PASS；通用校验 PASS** |
| E2-01 | 配对 OOF 固定/长度/否决规则 | 2026-11-22 | 在相同训练协议中寻找条件使用上下文的方法 | **7/7 测试、52,864 句、2,000 Bootstrap PASS** |
| E2-02 | 固定 20 词规则 Novel 复核 | 2026-11-22 | 不重训/不调参检查规则稳定性 | **6/6 测试、11,682 句、两种协议、2,000 Bootstrap PASS** |
| DOC-01 | 课程报告与正式提交格式 | 后续 | 把可复核结果转化为最终交付 | **仍未完成**；PDF 未指定正式文件字段 |

**复现代码与协议来源：** [PR #5](https://github.com/q3483109023/AIGTD/pull/5)、[PR #6](https://github.com/q3483109023/AIGTD/pull/6)、[PR #7](https://github.com/q3483109023/AIGTD/pull/7)、[PR #8](https://github.com/q3483109023/AIGTD/pull/8)。详细对应仓库文档：`docs/baseline_protocol_and_results.md`、`docs/freeze_candidate.md`、`docs/oof_stacking_test_inference.md`、`docs/e2_selective_context_protocol.md`、`docs/e2_fixed_rule_novel_recheck.md`。

### 12.1 EXP-00｜输入数据、分组方式与实验边界

**目的：** 确保不同来源 ID、同一文本重复、多来源数据与多 Fold 预测不会被错误去重、泄漏或重复计数。

**数据口径：** 原始训练 JSON 共 **5,000 个文章记录、66,471 句**；五折 OOF Train **3,983 个文章记录、52,864 句**；Full Validation **1,017 个文章记录、13,607 句**；Novel **11,682 句、1,017 个 `article_key`、619 个原始来源 ID**；无标签测试 **500 篇、4,264 句**。

**键与协议：** 文章记录用 `article_key` 唯一标识；源文章 `source_article_id`/`article_id` 可以跨不同记录重复；句子位置由 `sentence_index` 固定；OOF 与 Novel 用 `oof_row_id` / `novel_row_id` 成对对齐。来源组不得跨 OOF Fold；目标文本规范化后重叠需 purge。严格 Context 对照的 Sentence/Context 两模型共享**相同严格窗口 purge 训练划分与折**；与旧 Stacking 中的 RoBERTa 训练协议不同。

**完整性检查：** `results/context_oof_analysis/integrity_report.json` 原有记录为：Sentence/Context 各 **52,864** OOF 行、`oof_row_id` 唯一 52,864、Fold `[0,1,2,3,4]`、labels/folds match。E2 实验再次核对文本、文章/来源键、位置及来源组不跨折。Novel 每个目标有五折预测，每个模型原始行数 **58,410**，但**评估样本仍是 11,682**。

**限制：** 以上仅说明已使用代码协议中样本配对有效，不保证真实标签全无歧义。相同来源 ID 与精确文本重复不是同一个概念；“Novel” 名称不等于从未用于调参的独立测试。

### 12.2 EXP-01｜Stacking corrected Novel 聚合分析及一致性验证

**目的：** 修正按多折原始分数池化计算 RoBERTa/SVM AUC/AP 时的尺度问题；复核 Stacking 相对同协议基线的实际改善，避免把旧错误的 ranking 指标写入报告。

**方法：** 五折 OOF 训练得到 SVM 决策分数 `svm_decision_score`、RoBERTa logit margin `roberta_logit_margin`；在各自 held-out OOF 折估计均值和总体标准差（`ddof=0`），将 Novel **按来源组固定匹配到单折**，使用保存的折统计做标准化；五折 Logistic 元模型跨折拟合，Stacking 的最终分类阈值 **0.585** 在 OOF 确定；Novel 为开发验证。脚本 `scripts/analyze_stacking_errors.py --aggregate-only` 和 `scripts/verify_stacking_aggregates.py`，seed 42、source-cluster bootstrap 2,000 次（验证器对输出做结构/算术一致性检查，**不等于重新独立实现 Bootstrap**）。

**用户服务器实测：** `Rows: 11682; article_key: 1017; source IDs: 619`，`status=PASS`，四模型、三组 McNemar 比较、12 项 Bootstrap 结构核对。结果：

| Fold-matched Novel 模型 | Accuracy | Macro-F1 | Human F1 | Machine F1 | Human FP | Machine FN | ROC-AUC（修正） |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stacking | **0.921674** | **0.917374** | **0.898525** | **0.936224** | **562** | 353 | **0.972856** |
| RoBERTa | 0.914655 | 0.909330 | 0.887357 | 0.931303 | 686 | 311 | 0.958178 |
| SVM | 0.885722 | 0.879980 | 0.853731 | 0.906230 | 717 | 618 | 0.950582 |
| SVM Veto | 0.919106 | 0.914393 | 0.894307 | 0.934480 | 615 | 330 | 不适用（无连续排名分数） |

**Stacking vs RoBERTa 成对正确性：** 两者都正确 **10,501**、都错误 **731**、仅 Stacking 正确 **266**、仅 RoBERTa 正确 **184**，四组之和 = 11,682。Stacking 共多纠正 **82** 句，Macro-F1 提升 **0.008044**；Human FP **686→562** 减少 124，Machine FN **311→353** 增加 42。历史归档 McNemar 句级 exact p≈0.000129，但句子同来源相关；历史来源组 Bootstrap 的提升约 **+0.0081**，95% 区间约 **[+0.0038,+0.0125]**（2,000 次、619 来源组）。这些检验都基于开发结果，不自动等于独立泛化保证。

**复核层次：** GitHub [Stage-1 CI](https://github.com/q3483109023/AIGTD/actions/runs/44924113524) 通过安全聚合重算及验证；用户服务器再次重算并通过；不上传行级 Novel 文本/ID。**结论：** 冻结为**阶段一最强已记录的同协议开发融合模型**，随后完成真实测试推理（12.5），而非宣称未知测试集得分 0.9174。

**本地结果：** `results/stage1_local_audit/stacking_recheck/`（gitignore；具体 summary/CSV 文件名以脚本输出为准）。

### 12.3 EXP-02｜严格 Sentence/Context 配对 OOF 与 Fold 4 风险

**目的：** 回答“上下文是否总是增强判别？”为 E2 选择性融合提供问题动机与可验证的错误类型。输入来自**同一**严格 Context window-purged 五折的数据，不与旧 Stacking 直接比较。

**总体：** 52,864 OOF 句，原始来源组约 2,473、文章记录 3,983，固定分类阈值 **0.5**，Human 支持 17,731、Machine 支持 35,133。

| 严格配对 OOF 模型 | Accuracy | Macro-F1 | Human F1 | Machine F1 |
|---|---:|---:|---:|---:|
| Sentence-matched | 0.894087 | 0.875989 | 0.828614 | 0.923363 |
| Context | **0.914384** | **0.899834** | **0.861658** | **0.938010** |

**按 Fold（Macro-F1）：**

| Fold | 样本 | Sentence-matched | Context | Context − Sentence |
|---:|---:|---:|---:|---:|
| 0 | 10,566 | 0.856941 | 0.908889 | +0.051948 |
| 1 | 10,567 | 0.887305 | 0.916274 | +0.028970 |
| 2 | 10,578 | 0.893900 | 0.891594 | −0.002306 |
| 3 | 10,574 | 0.871779 | 0.913024 | +0.041245 |
| 4 | 10,579 | 0.868178 | 0.866750 | −0.001428 |

**Fold 4 关键反例：** Accuracy **0.8905→0.8910** 微升，但 Macro-F1 **0.8682→0.8668** 微降，AUC **0.9664→0.9129**、AP **0.9824→0.9004**；Human FP **1,015→1,091**、Machine FN **143→62**。成对的错误分组为：两者都对 9,051、都错 783、仅 Context 正确 375、仅 Sentence 正确 370。Context 帮助/伤害在 Human 上明显不对称，较长目标句与伤害相关；但相关性不构成机制性因果解释。概率 Brier/log-loss 差异在“帮助/伤害”子集内明显，需防止过度置信判错。

**无效证据：** 旧 Fold-4 风险分类器仅从 Sentence 正确的样本中构造标签，特征包含两模型预测与分歧；`harmed` 因此近似由输入可计算。**ExtraTrees AP≈0.9999 / AUC≈1.0 必须作废**，不能称为成功门控；所有后续 E2 规则只用推理时可见条件，不按真值选择输入子集。

**结论：** Context 对整体有帮助，但不保证所有 Fold、类别或长度段受益；提出“降低 Human 误报”的 E2 假设合理。不能说“句长导致错误”或“文章聚集显著”，除非独立控制混杂。

**原始已有产物：** `results/context_oof_analysis/metrics_overall.csv`、`metrics_by_fold.csv`、`paired_cluster_bootstrap.csv`、`results/context_harm_patterns/` 等；相关命令见 `scripts/analyze_context_oof.py`、`scripts/analyze_context_harm_patterns.py`。

### 12.4 ENG-01 至 ENG-03｜工件、基础干跑与 Stacking replay

**ENG-01：服务器代码/环境验收。**

- 分支：`handoff/stage1-baseline-freeze`；实测 HEAD `9a3b688`（只表示当次运行的提交，不代表后来服务器当前 HEAD）。
- `python -m unittest discover -s tests -p 'test_stage1_freeze.py' -v`：**8/8 PASS**。
- `python scripts/check_stage1_artifacts.py`：**15 个 `artifact_presence` 字段全部 true**，涵盖配置、原始无标签测试、旧单模型、5 折 SVM/RoBERTa/Context、Meta 模型与折标准化、Novel 行级预测和修正版聚合文件。`git_worktree_has_changes=true` 是工作区有未提交变化的提示，**不是测试失败**；需在切分支前确认变化。
- 当时 `stacking_end_to_end_test_inference_implemented=false`，仅描述 PR #5 的历史状态；**已由 PR #6 新入口解决**。

**ENG-02：单模型 SVM 干跑。**

- 输入：原始无标签 Test 的前两个**完整文章记录**，2 篇、16 句；不以重复源 ID 去重，不使用标签。
- 模型：既有单 SVM `scripts/predict.py --model-type svm`，其自身阈值 **0.5200**，与 Stacking 的 0.585 **不可混用**。
- 实测：Human **6**、Machine **10**；写出本地 `results/stage1_local_audit/dry_run/test_predictions.csv` 和 `.json`；验证器输出 `status=PASS`、`json_order_and_text_match=true`、`csv_consistent=true`。本次初期命令 `threshold_checked=false` 是未传显式 `--threshold` 的检查覆盖不足，**不是 SVM 错误**。
- 结论：原始 JSON→句子展开→预测→文章/顺序恢复→通用输出路径可运行，但此 SVM 不是最终优选模型。

**ENG-03：Stacking 已保存元模型一致性 replay。**

- 命令：`python scripts/predict_oof_stacking.py --replay-novel results/oof_fusion/novel_fusion_predictions.csv`。
- 数据：已保存 Novel **11,682 句、5 折**；各折 SVM/ RoBERTa 已有 raw margin 与折 z-score，加载保存的 `fusion_models.joblib` / normalization；并非重新执行 GPU 前向。
- 服务器输出：`status=PASS`；两模型标准化分数最大绝对差均 **0.0**；Stacking 概率最大绝对差 **2.220446049250313e−16**；`threshold_labels_match=true`。GitHub [PR #6 CI](https://github.com/q3483109023/AIGTD/actions/runs/44925128886) 附加 **6/6 合成/模拟推理测试 PASS**。
- 结论：保存的 Meta 特征顺序/折归一化/阈值没有漂移；**这一项单独不能证明真实神经 checkpoint 推理**，因此之后继续 ENG-04/05。

### 12.5 ENG-04 至 ENG-05｜最终 Stacking 推理路径实际验收

**冻结推理身份：** 不是 `scripts/predict.py --model-type fusion` 旧方案，而是 `scripts/predict_oof_stacking.py` 的**五折 Fold-matched OOF Stacking**。给定整个目标数据集：以来源 ID 用只依赖组大小和稳定哈希的确定性 LPT 策略分到一折；使用那一折的 SVM `decision_function` 与 RoBERTa logits[1]−logits[0]；采用存储的各折 OOF 统计转换成 `[svm_z, roberta_z]`；应用对应 Logistic meta model；阈值 **0.585**。其训练环境保存 `max_length=256`；不重训、不平均五个 Meta 概率、不在测试集选阈值。

**ENG-04：两篇文章（16 句）真实 CUDA 干跑：**

| 检查 | 实测 |
|---|---|
| 2 个文章记录 / 16 句 | 覆盖完整 |
| 使用的 Fold | Fold 0：8 句；Fold 1：8 句，其余 Fold 0 句 |
| Human / Machine | 4 / 12 |
| 计算设备 | `cuda` |
| 预测状态 | `COMPLETED_AWAITING_SCHEMA_VALIDATION`（模型计算完成待格式检查，并非失败） |
| 输出校验 | `status=PASS`、原文/顺序一致、CSV 一致、`threshold_checked=true` |

**ENG-05：无标签 Test 全量真实 CUDA 推理：**

| 验收内容 | 服务器实际输出 |
|---|---|
| 文章 / 句子 | **500 / 4,264** |
| Fold 0/1/2/3/4 分配句数 | **853 / 853 / 853 / 853 / 852**，合计 4,264 |
| Human 预测 | **1,548** |
| Machine 预测 | **2,716** |
| 概率阈值 | `0.5850000000000001`（浮点打印等价配置 0.585） |
| 设备 | `cuda`，五折全部执行 |
| 预测程序状态 | `COMPLETED_AWAITING_SCHEMA_VALIDATION` |
| 后续格式校验 | **`status=PASS`**；`articles=500`、`sentences=4264`、`json_order_and_text_match=true`、`csv_consistent=true`、`threshold_checked=true` |
| 重复 source ID | 按文章**记录**输出，不错误合并；`source_article_ids_may_repeat=true` |

**本地保留文件（隐私敏感，绝不公共上传）：**

```text
results/stage1_local_audit/stacking_full/test_predictions.json
results/stage1_local_audit/stacking_full/test_predictions.csv
results/stage1_local_audit/stacking_full_validation.json
```

**复现命令：**

```bash
python scripts/predict_oof_stacking.py \
  --test data/raw/AIGTD_test_without_labels.json \
  --expected-articles 500 --expected-sentences 4264 \
  --device auto --batch-size 32 \
  --output-dir results/stage1_local_audit/stacking_full

python scripts/validate_test_predictions.py \
  --raw-test data/raw/AIGTD_test_without_labels.json \
  --pred-json results/stage1_local_audit/stacking_full/test_predictions.json \
  --pred-csv results/stage1_local_audit/stacking_full/test_predictions.csv \
  --expected-articles 500 --expected-sentences 4264 --threshold 0.585 \
  --summary-out results/stage1_local_audit/stacking_full_validation.json
```

**结论和边界：** 五折端到端推理的工程路径已通过真实用户服务器校验，可以冻结为**最终工程基线**；通用校验结果的 `schema=generic_aigtd_predictions_not_official_submission` 明确提示**尚非已确认的官方提交格式**。没有真实标签，不能把 Human/Machine 数量换算成准确率，不能声称“Test Macro-F1 = 0.9174”。

### 12.6 E2-01｜创新规则定义及完整候选集合（先固定对照）

**研究问题：** 全上下文 Context 整体提高 Macro-F1，但部分长 Human 句受干扰。能否用推理时可见的模型**意见分歧**和目标句长，让 Context 的错误更少、而不大幅削弱 Machine 检出？

**主创新：** 固定 **long_human_veto_20**。目标文本使用 `text.split()` 按空白切分得到词数 `L`，Sentence-matched 模型概率 `p_s`，Context 模型概率 `p_c`，两个阈值都固定为 **0.5**：

```python
# y_hat = 1: machine; y_hat = 0: human
use_sentence = (L >= 20) & (p_c >= 0.5) & (p_s < 0.5)
p_final = np.where(use_sentence, p_s, p_c)
y_final = (p_final >= 0.5).astype(int)
```

这不是需训练的新神经架构；**只在 Context 判 Machine、Sentence 判 Human，且目标句 ≥20 词时否决 Context**。没有使用 `y_true`、`harmed`、`correct`、文章 ID 或来源 ID 作为决策特征。与错误的 Fold-4 harm 预测器不同，规则应用于**全部样本**，不提前筛出“Sentence 已预测正确”的数据。

**对照与消融：** `sentence_matched`（全句子）、`context`（全上下文）、`mean_50`（0.5/0.5 平均）、`context_75`（0.25 句子 + 0.75 上下文）、`short_context_12`、`short_context_24`（短句用 Context、长句用 Sentence）、`long_human_veto_20`、`long_human_veto_32`。均在已保存的相同严格窗口 OOF 配对预测、统一 **0.5** 阈值上执行，**无新模型训练**。

**按折选择诊断（探索性，非独立盲测）：** 对留出 Fold `k`，仅用其它四折已有 OOF 标签决定七个备选规则中哪个 Macro-F1 最高，训练折 Human FPR 不高于 Context +0.002，原始 Fold `k` 的标签不参与**该折**规则选择；随后把五折各自选中规则应用到全部 52,864 行，重新计算整体混淆矩阵和指标。历史其他分析已使用过同一 OOF 数据，且基础模型并没有在最外层嵌套重训；因此这只是**开发型 cross-fitting**，不可自称严格独立检验。

### 12.7 E2-01｜52,864 条严格配对 OOF 的本地实测与代价

**运行环境/复核：** 服务器分支 `experiment/e2-selective-context`，测试 `python -m unittest discover -s tests -p 'test_e2_selective_context.py' -v` **7/7 PASS**；`python -m scripts.experiment_e2_selective_context --input-dir results/context_oof --output-dir results/e2_selective_context --bootstrap-reps 2000 --seed 2026` 完成。此前 GitHub [E2 CI](https://github.com/q3483109023/AIGTD/actions/runs/44926936002) 也成功运行，同样精确的指标；CI 的 bootstrap 重复数可能为 1,000，以下使用**本地 2,000 次**。样本 **52,864**、source ID **2,473**、文章记录 **3,983**；阈值 **0.5**。

**全候选结果（User 服务器终端）：**

| 方法 | Macro-F1 | Accuracy | Human FP | Machine FN | Human FPR | Machine Recall |
|---|---:|---:|---:|---:|---:|---:|
| Sentence-matched | 0.875989 | 0.894087 | 4,196 | 1,403 | 0.236648 | 0.960066 |
| Context | 0.899834 | 0.914384 | 3,636 | 890 | 0.205065 | 0.974668 |
| mean_50 | 0.897943 | 0.912871 | 3,711 | 895 | 0.209294 | 0.974525 |
| context_75 | 0.899977 | 0.914498 | 3,630 | 890 | 0.204726 | 0.974668 |
| short_context_12 | 0.887548 | 0.903753 | 3,877 | 1,211 | 0.218657 | 0.965531 |
| short_context_24 | 0.900072 | 0.914573 | 3,626 | 890 | 0.204501 | 0.974668 |
| **long_human_veto_20** | **0.906659** | **0.919378** | **3,187** | 1,075 | **0.179742** | 0.969402 |
| long_human_veto_32 | 0.901488 | 0.915651 | 3,551 | 908 | 0.200271 | 0.974155 |
| fold_selected_rule | **0.906659** | **0.919378** | **3,187** | 1,075 | **0.179742** | 0.969402 |

**课程要求的五项指标（Machine 为 Precision/Recall 正类，保留 4 位小数）：**

| 配对 OOF 方法 | Accuracy | Precision (Machine) | Recall (Machine) | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|---:|---:|
| Sentence-matched | 0.8941 | 0.8894 | 0.9601 | 0.8760 | 0.8941 |
| Context | 0.9144 | 0.9040 | 0.9747 | 0.8998 | 0.9144 |
| **E2 long_human_veto_20** | **0.9194** | **0.9144** | 0.9694 | **0.9067** | **0.9194** |

单标签二分类 `Micro-F1 = Accuracy`；课程 PDF 不指定 Precision/Recall 的 averaging，因此主表注明 Machine 正类，并可在附表补充 Human 与 macro 版本。

**与 Context 同协议对比：** E2 Macro-F1 **+0.006825**，Accuracy **+0.004994**；Human FP **3,636→3,187** 减少 **449**（约 12.3%），Human FPR **0.205065→0.179742**（下降 0.025323 绝对率，即约 2.53 个百分点）；Machine FN **890→1,075** 增加 **185**，Machine Recall **0.974668→0.969402**（下降 0.005266）。必须明确这是错误代价交换，不能说所有指标都提升。

**五折（选中规则 − Context）：**

| Fold | ΔMacro-F1 | ΔHuman FPR | ΔMachine Recall |
|---:|---:|---:|---:|
| 0 | +0.002868 | −0.011854 | −0.002848 |
| 1 | +0.004196 | −0.015506 | −0.002991 |
| 2 | +0.011538 | −0.037204 | −0.005974 |
| 3 | +0.000599 | −0.018049 | −0.009391 |
| 4 | +0.016040 | −0.043981 | −0.005119 |

五折其它四折训练分数均选择 `long_human_veto_20`；五折的 Macro-F1 差值均非负，不过 Fold 3 只有 **+0.000599**，不能夸大成每折都有显著改善。

**来源分组 Bootstrap（2,473 source ID，2,000 次，seed=2026，配对重采样）：**

| E2 − Context | 观测差值 | Bootstrap 均值 | 95% 分位区间 |
|---|---:|---:|---:|
| Macro-F1 | **+0.006825** | +0.006793 | **[+0.005283,+0.008327]** |
| Human FPR | **−0.025323** | −0.025263 | **[−0.028649,−0.022099]** |
| Accuracy | **+0.004994** | +0.004966 | **[+0.003713,+0.006241]** |

**探索性停止判据：** 当改善 Macro-F1≥0.002、Human FPR 增量≤0.002、五折至少 3 折 Macro-F1 非负、来源 Bootstrap 区间下限>0 时允许考虑后续 E3。本次脚本标记 `consider_E3`，**只表示允许开展，不代表必须或已经开展**。基于课程产出/开销，决定先冻结简单 E2 规则，**E3 未实施**。

**输出：** `results/e2_selective_context/{overall,by_fold,by_length,transitions,fold_selection,bootstrap}.csv` + `decision.json`，均为本地安全**聚合**文件；不输出新行级文本、来源 ID 或模型权重。

### 12.8 E2-02｜20 词固定规则的 Novel 复核（两种协议，2,000 Bootstrap）

**目的：** 在不重训、不重新选句长阈值、不重新调概率阈值的前提下，评估相同 E2 规则能否在**另一批已存在的开发样本**上维持同协议增益，并验证五折概率平均和来源分组单折匹配这两种预测协议不能混为一谈。

**运行与检查：** 服务器分支 `validation/e2-fixed-veto-novel`；`python -m unittest discover -s tests -p 'test_e2_novel_fixed_rule.py' -v` **6/6 PASS**。运行 `python -m scripts.validate_e2_fixed_rule_novel --input-dir results/context_oof --reference results/context_oof_analysis/novel_metrics.csv --output-dir results/e2_novel_fixed_rule --expected-targets 11682 --bootstrap-reps 2000 --seed 2026`，结果 `status=PASS`、`reference_five_fold_mean_baselines_match=true`；每模型输入 **58,410** 五折行，按 target 去重后每协议 **11,682**，source ID **619**。

**协议 A：五折概率平均**，先对每个 `novel_row_id` 的五个 Sentence 预测求平均、Context 五个预测求平均，再应用规则；与历史 `results/context_oof_analysis/novel_metrics.csv` 的 Sentence/Context 基线数值完全一致。

**协议 B：来源组匹配单折**，先按唯一 target 的来源组、组大小和确定性算法分配 Fold，再取两个基础模型在相同 Fold 的预测；**不在五折间平均**。B 的阈值仍是 0.5，但因模型选择计算方式不同，B 分数不能与 A 数字混合，也不能选 A 或 B 中较高者作为“真正测试成绩”。

**两种协议都使用固定 20 词规则，用户本地实测：**

| Novel 协议 | 模型 | Macro-F1 | Accuracy | Human FP | Machine FN | Human FPR | Machine Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| 五折平均 | Sentence-matched | 0.885089 | 0.894967 | 1,098 | 129 | 0.238023 | 0.981751 |
| 五折平均 | Context | 0.906406 | 0.913970 | 935 | 70 | 0.202688 | 0.990098 |
| **五折平均** | **E2 Human-veto≥20** | **0.912574** | **0.919192** | **851** | 93 | **0.184479** | 0.986844 |
| 来源匹配单折 | Sentence-matched | 0.877963 | 0.888718 | 1,156 | 144 | 0.250596 | 0.979629 |
| 来源匹配单折 | Context | 0.895837 | 0.904640 | 1,027 | 87 | 0.222632 | 0.987693 |
| **来源匹配单折** | **E2 Human-veto≥20** | **0.905625** | **0.912943** | **907** | 110 | **0.196618** | 0.984439 |

**固定 E2 相对同协议 Context 的改变：** 五折平均 Macro-F1 **+0.006168**，Human FP **935→851（少 84）**、Machine FN **70→93（多 23）**，Human FPR **−0.018209**、Machine Recall **−0.003254**；来源匹配单折 Macro-F1 **+0.009788**，Human FP **1,027→907（少 120）**、Machine FN **87→110（多 23）**，Human FPR **−0.026013**、Machine Recall **−0.003254**。

**来源分组 Bootstrap（619 来源组、2,000 次、seed=2026；服务器本轮结果）：**

| Novel 协议 | ΔMacro-F1 观测 | 95% 分位区间 |
|---|---:|---:|
| 五折平均 E2 − Context | **+0.006168** | **[+0.003962,+0.008624]** |
| 来源匹配单折 E2 − Context | **+0.009788** | **[+0.006861,+0.012892]** |

**解释边界：** ① Novel 在此前多轮模型对比/错误分析中用过，因而这是**已使用开发资源上的固定规则复核**，不是全新独立测试；② Bootstrap 只在已固定的模型预测上重采样 source，未传播模型重训和研究者之前探索的全部不确定性；③ 与旧 Stacking 的 Novel Macro-F1 不同训练协议、不同折模型，不应把 E2 的两种 Novel 数字当作替代模型胜负证据。**实验结论可以用于课程报告的“有理有据的小创新”，不应升级成发表级别的泛化声明。**

**输出：** `results/e2_novel_fixed_rule/{metrics,class_errors,bootstrap}.csv` + `summary.json`；仅安全聚合统计，Git 忽略。

### 12.9 实验结论与无效主张的“黑名单”

**本项目目前支持的表述：**

1. 句子级 Human/Machine 分类已具备可复现模型和无标签测试的完整推理链，500/4,264 条数/顺序/阈值通用验收 PASS。
2. 同协议 Fold-matched Novel 上 SVM/RoBERTa 具有互补性，原有 Logistic Stacking 提高 Macro-F1，但需要承担少量 Machine FN 增加的代价。
3. 严格窗口清理的同协议 OOF 上，Context 整体改善 Macro-F1，却有 Fold 4 和 Human FP 的非均匀风险。
4. 在相同 Sentence/Context 训练和评估协议下，**固定长句 Human-veto** 相对 Context 降低 Human FP，并在有代价的情况下改善 Macro-F1；在重复使用的 Novel 两种协议下仍观察到同方向提升。

**不得在报告写出的主张：**

- “测试集 Stacking/E2 的 Accuracy 或 Macro-F1 已经达到 0.9174/0.9126”——**无标签 Test 没有真实分数**。
- “E2 击败 Stacking”——**没有同训练/切分/基线的 E4 公平验证**。
- “旧 Fold 4 学习式门控 AP≈1 证明 gate 泛化”——**目标泄漏，作废**。
- “20 词长度本身因果决定上下文伤害”——**观察相关，尚未控制全部混杂**。
- “Human FPR 获得严格统计保证”——**来源 bootstrap 非 conformal FPR guarantee**。
- “Novel 是独立未知测试集，95% CI 证明真实泛化”——**Novel 被重复用于开发**。
- “预测 JSON 是老师规定的官方提交格式”——**PDF 未说明，待教师其他通知**。

### 12.10 课程报告可以直接引用的最短结果表与实验文件索引

**建议报告重点一：** 同协议 Stacking/Fold-matched Novel 表（12.2）；重点二：同训练协议 E2 在严格配对 OOF 上的 5 指标表、每折与 Human FP/FN 风险权衡（12.7）；重点三：不调参的 E2 Novel 复核和条件性限制（12.8）。不必把大量每个探索脚本结果全放主文；技术细节放附录。

**适合制作的 3 张核心图（待绘制，不冒充已做）：**

1. **数据与类别/句长图：** Human/Machine 目标长度分布，配合总量与分箱，解释为什么引入句长但不把其当作因果证据。
2. **Context 错误风险转移图：** Sentence-matched→Context→Human-veto 的 Human FP、Machine FN 同图或并列图，显示“降低误报 vs 牺牲召回”的条件决策代价。
3. **E2 按 Fold 的收益图：** 五折 `ΔMacro-F1`（或对不同长度桶净纠错的条形图），突出 Fold 3 小增益、Fold 4 较明显变化，避免只报总体提升。

**主要可复现脚本：**

```text
scripts/check_stage1_artifacts.py
scripts/analyze_stacking_errors.py
scripts/verify_stacking_aggregates.py
scripts/predict.py                          # 单 SVM 干跑；非最终 Stacking
scripts/validate_test_predictions.py
scripts/predict_oof_stacking.py            # 冻结五折 Stacking 实际测试推理
scripts/experiment_e2_selective_context.py
scripts/validate_e2_fixed_rule_novel.py
```

**原始/汇总数据与配置定位：**

```text
data/raw/AIGTD_test_without_labels.json       # 服务器本地原始无标签测试
results/oof_fusion/novel_fusion_predictions.csv
results/oof_fusion/fusion_config.json
results/oof_fusion/fusion_models.joblib
results/oof_fusion/score_normalization.json
results/context_oof/sentence_matched_oof_predictions.csv
results/context_oof/context_oof_predictions.csv
results/context_oof/sentence_matched_novel_fold_predictions.csv
results/context_oof/context_novel_fold_predictions.csv
results/context_oof_analysis/novel_metrics.csv
results/stage1_local_audit/stacking_full/     # 私有逐句输出
results/e2_selective_context/                # E2 聚合结果
results/e2_novel_fixed_rule/                 # E2 Novel 聚合结果
```

**验收代码：** `tests/test_stage1_freeze.py`（8 项）、`tests/test_oof_stacking_inference.py`（6 项）、`tests/test_e2_selective_context.py`（7 项）、`tests/test_e2_novel_fixed_rule.py`（6 项）。这些数目反映各自单项测试文件，不应相加当成“全部课程单元测试覆盖率”。CI 的模拟神经推理检查不能替代用户服务器上真实 500 篇 GPU 前向。

---

## 13. 版本变更和后续记录格式

### 13.1 V2 → V3 的实际变更

- **纠正过时状态：** 从“Stacking corrected 尚待核验、无最终推理入口、E2 未开展”更新为真实本地 PASS 状态；同时保留 E4、正式 schema、图表、课程报告的未完成状态。
- **新增明确实验身份：** Stage-1 Stacking（阈值 0.585、不同模型训练/切分）与 E2 Sentence/Context（阈值 0.5、严格窗口相同训练对照）完全分开。
- **补齐实验台账：** `EXP-00/01/02`、`ENG-01—05`、`E2-01/02`，均有目的、样本、Fold、方法、关键参数、数值、代价、运行证据、局限和文件定位。
- **补齐工程验收：** 用户服务器 15 工件检查、8+6 项 Stage-1 测试、Novel replay、SVM/Stacking 16 句干跑、Stacking 全五折 500/4,264、输出结构 PASS。
- **补齐 E2：** 七项合成测试、同协议八条固定候选规则与跨折选择诊断、52,864 OOF/2,000 Bootstrap、固定规则 11,682 Novel 的两套协议/2,000 Bootstrap、六项合成测试。
- **明确风险红线：** 目标泄漏的无效高分、重复 Novel 的开发属性、Bootstrap 不能变成独立盲测、不同协议不可直排、没有官方 schema 不能声称已提交。

### 13.2 后续新增实验的最简登记模板（仅当真的需要再做实验时）

```markdown
#### EXP-XX｜[实验标题]（YYYY-MM-DD）
- 目标：本次要回答的唯一问题。
- 数据/协议：Train/OOF/Novel/Test；行数、文章记录、来源组、Fold；是否独立。
- 对照与设置：已有基线、模型权重、阈值来源、随机种子、训练/推理成本。
- 执行与检查：Git 分支/commit、命令、断言、日志、输出目录，PASS/FAIL。
- 结果：课程五项指标、FP/FN/FPR/Recall、分折/分组及 Bootstrap（适用才报）。
- 解释：有哪些改善、代价、混杂或失效；可以/不可以得出的结论。
- 决策：保留、回退、停止或进入下一步；为何值得继续。
```

**当前真实下一步不需要另开 EXP-XX：** 优先把第 12 节已经完成的可复核工作写成**短而有逻辑的正式报告**，制作少量关键图表，核对老师实际提交要求，保留最终 Stacking 的本地结果，不再重复用开发集寻找更好阈值。