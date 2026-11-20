AIGTD Novel Validation 更新包
================================

一、替换文件

把压缩包解压到 AIGTD 项目根目录，选择覆盖同名文件：

scripts/common.py
scripts/evaluate.py
scripts/train_svm.py
scripts/train_bert.py
scripts/train_context.py
scripts/train_fusion.py
scripts/transformer_utils.py
scripts/error_analysis.py

二、核心逻辑

1. 新增 Novel Validation：验证句子经 Unicode NFKC + casefold + 连续空白归一化后，若训练集不存在该句子，则标记为 novel。
2. 标点和词序保留，和 audit_duplicates.py 的精确重复检测规则一致。
3. 上下文模型仍使用完整文章序列构建上下文；只在计算指标和阈值搜索时筛选 novel 目标句子，不删句、不重排。
4. 阈值以 Novel Validation Macro-F1 为主目标搜索。
5. BERT/上下文 Transformer 的 early stopping 以 Novel Validation Macro-F1@0.5 选择 checkpoint。
6. Full Validation 使用相同阈值计算五项指标，作为辅助报告。
7. evaluate.py 返回的未加前缀五项指标（accuracy/precision/recall/macro_f1/micro_f1）代表 Novel Validation；full_* 和 novel_* 字段都保存在 metrics.csv/json。
8. error_analysis.py 默认把 Novel Validation 的 FP/FN 写入 error_analysis.csv，同时另存完整验证集及 seen-in-train 子集错误。

三、无需重新预处理

现有 data/splits/train.csv 和 validation.csv 可以继续使用。只要 CSV 含有 article_id、article_key、sentence_index、text、label_id 字段即可。

四、建议运行顺序

python scripts/train_svm.py
python scripts/error_analysis.py
python scripts/train_bert.py
python scripts/train_context.py
python scripts/train_fusion.py

Transformer 模型需要环境中已安装 torch、transformers、相关 tokenizer 依赖，且首次运行需要能够访问本地缓存或模型下载源。

五、主要输出

results/metrics.csv                         汇总各模型 Full/Novel 指标
results/svm_tfidf_metrics.json              SVM 双口径评估明细
results/svm_tfidf_thresholds.csv            阈值扫描结果
results/svm_tfidf_predictions.csv           完整验证集预测，含 is_novel_validation
results/svm_tfidf_novel_predictions.csv     仅 Novel Validation 的预测
results/confusion_matrix_full.png           完整验证集混淆矩阵
results/confusion_matrix_novel.png          Novel Validation 混淆矩阵
results/confusion_matrix.png                兼容旧文件名，内容为 Novel Validation
results/threshold_analysis.png              Novel 阈值优化曲线，同时显示 Full Macro-F1

注意：Novel Validation 目前作为主要模型选择口径。由于它仍来自同一验证集，反复围绕它调参也会逐渐造成验证集过拟合；最终比较不同方案时应保持实验次数和协议透明，最好保留独立测试集用于最终预测/评测。
