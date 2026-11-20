AIGTD 重复句子与潜在数据泄漏审计脚本

文件：scripts/audit_duplicates.py

安装要求：沿用现有 AIGTD 项目的环境与 requirements.txt；需要 pandas、PyYAML。

使用方式：
1. 将 audit_duplicates.py 放入项目根目录下的 scripts/ 目录。
2. 确认已成功运行 python scripts/preprocess.py。
3. 在项目根目录执行：
   python scripts/audit_duplicates.py

可选配置：
   python scripts/audit_duplicates.py --config configs/experiment.yaml

输出目录：
   results/duplicate_audit/

重点查看：
- summary.txt：总体审计摘要
- train_validation_sentence_overlap.csv：训练/验证集完全重复句子
- conflicting_duplicate_labels.csv：相同文本同时带 human 和 machine 标签
- cross_split_exact_article_duplicates.csv：跨 split 完全相同文章（句子顺序一致）
- cross_split_article_sentence_set_duplicates.csv：跨 split 句子集合相同（忽略句子顺序）
- development_test_sentence_overlap.csv：开发集与无标签测试集重复句子
- train_test_shared_article_ids.csv：训练/测试共享 article_id 的内容重叠情况
- duplicate_sentence_occurrences.csv：重复句子的全部出现位置

该脚本只报告，不删除数据、不修改原始文件、不自动重新划分数据。
