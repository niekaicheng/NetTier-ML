import pandas as pd
import glob

print("="*80)
print("深度学习数据量基准对比")
print("="*80)

print("\n常见深度学习任务的数据量要求：")
print("  - 小型任务：10K - 50K 样本")
print("  - 中型任务：50K - 500K 样本")
print("  - 大型任务：500K - 5M 样本")
print("  - 超大型：5M+ 样本")

print("\n当前项目 (NetTier-ML)：")
print("  - 总样本量：2,313,810（231万）")
print("  - 特征维度：77 个特征")
print("  - 分类：大型任务级别")
print("  - 结论：完全适合深度学习")

print("\n" + "="*80)
print("各类别样本分布")
print("="*80)

total = 0
samples_per_class = []

for f in sorted(glob.glob('archive/*.parquet')):
    df = pd.read_parquet(f)
    samples = len(df)
    total += samples
    filename = f.split('\\')[-1].replace('-no-metadata.parquet', '')
    samples_per_class.append((filename, samples))
    print(f"  {filename:30s} {samples:9,} 样本")

print("-"*80)
print(f"  {'总计':30s} {total:9,} 样本")

print("\n" + "="*80)
print("类别平衡性分析")
print("="*80)

min_samples = min(s[1] for s in samples_per_class)
max_samples = max(s[1] for s in samples_per_class)
avg_samples = total / len(samples_per_class)

print(f"  最小类样本数：{min_samples:,}（{min_samples/total*100:.1f}%）")
print(f"  最大类样本数：{max_samples:,}（{max_samples/total*100:.1f}%）")
print(f"  平均类样本数：{avg_samples:,.0f}")
print(f"  不平衡比：{max_samples/min_samples:.2f}:1")

if max_samples/min_samples < 10:
    print("\n  类别分布较为均衡（不平衡比 < 10:1）")
else:
    print("\n  存在类别不平衡，建议使用类别权重或过采样")

print("\n" + "="*80)
print("深度学习模型建议")
print("="*80)

print("\n基于数据规模的推荐：")
print("  CNN/RNN/LSTM：完全适合")
print("  Transformer：完全适合（当前使用）")
print("  混合架构（CNN+Transformer）：完全适合（TransECA-Net）")
print("  预训练大模型（BERT）：可行但需权衡计算成本")
print("  图神经网络（GNN）：需要额外的图构建步骤")

print("\n训练策略建议：")
print("  1. 使用全量数据训练（无需采样）")
print("  2. 启用数据增强（可选，进一步提升鲁棒性）")
print("  3. 使用 Dropout/BatchNorm 防止过拟合")
print("  4. 学习率调度器 + Early Stopping")
print("  5. 交叉验证评估模型稳定性（可选）")

print("\n" + "="*80)
