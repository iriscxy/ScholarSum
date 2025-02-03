import json
import numpy as np
from scipy.stats import ttest_ind
import pandas as pd
from math import sqrt
# 读取 JSON 数据
with open('../dataset/pubmed.json', 'r') as f:
    lines = f.readlines()

# 指标和模型的列表
metrics = [
    'gpt4_fm', 'gpt4_wo', 'gpt35_fm', 'gpt4', 'bert', 'delta',
    'newrouge1', 'newrouge2', 'newrougel', 'questeval','acu3','geval','llama'
]
models = ['llama2_70b', 'longt5', 'longt5_block', 'bigbird_pegasus', 'bigbird_pegasus_block']

# 初始化每个指标的字典，用于存储不同模型的分数
all_metrics_scores = {metric: {model: [] for model in models} for metric in metrics}

# 读取并解析数据
for line in lines:
    content = json.loads(line)
    for metric in metrics:
        for model_name in models:
            all_metrics_scores[metric][model_name].append(content[f'{model_name}_{metric}'])

# 将分数转换为 NumPy 数组
for metric in metrics:
    for model in models:
        all_metrics_scores[metric][model] = np.array(all_metrics_scores[metric][model])

# 定义 bootstrap 函数计算统计功效
def bootstrap_statistical_power(scores1, scores2, num_bootstrap_samples=1000, alpha=0.05):
    significant_count = 0
    for _ in range(num_bootstrap_samples):
        sample1 = np.random.choice(scores1, size=len(scores1), replace=True)
        sample2 = np.random.choice(scores2, size=len(scores2), replace=True)
        _, p_value = ttest_ind(sample1, sample2)
        if p_value < alpha:
            significant_count += 1
    return significant_count / num_bootstrap_samples

# 计算每个指标的统计功效
average_power_results = []
for metric in metrics:
    power_results = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1, model2 = models[i], models[j]
            scores1, scores2 = all_metrics_scores[metric][model1], all_metrics_scores[metric][model2]
            power = bootstrap_statistical_power(scores1, scores2)
            power_results.append(power)
    # 计算每个指标的平均统计功效
    average_power = np.mean(power_results)
    average_power_results.append({'Metric': metric, 'Average Statistical Power': average_power})

# 将结果存储为 DataFrame 并显示
power_df = pd.DataFrame(average_power_results)
print(power_df)

from numpy import mean, std
from scipy.stats import ttest_ind

def cohen_d(scores1, scores2):
    n1, n2 = len(scores1), len(scores2)
    pooled_std = sqrt(((n1-1)*std(scores1)**2 + (n2-1)*std(scores2)**2) / (n1 + n2 - 2))
    return (mean(scores1) - mean(scores2)) / pooled_std

# 计算所有模型对和指标的效应量
for metric in metrics:
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            m1, m2 = models[i], models[j]
            d = cohen_d(all_metrics_scores[metric][m1], all_metrics_scores[metric][m2])
            print(f"{metric}: {m1} vs {m2} | Cohen's d = {d:.3f}")