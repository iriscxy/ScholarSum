from scipy.stats import pearsonr
import json
import scipy.stats as stats
import pdb
import sys
import tempfile
from scipy.stats import pearsonr, kendalltau
import pandas as pd

import logging
import shutil
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


f = open('arxiv.json')

lines = f.readlines()
rouge1 = []
rouge2 = []
rougel = []
humans = []
gpt4_fm_scores = []
gpt4_scores = []
gpt35_scores = []
gpt35_direct_scores = []
gpt35_few_scores = []
gpt35_fm_scores = []
bert_scores = []
delta_scores = []
newrouge1_scores = []
newrouge2_scores = []
newrougel_scores = []
questeval_scores = []
acu3_scores = []
acu2_scores = []
llama_scores = []
for model_name in ['gpt35', 'llama2_70b', 'factsum', 'bartlarge']:
    # print(model_name)
    for index, line in enumerate(lines[:]):
        content = json.loads(line)
        human = content[f'{model_name}_human']
        humans.append(human)
        tgt = content['human']
        naive_summ = content[f'{model_name}']
        gpt4_fm_scores.append(content[f'{model_name}_gpt4_fm'])
        bert_scores.append(content[f'{model_name}_bert'])
        delta_scores.append(content[f'{model_name}_delta'])
        newrouge1_scores.append(content[f'{model_name}_newrouge1'])
        newrouge2_scores.append(content[f'{model_name}_newrouge2'])
        newrougel_scores.append(content[f'{model_name}_newrougel'])
        questeval_scores.append(content[f'{model_name}_questeval'])
        acu3_scores.append(content[f'{model_name}_acu3'])
        gpt4_scores.append(content[f'{model_name}_gpt4'])
        gpt35_fm_scores.append(content[f'{model_name}_gpt35_fm'])
        llama_scores.append(content[f'{model_name}_llama'])

data = np.array([humans, gpt4_fm_scores, gpt35_fm_scores, llama_scores, gpt4_scores,
                 newrougel_scores,
                 bert_scores, questeval_scores, acu3_scores, delta_scores])
correlation_matrix = np.corrcoef(data, rowvar=True)

# Metric names
metric_names = ['Human', 'FM (GPT-4)', 'FM (3.5)','FM(Llama)','GPT-4',
                'ROUGE-L',
                'BERTSCORE', 'QuestEval','ACU','DELTA']  # Adjust as per your metrics

# Mask the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure()

# Generate a custom green colormap
cmap = sns.light_palette("#79C", as_cmap=True)

# Set the figure size
plt.figure()  # Adjust the values as needed

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, annot=True, square=True,
            fmt=".2f",
            xticklabels=metric_names, yticklabels=metric_names, annot_kws={"size": 8},cbar=False
            # cbar_kws={'shrink': 0.8,'pad':-0.05}
            )

# Adjust font size and rotate x labels
plt.xticks(rotation=45, fontsize=8)  # Adjust the rotation and fontsize as needed
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('arxiv.pdf')
plt.show()
