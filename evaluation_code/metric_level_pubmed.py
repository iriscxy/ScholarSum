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


f = open('pubmed.json')

lines = f.readlines()
rouge1 = []
rouge2 = []
rougel = []
humans = []
gpt4_wo = []
gpt4_fm = []
gpt35_fm = []
llama_fm = []
gpt4 = []
bert_scores = []
backgrounds = []
methods = []
results = []
conclusions = []

delta_scores = []
newrouge1_scores = []
newrouge2_scores = []
newrougel_scores = []
questeval_scores = []
acu3_scores = []
acu2_scores = []
geval_scores = []
llama_fm = []
llama = []
for model_name in ['gpt35', 'llama2_70b', 'longt5', 'longt5_block', 'bigbird_pegasus', 'bigbird_pegasus_block']:
    for index, line in enumerate(lines[:]):
        content = json.loads(line)
        [background, method, result, conclusion] = content[f'{model_name}_human_list']
        backgrounds.append(background)
        methods.append(method)
        results.append(result)
        conclusions.append(conclusion)
        human = content[f'{model_name}_human']
        humans.append(human)
        tgt = content['human']
        naive_summ = content[f'{model_name}']
        gpt4_fm.append(content[f'{model_name}_gpt4_fm'])
        gpt4_wo.append(content[f'{model_name}_gpt4_wo'])
        gpt35_fm.append(content[f'{model_name}_gpt35_fm'])

        gpt4.append(content[f'{model_name}_gpt4'])
        bert_scores.append(content[f'{model_name}_bert'])
        delta_scores.append(content[f'{model_name}_delta'])
        newrouge1_scores.append(content[f'{model_name}_newrouge1'])
        newrouge2_scores.append(content[f'{model_name}_newrouge2'])
        newrougel_scores.append(content[f'{model_name}_newrougel'])
        questeval_scores.append(content[f'{model_name}_questeval'])
        acu3_scores.append(content[f'{model_name}_acu3'])
        geval_scores.append(content[f'{model_name}_geval'])
        llama_fm.append(content[f'{model_name}_llama'])
        llama.append(content[f'{model_name}_llama_base'])
data = np.array([humans, gpt4_fm, gpt4_wo, gpt35_fm, llama_fm, gpt4, llama,
                 newrougel_scores, bert_scores, geval_scores,
                 questeval_scores, acu3_scores, delta_scores])

correlation_matrix = np.corrcoef(data, rowvar=True)

metric_names = ['Human', 'FM (GPT-4)','FM (GPT-4)\n w/ few.','FM (GPT-3.5)','FM(Llama)','GPT-4','Llama',
               'ROUGE-L','BERTScore','GEval',
                 'QuestEval','ACU' ,'DELTA']
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure()

# Generate a custom green colormap
cmap = sns.light_palette("seagreen", as_cmap=True)

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
plt.savefig('matrix.pdf')
plt.show()
