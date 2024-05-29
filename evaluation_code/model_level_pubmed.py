from scipy.stats import pearsonr
import json
import pdb
import sys
import numpy as np
import tempfile
from scipy.stats import pearsonr, kendalltau

import logging
import shutil
import os
import json
import re

f = open('pubmed.json')

lines = f.readlines()
rouge1 = []
rouge2 = []
rougel = []

print('ROUGE\tBERT\tGEVAL\tQUEST\tACU\tFM(Llama)\tFM(GPT-3.5)\tFM(GPT-4)\tHumans')


for model_name in ['gpt35', 'llama2_70b', 'longt5', 'longt5_block', 'bigbird_pegasus', 'bigbird_pegasus_block']:
    humans = []
    gpt4_scores = []
    gpt4_few_scores = []
    gpt35_scores = []
    acu_scores = []
    rougel_scores = []
    gpt35_direct_scores = []
    quest_scores = []
    bert_scores = []
    delta_scores = []
    llama_scores = []
    geval_scores = []

    for index, line in enumerate(lines[:]):
        content = json.loads(line)
        human = content[f'{model_name}_human']
        humans.append(human)
        tgt = content['human']
        gpt4_scores.append(content[f'{model_name}_gpt4_fm'])
        gpt35_scores.append(content[f'{model_name}_gpt35_fm'])
        bert_scores.append(content[f'{model_name}_bert'])
        rougel_scores.append(content[f'{model_name}_newrougel'])
        quest_scores.append(content[f'{model_name}_questeval'])
        acu_scores.append(content[f'{model_name}_acu3'])
        delta_scores.append(content[f'{model_name}_delta'])
        llama_scores.append(content[f'{model_name}_llama'])
        geval_scores.append(content[f'{model_name}_geval'])

    print(
        f"{model_name}\t"
        f"{np.mean(rougel_scores):.4f}\t"
        f"{np.mean(bert_scores):.4f}\t"
        f"{np.mean(geval_scores):.4f}\t"
        f"{np.mean(quest_scores):.4f}\t"
        f"{np.mean(acu_scores):.4f}\t"
        f"{np.mean(llama_scores):.4f}\t"
        f"{np.mean(gpt35_scores):.4f}\t"
        f"{np.mean(gpt4_scores):.4f}\t"
        f"{np.mean(humans):.4f}"
    )

