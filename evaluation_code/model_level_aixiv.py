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

f = open('arxiv.json')

lines = f.readlines()
rouge1 = []
rouge2 = []
rougel = []

print('ROUGE' + '    ' + 'BERT' + '    ' + 'DELTA' + '    ' + 'QUEST' + '    ' + 'ACU' + '    ' +
       'FM (GPT-3.5)' + '    ' + 'Fm(GPT-4)' +  '    ' + 'Humans')

for model_name in ['gpt35', 'llama2_70b', 'factsum', 'bartlarge']:
    humans = []
    gpt4_scores = []
    gpt4_fm_scores = []
    gpt35_fm_scores = []
    acu_scores = []
    rougel_scores = []
    gpt35_direct_scores = []
    quest_scores = []
    bert_scores = []
    delta_scores = []

    for index, line in enumerate(lines[:]):
        content = json.loads(line)
        human = content[f'{model_name}_human']
        humans.append(human)
        tgt = content['human']
        gpt4_fm_scores.append(content[f'{model_name}_gpt4_fm'])
        gpt35_fm_scores.append(content[f'{model_name}_gpt35_fm'])
        bert_scores.append(content[f'{model_name}_bert'])
        rougel_scores.append(content[f'{model_name}_newrougel'])
        quest_scores.append(content[f'{model_name}_questeval'])
        acu_scores.append(content[f'{model_name}_acu3'])
        delta_scores.append(content[f'{model_name}_delta'])

    print(model_name + '    ' + str(np.mean(rougel_scores)) + '    ' + str(np.mean(bert_scores)) +
          str(np.mean(delta_scores)) + '    ' + str(np.mean(quest_scores)) + '    ' + str(np.mean(acu_scores)) + '    ' +
       '    ' + str(np.mean(gpt35_fm_scores)) + '    ' + str(
        np.mean(gpt4_fm_scores)) + '    ' +
          '    ' + str(np.mean(humans)))


