import json
import os
import pdb
import time

import openai
import re

openai.api_key = ''
title_set = set()
model_name = "gpt-4"
name = 'aspect_extraction.json'

if os.path.exists(f'{name}'):
    f = open(f'{name}')
    lines = f.readlines()
    for line in lines:
        content = json.loads(line)
        title = content['human']
        title_set.add(title)

fw = open(f'{name}', 'a')

f = open('pubmed.json')
lines = f.readlines()



for line in lines:
    content = json.loads(line)
    gpt35 = content['gpt35']
    llama2_70b = content['llama2_70b']

    longt5 = content['longt5']
    longt5_block = content['longt5_block']

    bigbird_pegasus = content['bigbird_pegasus']
    bigbird_pegasus_block = content['bigbird_pegasus_block']

    abstract = content['human']

    if abstract in title_set:
        continue

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user",
                   "content": f"What is the background/method/result/conclusion of this work? Extract segment of the input as the answer. Return the answer in json format, where the key is background/method/result/conclusion. If any category is not represented in the input, its value should be left empty. Input: {abstract}"}])
    content['human_aspect'] = completion['choices'][0]['message']['content']
    print('ground')

    time.sleep(1)

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user",
                   "content": f"What is the background/method/result/conclusion of this work? Extract segment of the input as the answer. Return the answer in json format, where the key is background/method/result/conclusion. If any category is not represented in the input, its value should be left empty. Input: {gpt35}"}])
    content['gpt3.5_aspect'] = completion['choices'][0]['message']['content']
    print('gpt')
    time.sleep(1)

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user",
                   "content": f"What is the background/method/result/conclusion of this work? Extract segment of the input as the answer. Return the answer in json format, where the key is background/method/result/conclusion. If any category is not represented in the input, its value should be left empty. Input: {llama2_70b}"}])
    content['llama2_70b_aspect'] = completion['choices'][0]['message']['content']
    print('llama')
    time.sleep(1)

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user",
                   "content": f"What is the background/method/result/conclusion of this work? Extract segment of the input as the answer. Return the answer in json format, where the key is background/method/result/conclusion. If any category is not represented in the input, its value should be left empty. Input: {longt5}"}])
    content['longt5_aspect'] = completion['choices'][0]['message']['content']
    print('longt5')
    time.sleep(1)

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user",
                   "content": f"What is the background/method/result/conclusion of this work? Extract segment of the input as the answer. Return the answer in json format, where the key is background/method/result/conclusion. If any category is not represented in the input, its value should be left empty. Input: {bigbird_pegasus}"}])
    content['bigbird_pegasus_aspect'] = completion['choices'][0]['message']['content']
    print('bigbird_pegasus')
    time.sleep(1)

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user",
                   "content": f"What is the background/method/result/conclusion of this work? Extract segment of the input as the answer. Return the answer in json format, where the key is background/method/result/conclusion. If any category is not represented in the input, its value should be left empty. Input: {longt5_block}"}])
    content['longt5_block_aspect'] = completion['choices'][0]['message']['content']
    print('longt5_block_aspect')
    time.sleep(1)

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user",
                   "content": f"What is the background/method/result/conclusion of this work? Extract segment of the input as the answer. Return the answer in json format, where the key is background/method/result/conclusion. If any category is not represented in the input, its value should be left empty. Input: {bigbird_pegasus_block}"}])
    content['bigbird_pegasus_block_aspect'] = completion['choices'][0]['message']['content']
    print('bigbird_pegasus_block_aspect')
    time.sleep(1)

    json.dump(content, fw)
    fw.write('\n')
