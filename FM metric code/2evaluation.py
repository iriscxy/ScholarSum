import json
import os
import pdb
import time

import openai
import re

openai.api_key = ''
title_set = set()
model_name = "gpt-4"
name = 'write.json'
if os.path.exists(f'{name}'):
    f = open(f'{name}')
    lines = f.readlines()
    for line in lines:
        content = json.loads(line)
        title = content['human']
        title_set.add(title)

fw = open(f'{name}', 'a')

f = open('aspect_extraction.json')
lines = f.readlines()

back_prompt = '''Using a less strict criterion, assess the alignment (1-3) between the two inputs. 

- 3: Input2 is generally consistent with Input1.
- 2: Input1 is not mentioned in Input2.
- 1: Input2 contradicts Input1.

Only return the number. 

Example 1:
Input1: the use of 2-[18f]fluoro-2-deoxy - d - glucose ( [ 18f]fdg ) may help to establish the antitumor activity of enzastaurin , a novel protein kinase c - beta ii ( pkc-ii ) inhibitor , in mouse xenografts .
Input2: Imaging techniques, such as positron emission tomography (PET), are important for diagnosing and monitoring cancer patients. The glucose analogue 2-[F]fluoro-2-deoxy-D-glucose (FDG) is commonly used as a tracer in PET imaging to assess tissue glucose utilization. FDG PET is widely used in diagnosing various types of cancer, and it is being evaluated as a tool to assess the effects of anticancer drugs. Enzastaurin is a novel compound that inhibits protein kinase C-beta (PKC-), which has been implicated in tumor growth.
Number: 3

Example 2:
Input1: nissen fundoplication is an effective treatment of gastroesophageal reflux in infants .\n laparoscopic procedures after previous laparotomy are technically more challenging .\n the role of laparoscopic nissen fundoplication after neonatal laparotomy for diseases unrelated to reflux is poorly described.
Input2: The article discusses the complex nature of gastroesophageal reflux in neonates and infants, which is often caused by a combination of developmental and anatomical factors. 
Number: 2

Example 3:
Input1: [ 18f]fdg pet imaging technique does not correlate with standard caliper assessments in xenografts to assess the antitumor activity of enzastaurin .
Input2: These findings suggest that [18F]FDG PET imaging is a useful tool for assessing the antitumor effects of novel compounds, such as enzastaurin, in preclinical studies.
Number: 1

'''

method_prompt = '''Assess the alignment (1-4) between the two inputs. 

- 4: Input2 generally covers the information present in Input1, or omits minor details from Input1.
- 3: Input2 omits important information from Input1.
- 2: Input1 is not mentioned in Input2.
- 1: Input2 contradicts Input1.

Only return the number. 

Example 1:
Input1: We analyzed the methylation status of protocadherin8 in 162 prostate cancer tissues and 47 benign prostatic hyperplasia tissues using methylation-specific PCR (MSP). The patients with prostate cancer were followed up for 15-60 months, and biochemical recurrence was defined as the period between radical prostatectomy and the measurement of 2 successive values of serum PSA level 0.2 ng/ml.
Input2: the promoter methylation status of protocadherin8 in 162 prostate cancer tissues and 47 normal prostate tissues was examined using methylation - specific pcr ( msp ) .   subsequently , the relationships between protocadherin8 methylation and clinicopathological features of prostate cancer patients and biochemical recurrence - free survival of patients were analyzed.
Number: 4

Example 2:
Input1: the present study included 515 patients admitted to the coronary care units or equivalent cardiology wards of the participating hospitals between 2011 and 2012 in north punjab , pakistan .   the analysis was focused on identifying the socioeconomic status , lifestyle , family history of mi , and risk factors ( i.e. hypertension , diabetes , smoking , and hyperlipidemia ) . a structured questionnaire was designed to collect data . the lipid profile was recorded from the investigation chart of every patient . for statistical analysis , the kruskal wallis , mann - whitney u , wilcoxon , and chi - square tests were used.
Input2: a population - based cross - sectional study was conducted in six regions in north punjab ( urban and rural patients ). data were collected using trained trained staff from the patients admitted in coronary care units or equivalent cardiology hospitals in the participating hospitals.
Number: 3

Example 3:
Input1: hyperglycemia , commencing on the first dose of the steroid given , persisted even after the discontinuation of steroids and improvement of other signs .   there were no signs of pancreatitis or type 1 diabetes clinically in laboratory tests .   her blood glucose levels were regulated at first with insulin and later with metformin . within 1 year of follow - up , still regulated with oral antidiabetics , she has been diagnosed with type 2 diabetes . 
Input2: The patient was treated with discontinuation of carbamazepine, antihistaminic and systemic steroids, and her hyperglycemia resolved with metformin treatment. The patient's lung, skin, liver, and renal findings regressed, and a patch test with carbamazepine was positive.
Number: 2

Example 4:
Input1: hyperglycemia , commencing on the first dose of the steroid given , persisted even after the discontinuation of steroids and improvement of other signs . there were no signs of pancreatitis or type 1 diabetes clinically in laboratory tests . within 1 year of follow - up , still regulated with oral antidiabetics , she has been diagnosed with type 2 diabetes .
Input2: the patient recovered without any sequelae.
Number: 1

'''


for index,line in enumerate(lines):
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
    for aspect in ['background','method', 'result', 'conclusion']:
        input1 = json.loads(content['ground_aspect'])[aspect]
        for name in ['gpt35', 'llama2_70b', 'longt5', 'longt5_block', 'bigbird_pegasus', 'bigbird_pegasus_block']:
            input2 = json.loads(content[f'{name}_aspect'])[aspect]
            if input1.strip() == '':
                content[f'{name}_{aspect}_comparison'] = '0'
            elif input2.strip() == '':
                content[f'{name}_{aspect}_comparison'] = '1'
            else:
                if aspect=='background' or aspect=='conclusion':
                    completion = openai.ChatCompletion.create(
                        model=model_name,
                        messages=[{"role": "user",
                                   "content": f"{back_prompt} Input1: '{input1}' \n Input2: '{input2}'"}])
                    content[f'{name}_{aspect}_comparison'] = completion['choices'][0]['message']['content']
                    print(f'{name} ' + content[f'{name}_{aspect}_comparison'])
                    time.sleep(10)
                else:
                    completion = openai.ChatCompletion.create(
                        model=model_name,
                        messages=[{"role": "user",
                                   "content": f"{method_prompt} Input1: '{input1}' \n Input2: '{input2}'"}])
                    content[f'{name}_{aspect}_comparison'] = completion['choices'][0]['message']['content']
                    print(f'{name} ' + content[f'{name}_{aspect}_comparison'])
                    time.sleep(10)

    json.dump(content, fw)
    fw.write('\n')
