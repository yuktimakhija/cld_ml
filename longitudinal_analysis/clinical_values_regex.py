import re 
import os
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

path = "files/"
dir_list = os.listdir(path)
# print(df['IPD'])
patient_id_list = []
for fname in dir_list:
	pid = int(fname[:-4][-5:])
	patient_id_list.append(pid)

regex_four = r'\s[a-z.]+\/[a-z.]+\/[a-z.]+\/[a-z.]+-\d+\.?\d*\/\d+\.?\d*\/\d+\.?\d*\/\d+\.?\d*'
regex_three = r'\s[a-z.]+\/[a-z.]+\/[a-z.]+-\d+\.?\d*\/\d+\.?\d*\/\d+\.?\d*'
regex_two = r'\s[a-z.]+\/[a-z.]+-\d+\.?\d*\/\d+\.?\d*'
regex_one = r'\s[a-z.]+-\d+\.?\d*'
names_regex = r'[a-z.]+\/[a-z.]+\/[a-z.]+-'

clinical_parameters_values = {'hb':[],'tlc':[],'plt':[],'pt':[],'inr':[],'bu':[],'creat':[],
                            'na':[],'k':[],'cl':[],'bil':[],'ast':[],'alt':[],'sap':[],'ggt':[],
                            'alb':[],'lvef':[]}

for patient_id in tqdm(patient_id_list):
    soup = BeautifulSoup(open(path+'IPID00'+str(patient_id)+'.doc').read(), 'html.parser')
    text = soup.get_text()
    text_lower = text.lower()
    text_lower = re.sub('-\s+','-',str(text_lower))
    text_lower = re.sub('\s+-','-',str(text_lower))
    text_lower = re.sub('\/\s+','/',str(text_lower))
    text_lower = re.sub('\s+\/','/',str(text_lower))
    text_lower = re.sub('\s+:','-',str(text_lower))
    text_lower = re.sub(':\s+','-',str(text_lower))
    # print(text_lower)
    # test = r'\s[a-z.]+\/[a-z.]+-\d+\.?\d*\/\d+\.?\d*'
    clinical_parameters_list = ['hb','tlc','plt','pt','inr','bu','creat',
                            'na','k','cl','bil','ast','alt','sap','ggt',
                            'alb','lvef']
    patient_values = {'hb':-1,'tlc':-1,'plt':-1,'pt':-1,'inr':-1,'bu':-1,'creat':-1,
                            'na':-1,'k':-1,'cl':-1,'bil':-1,'ast':-1,'alt':-1,'sap':-1,'ggt':-1,
                            'alb':-1,'lvef':-1}
    for regex_exp in [regex_four,regex_three,regex_two,regex_one]:
        # print(re.findall(regex_exp,str(text_lower)))
        found = re.findall(regex_exp,str(text_lower))
        if len(found)!=0:
            for f in found:
                # print(f.split('-'))
                feature_names, feature_values = f.split('-')[0].split('/'), f.split('-')[1].split('/')
                idx = -1
                for name in feature_names:
                    # print(name)
                    idx+=1
                    feature = name.replace(' ','').replace('.','')
                    for p in clinical_parameters_list:
                        if p in feature:
                            feature = p
                            break
                    if feature not in clinical_parameters_list:
                        continue
                    if patient_values[feature] != -1:
                        continue
                    else:
                        patient_values[feature] = float(feature_values[idx])
    for key, value in patient_values.items():
        clinical_parameters_values[key].append(value)
    # print(patient_values)

df = pd.read_csv('patients.csv').set_index('IPD')
for key, value in clinical_parameters_values.items():
    df[key] = value
df.to_csv('patients.csv')
