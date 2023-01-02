# Used for comparing the performance of pretrained NLP models and adding missing symptoms and comorb
import os
import zipfile
import re
# from xml.etree.cElementTree import XML
from lxml import etree
import docx
import unicodedata
from tqdm import tqdm
import pandas as pd
import numpy as np
import tempfile, shutil
from bs4 import BeautifulSoup

search_terms_bleeder = ['bleed','mucosal','cutaneous','hemorrhage','coagulopath','melena']
search_terms_non_bleeder = ['non bleed', 'non-bleed','not bleed', 'no bleed', 'variceal','esophageal','varices']
search_terms_cld_type = ['ethanol','cryptogenic','hcv','hbv','hepatitis c', 'hepatitis b', 'nash',
                        'nafl','nafld','nonalcoholic fatty','non alcoholic fatty','non-alcoholic fatty',
                        'nonalcoholic steatohepatitis','non-alcoholic steatohepatitis','non alcoholic steatohepatitis']
search_terms_excel = ['A5', 'A10', 'A15', 'A20', 'A25', 'A30', 'CFT', 'MCF', 'MCF-t', 'alpha', 'LI 30',
       'ml', 'CFR', 'LOT', 'CLR', 'AR5', 'AR10', 'AR15','AR20', 'AR25', 'AR30', 'MCE', 
       'ACF', 'G', 'TPI', 'MAXV', 'MAXV-t', 'AUC', 'LT'] 
regex_meld = ['meld - \d', 'meld -\d','meld- \d','meld-\d']
regex_ctp = ['ctp - \d', 'ctp -\d','ctp- \d','ctp-\d']
regex_child =  ['child - [abcd]', 'child-[abcd]', 'child -[abcd]', 'child- [abcd]']

search_terms_symptoms = ['pneumonia','clot','necrosis','sepsis','infection', 'sirs', 'sah','portal hypertension']
search_terms_diabetes = ['diabetes', 't2dm', 't2 dm']

path = "files/"
dir_list = os.listdir(path)
# print(df['IPD'])
patient_id_list = []
for fname in dir_list:
	pid = int(fname[:-4][-5:])
	patient_id_list.append(pid)


def get_outcome(): 
    # old version
    y_cld_type = []
    y_bleeder = []
    for patient_id in tqdm(patient_id_list):
        found_in_file = {}
        soup = BeautifulSoup(open(path+'IPID00'+str(patient_id)+'.doc').read(), 'html.parser')
        text = soup.get_text()
        text_lower = text.lower()
        for term in search_terms_bleeder:
            if (text_lower.find(term) != -1):
                found_in_file[term] = 1
            else:
                found_in_file[term] = 0       
        for term in search_terms_cld_type:
            if (text_lower.find(term) != -1):
                found_in_file[term] = 1
            else:
                found_in_file[term] = 0  
        b = 0 # 1: bleeder 0:non-bleeder
        c = 1 # variable for cld type 0:NASH, 1:ethanol, 2:infectious
        if (found_in_file['bleed'] == 1 and found_in_file['non bleed'] == 0 and found_in_file['non-bleed'] == 0 and
            found_in_file['not bleed'] == 0 and found_in_file['no bleed'] == 0):
            b = 1
        y_bleeder.append(b)
        if (found_in_file['hcv'] == 1 or found_in_file['hbv'] == 1 or found_in_file['hepatitis c'] == 1 or 
        found_in_file['hepatitis b']==1 or found_in_file['cryptogenic']==1):
            c = 2
        elif (found_in_file['nash'] == 1 or found_in_file['nafld'] == 1) :
            c = 0
        elif (found_in_file['ethanol'] == 1):
            c = 1
        elif (found_in_file['cld']==1):
            c = 2
        y_cld_type.append(c)

    df = pd.DataFrame()
    df['IPD'] = patient_id_list
    df['y_cld_type'] = y_cld_type
    df['y_bleeder'] = y_bleeder
    df.to_csv('file_patients_outcome.csv')

# def write_docx (xml_file_text, output_filename):
#     """ Create a temp directory, expand the original docx zip.
#         Write the modified xml to word/document.xml
#         Zip it up as the new docx
#     """
#     tmp_dir = tempfile.mkdtemp()
#     os.mkdir(os.path.join(tmp_dir,'word'))
#     with open(os.path.join(tmp_dir,'word/document.xml'), 'w') as f:
#         # xmlstr = etree.tostring(xml_content, pretty_print=True)
#         # xml_content.write(f)
#         f.write(xml_file_text)
#     # Now, create the new zip file and add all the filex into the archive
#     zip_copy_filename = output_filename
#     filenames = ['word/document.xml']
#     with zipfile.ZipFile(zip_copy_filename, "w") as docx:
#         for filename in filenames:
#             docx.write(os.path.join(tmp_dir,filename), filename)
#     # Clean up the temp dir
#     shutil.rmtree(tmp_dir)

def create_df():
    # new version
    y_bleeder, y_cld, y_diabetes = [], [], []
    y_symptoms = {'pneumonia':[],'clot':[],'necrosis':[],'sepsis':[],'infection':[], 'sirs':[], 'sah':[],'portal hypertension':[]}
    meld, ctp, child = [], [], []
    clinical_parameters = {'hb':[],'tlc':[],'plt':[],'pt':[],'inr':[],'bu':[],'s.creat':[],
                            's.bil':[],'ast':[],'alt':[],'sap':[],'ggtp':[],'alb':[]}
    for patient_id in tqdm(patient_id_list):
        found_in_file = []
        soup = BeautifulSoup(open(path+'IPID00'+str(patient_id)+'.doc').read(), 'html.parser')
        text = soup.get_text()
        text_lower = text.lower()

        for term in search_terms_bleeder+search_terms_non_bleeder:
            if (text_lower.find(term) != -1):
                found_in_file.append(1)
            else:
                found_in_file.append(0) 
        if sum(found_in_file[6:])>0:
            y_bleeder.append(0)
        elif sum(found_in_file[:6])>0 and sum(found_in_file[6:])==0:
            y_bleeder.append(1)
        else:
            y_bleeder.append(0)

        found_in_file=[]
        for term in search_terms_cld_type:
            if (text_lower.find(term) != -1):
                found_in_file.append(1)
            else:
                found_in_file.append(0)
        if sum(found_in_file[1:6])>0:
            y_cld.append('infectious')
        elif sum(found_in_file[6:])>0:
            y_cld.append('nash')
        elif found_in_file[0]>0:
            y_cld.append('ethanol')
        else:
            y_cld.append('infectious')
        
        found_in_file = []
        for term in search_terms_diabetes:
            if (text_lower.find(term) != -1):
                found_in_file.append(1)
            else:
                found_in_file.append(0)
        if sum(found_in_file)>0:
            y_diabetes.append(1)
        else:
            y_diabetes.append(0)
        
        for term in search_terms_symptoms:
            if (text_lower.find(term) != -1):
                y_symptoms[term].append(1)
            else:
                y_symptoms[term].append(0)
        
        found = False
        for reg_exp in regex_meld:
            f = re.findall(reg_exp,str(text_lower))
            if (len(f)>0):
                meld.append(int(f[0].split('-')[-1]))
                found = True
                break
        if not found:
            meld.append(-1)

        found = False
        for reg_exp in regex_ctp:
            f = re.findall(reg_exp,str(text_lower))
            if (len(f)>0):
                ctp.append(int(f[0].split('-')[-1]))
                found = True
                break
        if not found:
            ctp.append(-1)

        found = False
        for reg_exp in regex_child:
            f = re.findall(reg_exp,str(text_lower))
            if (len(f)>0):
                child.append(f[0].split('-')[-1])
                found = True
                break
        if not found:
            child.append(-1)

    df = pd.DataFrame()
    df['IPD'] = patient_id_list
    df['cld_type'] = y_cld
    df['y_bleeder'] = y_bleeder
    df['diabetes'] = y_diabetes
    df['MELD'] = meld
    df['CTP'] = ctp
    df['CHILD'] = child
    for key, value in y_symptoms.items():
        df[key] = value
    df.to_csv('patients.csv')

create_df()


        
