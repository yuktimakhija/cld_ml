# Used for bleeder prediction

import os
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection

def excel_preproc(df,col):
	# print(df)
	df = df.drop(columns=['S. No', "Patient's Name ",'IPD.1','UHID','Date', 'Time','Test Permformed'])[3:]
	# print(df)
	df['Diagnosis']= df['Diagnosis'].str.lower()
	df = df.replace({'---': np.nan}, regex=True)
	df = df.replace('\*', '', regex=True)
	df = df.replace(',', '', regex=True)
	df['Sex'] = df['Sex'].replace('M',1)
	df['Sex'] = df['Sex'].replace('F',0)
	df['Sex'] = df['Sex'].replace('Years',0)
	df['Sex'] = df['Sex'].fillna(0)

	df['Age '] = df['Age '].replace(regex={r"\D+": 1})
	df['Age '] = df['Age '].fillna(df['Age '].mean())

	# x = df.groupby('IPD').first()
	diagnosis = df.groupby(df.index)['Diagnosis'].first()

	numeric_cols = df.columns.drop(['Diagnosis'])
	# numeric_cols = df.columns.drop(['Diagnosis','IPD'])
	df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
	x = df.groupby(df.index).agg('mean')
	# print(x)
	x[numeric_cols] = x[numeric_cols].apply(pd.to_numeric, errors='coerce')

	x = x[col]

	return x

def find_common_patients(df1,df2):
    idx = df1.index.intersection(df2.index)
    return idx

def merge_df(df1,df2,idx):
    return pd.concat([df1.loc[idx],df2.loc[idx]],axis=1)

df_excel_raw = pd.read_excel('Updated Rotam Records - 30-06-2021.xlsx').set_index('IPD')
clotting_parameters_csv = pd.read_csv('preprocessed_patient_records.csv').set_index('IPD')
final_clotting_parameters = list(clotting_parameters_csv.drop(columns=['y_ethanol','y_cld_type']).columns)
df_excel = excel_preproc(df_excel_raw,final_clotting_parameters)

df_ehr_files = pd.read_csv('patients.csv').set_index('IPD')
common_patients_ipd = find_common_patients(df_excel_raw,df_ehr_files)
df_merged = merge_df(df_excel,df_ehr_files,common_patients_ipd)

# print(df_merged)
# print(df_merged.columns)
# for col in df_merged.columns:
#     print(df_merged[col].value_counts())
final_df = df_merged
for col in df_merged.columns:
    if -1 in df_merged[col].unique():
        if df_merged[col].value_counts().loc[-1] > 0.35*len(df_merged):
            # print(df_merged[col].value_counts())
            final_df = final_df.drop(columns=[col]) 

for col in final_df:
    if -1 in final_df[col].unique():
        final_df[col] = final_df[col].replace([-1.0],final_df[col].mean())

# print(final_df)

# for col in final_df.columns:
#     print(final_df[col].value_counts())

y = final_df['y_bleeder']
ind_0 = np.where(y==0)[0][:205]
ind_1 = np.where(y==1)[0]
# print(len(ind_0))

final_df = final_df.iloc[np.concatenate([ind_0,ind_1])]
final_df = final_df.sample(frac=1)
outcome = final_df['y_bleeder']
final_df = final_df.drop(columns=['y_bleeder','Unnamed: 0'])
final_df = pd.get_dummies(final_df, columns = ['CHILD', 'cld_type'])
x = final_df.fillna(final_df.mean())
# enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
scaler = preprocessing.StandardScaler().fit(x)
X_scaled = scaler.transform(x)
kfold = model_selection.KFold(n_splits=10, random_state=100, shuffle=True)

# print(final_df.columns)

# Training ML models
import models


# Logistic Regression
print('LR')
models.logistic_regression(X_scaled,outcome,x,imbalance=1)
# print('SVM Linear')
# models.svm_cv(X_scaled,outcome,x, kernelt='linear',imbalance=1)
# print('SVM RBF')
# models.svm_cv(X_scaled,outcome, x, kernelt='rbf',imbalance=1)
# print('UMAP')
# models.cluster_umap(X_scaled,outcome,num_classes=2,classification='bleeder')
# print('DT')
# models.decision_tree(X_scaled,outcome,x,imbalance=1)
# print('XGB')
# models.xgboost(X_scaled,outcome, x,imbalance=1)
