import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import cross_validate, cross_val_predict, train_test_split, GridSearchCV
from sklearn import preprocessing
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.express as px
import utils
from sklearn import exceptions
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore', category=exceptions.ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

df = pd.read_excel('Updated Rotam Records - 30-06-2021.xlsx')

columns = ['S. No', "Patient's Name ", 'IPD', 'UHID', 'IPD.1', 'Age ', 'Sex',
	   'Diagnosis', 'Date', 'Time', 'Test Permformed', 'CT', 'A5', 'A10',
	   'A15', 'A20', 'A25', 'A30', 'CFT', 'MCF', 'MCF-t', 'alpha', 'LI 30',
	   'LI 45', 'LI 60', 'ml', 'CFR', 'LOT', 'CLR', 'AR5', 'AR10', 'AR15',
	   'AR20', 'AR25', 'AR30', 'MCE', 'ACF', 'G', 'TPI', 'MAXV', 'MAXV-t',
	   'AUC', 'LT']


df = df.drop(columns=['S. No', "Patient's Name ",'IPD.1','UHID','Date', 'Time','Test Permformed'])[3:]
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
diagnosis = df.groupby('IPD')['Diagnosis'].first()

numeric_cols = df.columns.drop(['Diagnosis','IPD'])
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
x = df.groupby('IPD').agg('mean')
x[numeric_cols] = x[numeric_cols].apply(pd.to_numeric, errors='coerce')

unknown_diagnosis = []
for i in x.index:
	if diagnosis[i] == None:
		unknown_diagnosis.append(i)

# print(x.columns)
x = x.dropna(axis='columns',thresh=0.86*len(x))
# print(x.columns)
x['Diagnosis'] = diagnosis
# df = df[x.columns]

x = x.drop(axis='rows',index=unknown_diagnosis)

# print(x.isna().sum())
x = x.fillna(x.mean())
# print(x.isna().sum())

# df = df.set_index('IPD')

# for patient in x.index:
#     for col in x.columns:
#         n=1
#         while n<df.groupby('IPD').size().loc[patient] and pd.isna(x.loc[patient][col]):
#             print(n)
#             print(x.loc[patient][col])
#             print(df.groupby('IPD'))
#             print(df.groupby('IPD').nth(n))
#             print(df.groupby('IPD').nth(n).loc[patient][col])
#             if pd.isna(df.groupby('IPD').nth(n).loc[patient][col]):
#                 n+=1
#             else:
#                 x.loc[patient][col] = df.groupby('IPD').nth(n).loc[patient][col]


# CLD categories:
# cld-ethanol = 448
# cld-nash = 150
# cld-hcv = 
# cld-hbv
# cld-cryptogenic

y_cld = []
y_ethanol = []
y_cld_type = []
for cat in ['nash','ethanol','hcv', 'hbv','cryptogenic']:
	count = 0
	for diag in x['Diagnosis']:
		if diag != None and 'cld' in diag and cat in diag:
			count += 1
	print('cld-'+cat,count)

# For binary classification (CLD vs non-CLD)
for diag in x['Diagnosis']:
	if diag!=None:
		if 'cld' in diag:
			y_cld.append(1)
		else:
			y_cld.append(0)

# For binary classification (CLD-ethanol vs non-ethanol)
for diag in x['Diagnosis']:
	if diag!=None:
		if 'cld' in diag and 'ethanol' in diag:
			y_ethanol.append(1)
		else:
			y_ethanol.append(0)

# For multiclass classification (CLD Type: ethanol vs nash vs infectious)
for diag in x['Diagnosis']:
	if 'cld' in diag and 'nash' in diag:
		# y_cld_type.append('nash')
		y_cld_type.append(0)
	elif 'cld' in diag and 'ethanol' in diag:
		# y_cld_type.append('ethanol')
		y_cld_type.append(1)
	else:
		# y_cld_type.append('infectious')
		y_cld_type.append(2)
# print(y_cld_type[:20])
# y_cld_type = preprocessing.LabelBinarizer().fit_transform(y_cld_type)

x = x.drop(columns=['Diagnosis'])
# df_final = x
# df_final['y_ethanol'] = y_ethanol
# df_final['y_cld_type'] = y_cld_type
# df_final.to_csv('preprocessed_patient_records.csv')
#Standardization of data
scaler = preprocessing.StandardScaler().fit(x)
# print(x)
X_scaled = scaler.transform(x)
# print(X_scaled)
# X_scaled = x
# sklearn - Regression and kfold model
kfold = model_selection.KFold(n_splits=10, random_state=100, shuffle=True)

def cluster_umap(data, y, num_classes=2,classification='bleeder'):
	kmeans = KMeans(n_clusters=num_classes)
	if classification == 'type':
		if num_classes ==2:
			t = ['CLD-not Ethanol', 'CLD-Ethanol']
		else:
			t = ['CLD-NASH', 'CLD-Ethanol','CLD-Infectious']
	elif classification=='bleeder':
		t = ['Non-bleeder', 'Bleeder']
	clusters = kmeans.fit_predict(data)
	# fit = umap.UMAP(metric='l2',random_state=42).fit_transform(data)
	fit = umap.UMAP(metric='l2').fit_transform(data)
	labels = list(map(lambda x: t[x], y))
	print(fit.shape)
	print('C:', min(fit[:,0]), max(fit[:,0]))
	print('C:', min(fit[:,1]), max(fit[:,1]))
	ac = px.scatter(x=fit[:,0], y=fit[:,1], color=[str(x) for x in labels])
	ac.write_image("clustering - actual.png")
	cl = px.scatter(x=fit[:,0], y=fit[:,1], color=[str(x) for x in clusters])
	cl.write_image("clustering - cluster labels.png")

def logistic_regression(X_scaled, y,x,num_classes=2,imbalance=0):
	# print(X_scaled.columns)
	if num_classes > 2:
		model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='none')
		X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=5,stratify=y)
		clf  = model.fit(X_train,y_train)
		y_test_pred = model.predict(X_test)
		y_train_pred = model.predict(X_train)
		print('Training accuracy: ', accuracy_score(y_train,y_train_pred))
		print('Testing accuracy: ', accuracy_score(y_test,y_test_pred))
		auc = utils.eval(y_test, y_test_pred, avg='macro',num_classes=3)
		utils.confuse(y_test,y_test_pred,'Confusion Matrix Logistic Regression (Multiclass)')

	else:
		print(sum(y),len(y))
		weights_ = []
		if imbalance==1:
			weight_1 = 1 - (sum(y)/len(y))
			weight_0 = 1 - ((1-sum(y))/len(y))
			weight_dict = {0:weight_0,1:weight_1}
			for i in y:
				weights_.append(weight_dict[i])
			model_kfold = LogisticRegression(penalty='none',max_iter=10000,class_weight=weight_dict)
		else:
			model_kfold = LogisticRegression(penalty='none',max_iter=10000)
		results_kfold = model_selection.cross_val_score(model_kfold, X_scaled, y, cv=kfold)
		predictions = cross_val_predict(model_kfold,X_scaled,y,cv=kfold,method="predict")
		results=cross_validate(model_kfold, X_scaled, y, cv=kfold, return_estimator=True, return_train_score=True, n_jobs=4)
		print('Training Accuracy',results['train_score'].mean()*100.0)
		print('Test Accuracy', results_kfold.mean()*100.0)
		auc = utils.eval(y, predictions)
		probs = cross_val_predict(model_kfold,X_scaled,y,cv=kfold,method="predict_proba")
		utils.aucroc2(probs, y, auc, "Logistic Regression")
		# Coef and Bias of the model
		model = LogisticRegression(penalty='none',max_iter=10000)
		# model.fit(X_scaled, y)
		if len(weights_)!=0:
			clf = model.fit(X_scaled,y,sample_weight=weights_)
		else:
			clf = model.fit(X_scaled,y)
		# utils.aucroc(model, X_scaled, y, auc, "Logistic Regression")
	coef = model.coef_[0]
	print('Bias', model.intercept_[0])
	dict_coef = {}
	for k in range(len(x.columns)):
		dict_coef[x.columns[k]] = coef[k]
	print(dict_coef)
	utils.bar_plot_from_dict(dict_coef)

def svm_cv(X_scaled, y, x, num_classes=2, kernelt = 'linear',imbalance=0):
	# X_scaled is training set and x is the complete set
	if num_classes>2:
		model = SVC(kernel = kernelt, max_iter=100000, probability=True,decision_function_shape='ovr')
		X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=5,stratify=y)
		clf  = model.fit(X_train,y_train)
		y_test_pred = model.predict(X_test)
		y_train_pred = model.predict(X_train)
		print('Training accuracy: ', accuracy_score(y_train,y_train_pred))
		print('Testing accuracy: ', accuracy_score(y_test,y_test_pred))
		auc = utils.eval(y_test, y_test_pred, avg='macro',num_classes=3)
		utils.interpret_shap(model,x,f'SHAP Feature Importance - SVM {kernelt}',f'svm_{kernelt}')
	else:
		weights_ = []
		if imbalance==1:
			weight_1 = 1 - (sum(y)/len(y))
			weight_0 = 1 - ((1-sum(y))/len(y))
			weight_dict = {0:weight_0,1:weight_1}
			for i in y:
				weights_.append(weight_dict[i])
			model_kfold = SVC(kernel = kernelt, max_iter=100000, probability=True,class_weight=weight_dict)
		else:
			model_kfold = SVC(kernel = kernelt, max_iter=100000, probability=True)
		results_kfold = model_selection.cross_val_score(model_kfold, X_scaled, y, cv=kfold, n_jobs=4)
		predictions = cross_val_predict(model_kfold,X_scaled,y,cv=kfold,method="predict")
		results=cross_validate(model_kfold, X_scaled, y, cv=kfold, return_estimator=True, return_train_score=True, n_jobs=4)
		print('Training Accuracy',results['train_score'].mean()*100.0)
		print('Test Accuracy', results_kfold.mean()*100.0)
		auc = utils.eval(y, predictions)
		probs = cross_val_predict(model_kfold,X_scaled,y,cv=kfold,method="predict_proba")
		utils.aucroc2(probs, y, auc, f"SVM ({kernelt} kernel)")
		model = SVC(kernel = kernelt, max_iter=100000, probability=True)
		if len(weights_)!=0:
			clf = model.fit(X_scaled,y,sample_weight=weights_)
		else:
			clf = model.fit(X_scaled,y)
		# model.fit(X_scaled,y)
		# utils.aucroc(model, X_scaled, y, auc, f"SVM ({kernelt} kernel)")
		x.loc[:,:] = X_scaled
		utils.interpret_shap(model,x,f'SHAP Feature Importance - SVM {kernelt}',f'svm_{kernelt}')

def decision_tree(X_scaled, y,x, num_classes=2,classification='bleeder',imbalance=0):
	if num_classes>2:
		model = DecisionTreeClassifier(random_state=0,max_depth=3)
		X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=5,stratify=y)
		clf  = model.fit(X_train,y_train)
		y_test_pred = model.predict(X_test)
		y_train_pred = model.predict(X_train)
		print('Training accuracy: ', accuracy_score(y_train,y_train_pred))
		print('Testing accuracy: ', accuracy_score(y_test,y_test_pred))
		auc = utils.eval(y_test, y_test_pred, avg='macro',num_classes=3)
		cn = ['CLD-NASH', 'CLD-Ethanol','CLD-Infectious']
	else:
		weights_ = []
		if imbalance==1:
			weight_1 = 1 - (sum(y)/len(y))
			weight_0 = 1 - ((1-sum(y))/len(y))
			weight_dict = {0:weight_0,1:weight_1}
			for i in y:
				weights_.append(weight_dict[i])
			model_kfold = DecisionTreeClassifier(max_depth=3,class_weight=weight_dict)
		else:
			model_kfold = DecisionTreeClassifier(max_depth=3)
		results_kfold = model_selection.cross_val_score(model_kfold, X_scaled, y, cv=kfold)
		predictions = cross_val_predict(model_kfold,X_scaled,y,cv=kfold,method="predict")
		results=cross_validate(model_kfold, X_scaled, y, cv=kfold, return_estimator=True, return_train_score=True, n_jobs=4)
		print('Training Accuracy',results['train_score'].mean()*100.0)
		print('Test Accuracy', results_kfold.mean()*100.0)
		auc = utils.eval(y, predictions)
		probs = cross_val_predict(model_kfold,X_scaled,y,cv=kfold,method="predict_proba")
		utils.aucroc2(probs, y, auc, "Decision Tree")
		model = DecisionTreeClassifier(random_state=0,max_depth=3)
		if len(weights_)!=0:
			clf = model.fit(X_scaled,y,sample_weight=weights_)
		else:
			clf = model.fit(X_scaled,y)
		# clf = model.fit(X_scaled,y)
		if classification == 'bleeder':
			cn = ['Non-bleeder', 'Bleeder']
		else:
			cn = ['CLD-not Ethanol', 'CLD-Ethanol']
	# Plotting the tree
	# utils.aucroc(model, X_scaled, y, auc, "Decision Tree")
	fig = plt.figure(figsize=(25,20))
	_ = plot_tree(clf, feature_names=list(x.columns), class_names=cn, 
		filled=True, proportion=True)
	fig.savefig('tree.png')


def xgboost(X_scaled, y, x, num_classes = 2,classification='bleeder',imbalance=0):
	y = np.array(y,dtype=int)
	print(np.unique(y))
	if num_classes == 2:
		model = xgb.XGBClassifier(max_depth=3,random_state=0, use_label_encoder=False, 
								colsample_bytree=0.3, alpha=3,eval_metric='logloss')
		X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=5,stratify=y)
		weights_ = []
		if imbalance==1:
			weight_1 = 1 - (sum(y)/len(y))
			weight_0 = 1 - ((1-sum(y))/len(y))
			weight_dict = {0:weight_0,1:weight_1}
			for i in y_train:
				weights_.append(weight_dict[i])
				
		if len(weights_)!=0:
			clf = model.fit(X_train,y_train,sample_weight=weights_)
		else:
			clf = model.fit(X_train,y_train)
		y_test_pred = model.predict(X_test)
		y_train_pred = model.predict(X_train)
		print('Training accuracy: ', accuracy_score(y_train,y_train_pred))
		print('Testing accuracy: ', accuracy_score(y_test,y_test_pred))
		# utils.aucroc(model, X_scaled, y, auc, "Decision Tree")
		auc = utils.eval(y_test, y_test_pred)
		fig = plt.figure(figsize=(25,20))
		xgb.plot_tree(clf, num_trees=0)
		if classification=='bleeder':
			fig.savefig('xgbtree_bleeder.png')
			cn = ['Non-bleeder', 'Bleeder']
		else:
			fig.savefig('xgbtree_ethanol.png')
			cn = ['CLD-not Ethanol', 'CLD-Ethanol']
		# utils.interpret_shap(model,x,'SHAP Feature Importance - XGBoost','xgboost')
	else:
		print('XGBoost Multiclass Classification')
		params = {
			"learning_rate"    : [0.10,0.20,0.3] ,
			"max_depth"        : [3],
			"colsample_bytree" : [0.25,0.5,0.75,1],
			"n_estimators"     : [50,100,200],
			"alpha"            : [0,0.25,0.5,1,3],
			"gamma"	           : [0,0.5,2.5,5]
		}
		model = xgb.XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss',objective='multi:softmax')
		# model = GridSearchCV(model, params, verbose=2)
		X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=5,stratify=y)
		# model.fit(X_train, y_train)
		# print(model.best_score_)
		# print(model.best_params_)
		clf = model.fit(X_train,y_train)
		y_test_pred = model.predict(X_test)
		y_train_pred = model.predict(X_train)
		print('Training accuracy: ', accuracy_score(y_train,y_train_pred))
		print('Testing accuracy: ', accuracy_score(y_test,y_test_pred))
		auc = utils.eval(y_test, y_test_pred, avg='macro',num_classes=3)
		utils.confuse(y_test,y_test_pred,'Confusion Matrix XGBoost (Multiclass)')
		fig = plt.figure(figsize=(25,20))
		xgb.plot_tree(clf, num_trees=0)
		cn = ['CLD-NASH', 'CLD-Ethanol','CLD-Infectious']
		fig.savefig('xgbtree_multi.png')
		# print(y_pred[:20])


