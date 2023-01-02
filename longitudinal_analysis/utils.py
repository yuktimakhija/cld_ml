import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import auc, accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, f1_score, roc_curve, precision_recall_curve
from sklearn import preprocessing
from matplotlib.colors import ListedColormap
import numpy as np
import shap 
import pandas as pd
import plotly.express as px

# CONFUSION MATRIX
def confuse(df_target,results, name):
	cmat = confusion_matrix(df_target, results)
	a = []
	for row in cmat:
		a.append(list(row))
	# print(cmat,type(cmat))
	# print(a)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(a)
	for (i, j), z in np.ndenumerate(cmat):
		ax.text(j, i, '{:0.1f}'.format(z))
	fig.colorbar(cax)
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	ax.xaxis.label.set_size(22)
	ax.yaxis.label.set_size(22)
	ax.tick_params(axis='both', labelsize=22)
	plt.title(name)
	plt.savefig(f'cm {name}.png')
	plt.close()
	plt.clf()

# EVALUATION PARAMETERS (F1 Score, Recall, Precision)
def eval(target,pred,avg='binary',num_classes = 2):
	print('F-score:',f1_score(target,pred,average=avg))
	print('Recall: ',recall_score(target,pred,average=avg))
	print('Precision: ',precision_score(target,pred, average=avg))
	if num_classes==2:
		print('AUCROC: ', roc_auc_score(target,pred))
		return roc_auc_score(target,pred)
	else:
		pred = preprocessing.LabelBinarizer().fit_transform(pred)
		target = preprocessing.LabelBinarizer().fit_transform(target)
		print('AUCROC: ', roc_auc_score(target,pred,multi_class='ovr'))
		return roc_auc_score(target,pred,multi_class='ovr')

# AUC-ROC Plot
def aucroc(model, df_test, test_target, auc, name):
	lr_probs = model.predict_proba(df_test)
	lr_probs = lr_probs[:, 1]
	fpr, tpr, thresh = roc_curve(test_target, lr_probs)
	# fpr_i, tpr_i, thresh_i = roc_curve(test_target, test_target)
	# fpr_n, tpr_n, thresh_n = roc_curve(test_target, [0.5]*len(test_target))
	plt.plot(fpr, tpr)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(name)
	plt.legend()
	plt.savefig(f'aucroc {name}.png')
	plt.close()
	plt.clf()
	
# AUC-ROC2 Plot
def aucroc2(lr_probs, test_target, auc, name):
	lr_probs = lr_probs[:, 1]
	fpr, tpr, thresh = roc_curve(test_target, lr_probs)
	# fpr_i, tpr_i, thresh_i = roc_curve(test_target, test_target)
	# fpr_n, tpr_n, thresh_n = roc_curve(test_target, [0.5]*len(test_target))
	plt.plot(fpr, tpr)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(name)
	plt.legend()
	plt.savefig(f'aucroc {name}.png')
	plt.close()
	plt.clf()

# PRECISION-RECALL Curve
def prcurve(model, df_test, test_target, name):
	lr_probs = model.predict_proba(df_test)
	lr_probs = lr_probs[:, 1]
	lr_precision, lr_recall, threshold = precision_recall_curve(test_target, lr_probs)
	prc = auc(lr_recall, lr_precision)
	plt.plot(lr_recall, lr_precision)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title(name)
	plt.legend()
	plt.savefig('prcurve.png')

# MODEL INTERPRETABLITY USING SHAP
def interpret_shap(model,fulldf,name,mtype):
	df = fulldf[:20]
	print(fulldf.shape, " -> ", df.shape)
	# compute the SHAP values for every prediction in the dataset
	if isinstance(model, DecisionTreeClassifier):
		explainer = shap.TreeExplainer(model)
	else:
		explainer = shap.KernelExplainer(model.predict, df, output_names=['CLD-not Ethanol', 'CLD-Ethanol'])
		print('Explainer built')
	shap_values = explainer.shap_values(df)
	# print(shap_values)
	print('Explained values of len:', len(shap_values))
	print(len(shap_values[0]))
	# Force Plot
	# shap.force_plot(explainer.expected_value, shap_values[0,:], df.iloc[0,:], show=True)
	print('Force plot')
	# plt.show()
	# Feature Imoortance using mean(SHAP)
	# shap.summary_plot(shap_values, df)
	# Feature Importance Bar Plot using mean(|SHAP|)
	# shap.summary_plot(shap_values,df, plot_type="bar")
	# plt.show()
	# sort the features indexes by their importance in the model
	# (sum of SHAP value magnitudes over the validation dataset)
	# top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))
	# make SHAP plots of the three most important features
	# for i in range(5):
	# 	shap.dependence_plot(top_inds[i], shap_values, df)
	# shap_interaction_values = explainer.shap_interaction_values(df)
	f = plt.figure(figsize=(10,10))
	# f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
	ax1 = f.add_subplot(121)
	shap.summary_plot(shap_values, df, plot_type="bar", show=False)
	print('Summary plot 1')
	ax2 = f.add_subplot(122)
	shap.summary_plot(shap_values, df, show=False)
	print('Summary plot 2')
	ax1.set_title('mean(|SHAP value|)\navg impact on pred')
	ax1.xaxis.label.set_size(22)
	ax1.yaxis.label.set_size(22)
	ax1.tick_params(axis='both', labelsize=22)
	ax2.set_title('SHAP value\nimpact on pred')
	ax2.xaxis.label.set_size(22)
	ax2.yaxis.label.set_size(22)
	ax2.tick_params(axis='both', labelsize=22)
	ax2.axes.yaxis.set_visible(False)
	f.suptitle(name,fontsize=25)
	f.set_size_inches(10,10)
	# plt.rcParams['figure.figsize'] = 5, 5
	plt.savefig(f'{mtype}_shap.png')
	plt.close()
	plt.clf()
	vals= np.abs(shap_values)
	print(vals.shape)
	vals= vals.mean(0)
	print(vals.shape)
	# vals= vals.mean(0)
	# print(vals.shape)
	feature_importance = pd.DataFrame(list(zip(df.columns,vals)),columns=['col_name','feature_importance_vals'])
	print(feature_importance)
	feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)

def bar_plot_from_dict(data_dict):
	dd = {'Features': [], 'Feature Weights':[]}
	sorted_data_dict = sorted(data_dict.items(), key=lambda x: abs(data_dict[x[0]]), reverse=True)[:15]
	dd['Features'] = [x[0] for x in sorted_data_dict]
	dd['Feature Weights'] = [x[1] for x in sorted_data_dict]
	b = px.bar(dd, x='Features', y='Feature Weights')
	b.write_image('logistic_weights.png')