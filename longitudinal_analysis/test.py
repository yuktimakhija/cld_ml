import pandas as pd
from sklearn import preprocessing,model_selection
import ml_models as models

final_df = pd.read_csv('patients_bleeder.csv').set_index('IPD')
outcome = final_df['y_bleeder']
x = final_df.drop(columns=['y_bleeder'])
# enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
scaler = preprocessing.StandardScaler().fit(x)
X_scaled = scaler.transform(x)

# Logistic Regression
print('LR')
models.logistic_regression(X_scaled,outcome,x,imbalance=1)
print('SVM Linear')
models.svm_cv(X_scaled,outcome,x, kernelt='linear',imbalance=1)
print('SVM RBF')
models.svm_cv(X_scaled,outcome, x, kernelt='rbf',imbalance=1)
print('UMAP')
# PLease install threadpoolctl
models.cluster_umap(X_scaled,outcome,num_classes=2,classification='bleeder')
print('DT')
models.decision_tree(X_scaled,outcome,x,imbalance=1)
print('XGB')
# For using XGB please install graphviz
models.xgboost(X_scaled,outcome, x,imbalance=1)