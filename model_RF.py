import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('liver_cirrhosis.csv')
# print(dataset.head())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Applying Standard Scale to avoid big numbers influence the algorithm into predicting wrong 
dataset['N_Days'] = scaler.fit_transform(dataset[['N_Days']])
dataset['Age'] = scaler.fit_transform(dataset[['Age']])
dataset['Bilirubin'] = scaler.fit_transform(dataset[['Bilirubin']])
dataset['Cholesterol'] = scaler.fit_transform(dataset[['Cholesterol']])
dataset['Albumin'] = scaler.fit_transform(dataset[['Albumin']])
dataset['Copper'] = scaler.fit_transform(dataset[['Copper']])
dataset['Alk_Phos'] = scaler.fit_transform(dataset[['Alk_Phos']])
dataset['SGOT'] = scaler.fit_transform(dataset[['SGOT']])
dataset['Tryglicerides'] = scaler.fit_transform(dataset[['Tryglicerides']])
dataset['Platelets'] = scaler.fit_transform(dataset[['Platelets']])
dataset['Prothrombin'] = scaler.fit_transform(dataset[['Prothrombin']])

# Converting the categorical values into binary values
dataset['Drug'] = dataset['Drug'].map({'Placebo': 0, 'D-penicillamine':1})
dataset['Sex'] = dataset['Sex'].map({'F': 0, 'M':1})
dataset['Ascites'] = dataset['Ascites'].map({'N': 0, 'Y':1})
dataset['Hepatomegaly'] = dataset['Hepatomegaly'].map({'N': 0, 'Y':1})
dataset['Spiders'] = dataset['Spiders'].map({'N': 0, 'Y':1})

# print(dataset.head())

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), [1,8])],
                        remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Applying this to obtain the names of the column for readability
encoded_col_names = ct.named_transformers_['encoder'].get_feature_names_out()
non_encoded_col_names = [col for i, col in enumerate(dataset.columns[:-1]) if i not in [1,8]]
all_col_names = list(encoded_col_names) + non_encoded_col_names
X_df = pd.DataFrame(X, columns=all_col_names)
# print(X_df.head())

# print(X_df.columns)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 0)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)
# print(y_pred)

from sklearn.metrics import roc_auc_score
y_pred_proba = rf.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print("ROC AUC Score (OvR):", roc_auc)

from sklearn.metrics import precision_score,recall_score,confusion_matrix
print("Precision:", precision_score(y_test,y_pred, average='weighted'))
print('Recall:', recall_score(y_test,y_pred, average='weighted'))
print('Confusion Matrix:', confusion_matrix(y_test,y_pred))
