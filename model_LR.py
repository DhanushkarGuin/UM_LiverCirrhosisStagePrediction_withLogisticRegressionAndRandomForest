import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('liver_cirrhosis.csv')

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

X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]

# Applying OneHotEncoder to convert the categorical values into columns with binary values in it
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
#print(X_df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='newton-cg', max_iter=100000)
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)
#print('Predictions:', Y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy %:", (accuracy_score(Y_test, Y_pred))*100)
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))