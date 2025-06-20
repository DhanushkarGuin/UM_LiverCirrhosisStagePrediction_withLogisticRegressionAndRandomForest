import pandas as pd

dataset = pd.read_csv('liver_cirrhosis.csv')
# print(dataset.head())

# print(dataset.columns.tolist())

# print(dataset.isnull().sum()) # No empty values

categorical_columns = ['Status','Drug','Sex','Ascites', 'Hepatomegaly', 'Spiders', 'Edema']

numerical_columns = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']

X = dataset.drop(columns = ['Stage'])
y = dataset['Stage']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

preprocces = ColumnTransformer([
    ('scaler', StandardScaler(), numerical_columns),
    ('ohe', OneHotEncoder(sparse_output=False,handle_unknown='ignore'),categorical_columns)
],remainder='passthrough')

pipeline = Pipeline([
    ('preprocess', preprocces),
    ('rf',RandomForestClassifier())
])

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)

from sklearn.metrics import precision_score,recall_score,confusion_matrix
print('Precision', precision_score(y_test,y_pred, average='weighted'))
print('Recall', recall_score(y_test,y_pred,average='weighted'))
print('Confusion Matrix \n', confusion_matrix(y_test,y_pred))

import pickle
pickle.dump(pipeline, open('pipeline.pkl', 'wb'))