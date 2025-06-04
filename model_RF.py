import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
# print("ROC AUC Score (OvR):", roc_auc)

from sklearn.metrics import precision_score,recall_score,confusion_matrix
# print("Precision:", precision_score(y_test,y_pred, average='weighted'))
# print('Recall:', recall_score(y_test,y_pred, average='weighted'))
# print('Confusion Matrix:', confusion_matrix(y_test,y_pred))

print("Enter data for prediction!")
n_days = int(input('Enter number of days:'))
status = (input('Enter status of patient(C (censored), CL (censored due to liver tx), or D (death)):'))
drug = (input('Enter the type of Drug:'))
age = int(input('Enter age in days:'))
sex = input('Enter sex(F for Female or M for Male):')
ascites = input('Enter presence of ascites(N for No or Y for Yes):')
hepatomegaly = input('Enter presence of hepatomegaly(N for No or Y for Yes):')
spiders = input('Enter presence of spiders(N for No or Y for Yes):')
edema = input(
    'Enter presence of edema N (no edema and no diuretic therapy for edema), S (edemapresent without diuretics, or edema resolved by diuretics), or Y (edema despite diuretictherapy):'
    )
bilirubin = float(input('Serum Bilirubin in mg/dl:'))
cholestrol = float(input('Serum Cholestrol in mg/dl:'))
albumin = float(input('Albumin in gm/dl:'))
copper = float(input('Urine Copper in ug/day:'))
alk_phos = float(input('Alkaline phosphatase in U/l:'))
sgot = float(input('SGOT in U/ml:'))
tryglicerides = float(input('Tryglicerides in mg/dl:'))
platelets = float(input('Platelets per cubic [ml/1000]:'))
prothrombin = float(input('Prothrombin time in seconds [s]'))

drug = 1 if drug == 'D-penicillamine' else 0
sex = 1 if sex == 'M' else 0
ascites = 1 if ascites == 'Y' else 0
hepatomegaly = 1 if hepatomegaly == 'Y' else 0
spiders = 1 if spiders == 'Y' else 0

status_encoded = [0,0,0]
if status == 'C':
    status_encoded[0] = 1
elif status == 'CL':
    status_encoded[1] = 1
else:
    status_encoded[2] = 1

edema_encoded = [0,0,0]
if edema == 'N':
    edema_encoded[0] = 1
elif edema == 'S':
    edema_encoded[1] = 1
else:
    edema_encoded[2] = 1

user_input = status_encoded + edema_encoded + [
    n_days, drug, age, sex, ascites, hepatomegaly, spiders,
    bilirubin, cholestrol, albumin, copper,
    alk_phos, sgot, tryglicerides, platelets, prothrombin
]

user_input = np.array(user_input).reshape(1, -1)

prediction = rf.predict(user_input)
print(f"Stage of Liver Cirrhosis: {prediction[0]}")




