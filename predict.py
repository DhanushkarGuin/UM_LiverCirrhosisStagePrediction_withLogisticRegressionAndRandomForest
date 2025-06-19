import pickle
import pandas as pd

pipeline = pickle.load(open('pipeline.pkl', 'rb'))

columns = ['N_Days', 'Status', 'Drug',
            'Age', 'Sex', 'Ascites',
            'Hepatomegaly', 'Spiders', 'Edema',
            'Bilirubin', 'Cholesterol',
            'Albumin', 'Copper', 'Alk_Phos', 
            'SGOT', 'Tryglicerides', 'Platelets', 
            'Prothrombin']

test_input = pd.DataFrame([[2000, 'C', 'Placebo', '18000', 'M', 'Y', 'N', 'N', 'Y', 0.5, 200, 4.04, 22, 1200, 105, 80, 150, 10]], columns = columns)

prediction = pipeline.predict(test_input)
print('Stage:', prediction)