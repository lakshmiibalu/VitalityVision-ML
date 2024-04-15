
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import joblib
import warnings
warnings.filterwarnings('ignore')

diabetes_dataset = pd.read_csv('diabetes.csv')

# Replace categorical features with numerical values
diabetes_dataset['Medicine_Type'].replace(['None', 'Insulin','Metformin'],[0,1,2], inplace=True)
diabetes_dataset['Exercise_Type'].replace(['None', 'Running', 'Walking', 'Yoga', 'Swimming', 'Cycling'],[0,1,2,3,4,5], inplace=True)
diabetes_dataset['Unusual_Event'].replace(['None', 'High stress', 'Stressful day', 'Feeling unwell', 'Sick'],[0,1,2,3,4], inplace=True)
diabetes_dataset['Diabetic_Status'].replace(['Normal', 'Moderate', 'Needs Medical Assistance'],[0,1,2], inplace=True)

# Drop unnecessary columns
diabetes_dataset.drop('Date', axis=1, inplace=True)
diabetes_dataset.drop('Time', axis=1, inplace=True)
#Removed Other because the values were very less

#diabetes_dataset = diabetes_dataset.reset_index(drop=True)

X_train, X_test, Y_train, Y_test = train_test_split(diabetes_dataset.drop("Diabetic_Status", axis=1), diabetes_dataset["Diabetic_Status"], test_size=0.3)

print(X_train.dtypes)

classifier = XGBClassifier()
classifier.fit(X_train, Y_train)

joblib.dump(classifier, 'diabetic.joblib')






