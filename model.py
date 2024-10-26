import pandas as pd
import numpy as np
df=pd.read_csv('data_csv.csv')
print(df.head())
df['Qchat_10_Score'].fillna(df['Qchat_10_Score'].mean(), inplace=True)
num=list(df.select_dtypes(include=['number']))
cat=list(df.select_dtypes(include=['object']))
for i in num:
  df[i].fillna(df[i].mean(), inplace=True)

des=df[cat].describe(include='all')
for i in cat:
  df[i].fillna(df[i].mode()[0], inplace=True)

cols_to_drop=["CASE_NO_PATIENT'S",'Qchat_10_Score','Ethnicity','Who_completed_the_test']
df.drop(cols_to_drop, axis=1, inplace=True)

print(df.head())
cat=list(df.select_dtypes(include=['object']))
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in cat:
  df[i]=l.fit_transform(df[i])

X=df.drop('ASD_traits', axis=1)
y=df['ASD_traits']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)
y_preds=model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
score1=accuracy_score(y_test, y_preds)
from sklearn.tree import DecisionTreeClassifier
model2=DecisionTreeClassifier()
model2.fit(X_train,y_train)
y_preds2=model2.predict(X_test)
score2=accuracy_score(y_test,y_preds2)

from sklearn.ensemble import RandomForestClassifier
model3=RandomForestClassifier()
model3.fit(X_train,y_train)
y_preds3=model3.predict(X_test)
score3=accuracy_score(y_test, y_preds3)

from sklearn.linear_model import LogisticRegression
model4=LogisticRegression()
model4.fit(X_train,y_train)
y_preds4=model4.predict(X_test)
score4=accuracy_score(y_test, y_preds4)

from sklearn.neighbors import KNeighborsClassifier
model5=KNeighborsClassifier()
model5.fit(X_train,y_train)
y_preds5=model5.predict(X_test)
score5=accuracy_score(y_test,y_preds5)

from sklearn.ensemble import VotingClassifier
model6=VotingClassifier(estimators=[('svm',model),('dt',model2),('rf',model3),('lr',model4),('knn',model5)],voting='hard')
model6.fit(X_train,y_train)
y_preds6=model6.predict(X_test)
score6=accuracy_score(y_test,y_preds6)

data = {'Model': ['SVM', 'Decision Tree', 'Random Forest','Logistic Regression','KNN','VotingClassifier'],
        'Accuracy Score': [score1, score2, score3, score4, score5,score6]}

df_compare = pd.DataFrame(data)
print(df_compare)

print(df.columns)

import pickle

pickle.dump(model3,open('random.pkl','wb'))
