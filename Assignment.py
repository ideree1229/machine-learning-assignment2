
#Iderbat 08470101 
import pandas as pd
import numpy as np 
import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics

df = pd.read_csv('winequality-white.csv',sep=";")
X = df.drop(columns=['quality'])
y = df['quality']


#LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#Evaluation

MAE = metrics.mean_absolute_error(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = metrics.mean_squared_error(y_test, y_pred)


#DecisionTree
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3, random_state=0)

dect = DecisionTreeClassifier()
dect.fit(X_train2, y_train2)

y_pred2 = dect.predict(X_test2)
#Evaluation
score = accuracy_score(y_test2, y_pred2)


#RandomForestClassifier
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.3, random_state=0)

clf = RandomForestClassifier()
clf.fit(X_train3, y_train3)
y_pred3 = clf.predict(X_test3)
clf.predict_proba(X_test3)
#Evaluation
score2 = accuracy_score(y_test3, y_pred3)

#streamlit
st.write("""
# White Wine Quality Prediction App
""")
if st.button("View Data"):
    st.subheader('Wine Data')
    st.write(df)

model = st.sidebar.selectbox("Models", ["Linear Regression", "Decision Tree", "Random Forest"])
if model == "Decision Tree":
    st.subheader("Prediction")  
    st.write(y_pred2)
    st.subheader("Prediction Probability")
    st.write(dect.predict_proba(X_test2) )
    st.subheader("Evaluation")
    st.write("Accuracy score:",score)
if model == "Linear Regression":
    st.subheader("Prediction")  
    st.write(y_pred)
    st.subheader("Evaluation")
    st.write("Mean Absolute Error:", MAE)
    st.write("Mean Squared Error:", MSE)
    st.write("Root Mean Squared Error:",RMSE)
if model == "Random Forest":
    st.subheader("Prediction")  
    st.write(y_pred3)
    st.subheader("Prediction Probability")
    st.write(clf.predict_proba(X_test3) )
    st.subheader("Evaluation")
    st.write("Accuracy score:",score2)
