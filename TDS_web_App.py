import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xg
from xgboost.sklearn import XGBClassifier
im=Image.open('Data-Science_FB.jpeg')
st.image(im,use_column_width=False)
st.title(" Data Analysis Web App")
#def  main():
activities=['EDA','visualisation','model']
option=st.sidebar.selectbox('Selection option :',activities)
    
if option=='EDA':
    st.subheader("Exploratory Data Analysis")
        
    data=st.file_uploader("upload dataset:",type=['csv','xlsx','txt','json'])
    st.success("Data Successfully loaded")
    if data is not None:
        df=pd.read_csv(data)
        st.dataframe(df.head(10))
    if st.checkbox("Display Shape of data"):
        st.write(df.shape)
    if st.checkbox("Columns name"):
        st.write(df.columns)
    if st.checkbox("Display data types"):
        st.write(df.dtypes)
    if st.checkbox("Check Null values"):
        st.write(df.isnull().sum())
    if st.checkbox("Select independent variables"):
        select_col=st.multiselect('Select preferred columns:',df.columns)
        df1=df[select_col]
        st.dataframe(df1)
    if st.checkbox("Give summary of data"):
        st.write(df1.describe())
    if st.checkbox("Select dependent variable"):
        select2_col=st.selectbox('Select preferred columns:',df.columns)
        df2=df[select2_col]
        st.dataframe(df2)
    if st.checkbox("Correlation Between independent variables"):
        st.write(df1.corr())
        
if option=='visualisation':
    st.subheader("Data Visualisation")
    data=st.file_uploader("upload dataset:",type=['csv','xlsx','txt','json'])
    st.success("data Successfully loaded")
    if data is not None:
        df=pd.read_csv(data)
        st.dataframe(df.head(5))
    if st.checkbox("Select columns"):
        select_col1=st.multiselect("Select preffered columns:",df.columns)
        df3=df[select_col1]
        st.dataframe(df3)
    if st.checkbox("Display Heatmap"):
        st.write(sns.heatmap(df.corr(),annot=True))
        st.pyplot()
    if st.checkbox("Display boxplot"):
        st.write(df.boxplot())
        st.pyplot()
    if st.checkbox("Display pairplot"):
        st.write(sns.pairplot(df))
        st.pyplot()
        
if option=='model':
    st.subheader("Model Building") 
    data=st.file_uploader("upload dataset:",type=['csv','xlsx','txt','json'])
    st.success("Data loaded")
    if data is not None:
        df=pd.read_csv(data)
        st.dataframe(df.head(5))
    algorithm=['Logistic Regression','KNN','svm','decisiontree','Randomforest','Adapboosting']
    model=st.sidebar.selectbox("selection option:",algorithm)
    
 
    x=df.iloc[:, 0:-1]
    y=df.iloc[:,-1]
  
     #split data   
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)  
 
        
    if model=='Logistic Regression':
        st.subheader("Logistic Regression :")
        from sklearn.linear_model import LogisticRegression
        clf1=LogisticRegression()
        clf1.fit(x_train,y_train)
        y_pred=clf1.predict(x_test)
        accu1=accuracy_score(y_pred,y_test)
        st.write("Accuracy of the model:",accu1)
    if model=='KNN':
        st.subheader("KNN :")
        k=st.sidebar.slider('k',1,15)
        from sklearn.neighbors import KNeighborsClassifier
        clf2=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
        clf2.fit(x_train,y_train)
        y_pred2=clf2.predict(x_test)
        accu2=accuracy_score(y_pred2,y_test)
        st.write("Accuracy of the model:",accu2)
    if model=='svm':
        st.subheader("Support vector Machine :")
        c=st.sidebar.slider('c',0.1,1.5)
        clf3=SVC(kernel='rbf')
        clf3.fit(x_train,y_train)
        y_pred3=clf3.predict(x_test)
        accu3=accuracy_score(y_pred3,y_test)
        st.write("Accuracy of the model:",accu3)
    if model=='decisiontree':
        st.subheader("Decision Tree :")
        from sklearn.tree import DecisionTreeClassifier
        clf4=DecisionTreeClassifier()
        clf4.fit(x_train,y_train)
        y_pred4=clf4.predict(x_test)
        accu4=accuracy_score(y_pred4,y_test)
        st.write("Accuracy of the model:",accu4)
    if model=='Randomforest':
        st.subheader("Random forest :")
        from sklearn.ensemble import RandomForestClassifier
        clf5=RandomForestClassifier()
        clf5.fit(x_train,y_train)
        y_pred5=clf5.predict(x_test)
        accu5=accuracy_score(y_pred5,y_test)
        st.write("Accuracy of the model:",accu5)
    if model=='Adapboosting':
        st.subheader("Adapboosting :")
        E=st.sidebar.slider('E',100,10000)
        L=st.sidebar.slider('L',1,15)
        from sklearn.ensemble import AdaBoostClassifier
        clf6=AdaBoostClassifier(n_estimators=E,learning_rate=L)
        clf6.fit(x_train,y_train)
        y_pred6=clf6.predict(x_test)
        accu6=accuracy_score(y_pred6,y_test)
        st.write("Accuracy of the model:",accu7)

