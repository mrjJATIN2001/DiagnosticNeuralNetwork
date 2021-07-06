import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_option('deprecation.showfileUploaderEncoding', False)

# Load the pickled model
model = load(open('diagnostic.h5', 'rb')) 
dataset= pd.read_csv('PCA_NN_dataset1.csv')

x = dataset.iloc[:, :-1].values

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
imputer = imputer.fit(x[:, 2:7 ]) 
x[:, 2:7 ]= imputer.transform(x[:, 2:7 ])  


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)


def predict_note_authentication(Gender, Glucose, BP, SkinThickness, Insulin, BMI, PedigreeFunction, Age):
  output= model.predict([[Gender, Glucose, BP, SkinThickness, Insulin, BMI, PedigreeFunction, Age]])
  print("Outcome", output)
  if output==[1]:
    prediction="Patient should have diagnostic"
  else:
    prediction="Patient should not have diagnostic"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Diagnostic Neural Network")

    Gender = st.number_input('Insert Gender Male:1 Female:0')
    Glucose = st.number_input('Insert Glucose')
    BP = st.number_input('Insert BP')
    SkinThickness = st.number_input('Insert SkinThickness')
    Insulin =  st.number_input('Insert Insulin')
    BMI =st.number_input('Insert BMI')
    PedigreeFunction = st.number_input('Insert PedigreeFunction')
    Age = st.number_input('Insert Age')

    resul=""
    if st.button("Prediction"):
      result=predict_note_authentication(Gender, Glucose, BP, SkinThickness, Insulin, BMI, PedigreeFunction, Age)
      st.success('Model has predicted {}'.format(result))  
    if st.button("About"):
      st.header("Developed by Jatin Tak")
      st.subheader("Student, Department of Computer Engineering")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">           
    <div class="col-md-12">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Model Prediction</p></center> 
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()
