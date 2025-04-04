import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pickle as pkl

#load the trained model 
model = tf.keras.models.load_model('model.h5')

# load the encoder and scaler
with open('Onehot_encoder_geo.pkl','rb') as file:
    Onehot_encoder_geo = pkl.load(file)
    
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pkl.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pkl.load(file)


#Streamlit app

st.title('Customer Churn Prediction')

Credit_Score = st.number_input('CreditScore')
Geography= st.selectbox('Geography',Onehot_encoder_geo.categories_[0])
Gender= st.selectbox('Gender',label_encoder_gender.classes_)
Age= st.slider('Age',18,92)
Tenure = st.slider('Tenure',0,10)
Balance = st.number_input('Balance')
Num_Of_Products = st.slider('Number of products',1, 4)
Has_Cr_Card = st.selectbox('Has Credit Card', [0,1])
Is_Active_Member= st.selectbox('Is Active Member',[0,1])
Estimated_Salary = st.number_input('Estimated Salary')

input_data = pd.DataFrame({
    'CreditScore':[Credit_Score],
    'Gender': [label_encoder_gender.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance' :[Balance],
    'NumOfProducts' : [Num_Of_Products],
    'HasCrCard' : [Has_Cr_Card],
    'IsActiveMember':[Is_Active_Member],
    'EstimatedSalary': [Estimated_Salary]
})

geo_encoded = Onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = Onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop = True), geo_encoded_df], axis = 1)

input_data_scaled = scaler.transform(input_data)

#predict churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn Probability: {prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.write('The customer is likely to churn')

else:
    st.write('The customer is not likely to churn')