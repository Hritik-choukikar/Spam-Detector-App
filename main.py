import streamlit as st
import pickle
@st.cache
def load_model():
    with open('./model/naivebayes_clf','rb') as f:
        model=pickle.load(f)
    return model



st.header("Spam mail Detector app")
col1,col2=st.columns([2,2])

col1.subheader("Enter the text")
spam=col1.text_area(label='')

model=load_model()

k=st.button(label='Predict')
if k:
    predict=model.predict([spam])
    print(predict[0])
    if predict[0]==1:
        st.write('Your Email is : Spam')
    else:
        st.write('Your Email is : Ham')



st.text("")
st.text("")
st.write('Note: This Machine Learning model is trained on less amount of data, so the prediction of the model is not accurate')

