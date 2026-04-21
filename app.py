import pickle
import pandas as pd
import numpy as np
import streamlit as st

def predict_species(sep_len,sep_width,pet_len,pet_wild,scaler_path,model_path):
    try:
        # load the scaler
        with open(scaler_path,'rb') as file1:
            scaler = pickle.load(file1)

        # load the model
        with open(model_path,'rb')as file2:
            model = pickle.load(file2)

        # prepare input data
        dct = {
            'SepalLengthCm':[sep_len],
            'SepalWidthCm':[sep_width],
            'PetalLengthCm':[pet_len],
            'PetalWidthCm':[pet_wild]
        }
        x_new = pd.DataFrame(dct)

        # Transform input data
        xnew_pre = scaler.transform(x_new)

        # make predictions
        pred =  model.predict(xnew_pre)
        probs = model.predict_proba(xnew_pre)
        max_prob = np.max(probs)

        return pred,max_prob
    except Exception as e:
        # log and display errors
        st.error(f"Error during predication: {str(e)}")
        return None, None
    
# StreamLit UI
st.title("Iris Species Predictor")

# input fields for the features
sep_len = st.number_input("SepalLengthCm",min_value=0.0,step=0.1,value=5.1)
sep_width = st.number_input("SepalWidthCm",min_value=0.0,step=0.1,value=3.5)
pet_len = st.number_input("PetalLengthCm",min_value=0.0,step=0.1,value=1.4)
pet_wid = st.number_input("PetalWidthCm",min_value=0.0,step=0.1,value=3.5)

# prediction button
if st.button("Predict"):
    # file paths
    scaler_path = 'notebook/scaler.pkl'
    model_path = 'notebook/model.pkl'

    #call the predictions functions
    pred, max_prob = predict_species(sep_len,sep_width,pet_len,pet_wid,scaler_path,model_path)

    # Display results
    if pred is not None and max_prob is not None:
        st.subheader(f'Predicated Species:{pred[0]}')
        st.subheader(f'Prediction Probabiltiy:{max_prob:.4f}')
        st.progress(max_prob)
    else:
        st.error("Predication Failed. Check the input values are model files. ")