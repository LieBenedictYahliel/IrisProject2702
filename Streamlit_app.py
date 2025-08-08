import streamlit as st
import pandas as pd
import joblib
# How to run streamlit app localy
# open terminal 
# cd "D:\Purwadhika\Capstone 2\IPYNB\Machine Learning\Day 18"
# streamlit run streamlit_app.py
st.title("Iris Classifier")
st.write("This is a simple Iris Classifier app")

# https://docs.streamlit.io/get-started/installation

# === Inference Function
model=joblib.load("model.joblib")
def get_prediction(data: pd.DataFrame):
    pred=model.predict(data)
    pred_proba=model.predict_proba(data)
    return pred, pred_proba

# User Input
left, right=st.columns(2, gap="medium", border=True)

# --- Sepal Input
left.subheader("sepal")
sepal_length=left.slider("sepal length", min_value=4.3, max_value=10.0, value=5.4, step=0.1)
sepal_width=left.slider("sepal width", min_value=4.3, max_value=10.0, value=5.4, step=0.1)
# -- petal input
right.subheader("petal")
petal_length=right.slider("petal length", min_value=4.3, max_value=10.0, value=5.4, step=0.1)
petal_width=right.slider("petal width", min_value=4.3, max_value=10.0, value=5.4, step=0.1)
# show input value
import streamlit as st
import pandas as pd

# Show input value
data=pd.DataFrame({"sepal length (cm)": [sepal_length], 
                    "sepal width (cm)": [sepal_width], 
                    "petal length (cm)": [petal_length], 
                    "petal width (cm)": [petal_width]})

st.dataframe(data, use_container_width=True)
# Prediction button
button=st.button("predict", use_container_width=True)
if button:
    st.write("Prediksi Berhasil")
    pred, pred_proba=get_prediction(data)
    label_map={0:"Iris-Setosa",
            1:"Versicolor",
            2:"Virginica"}
    label_pred=label_map[pred[0]]
    label_proba=pred_proba[0][pred[0]]
    output=f"Iris Anda diklasifisikan sebagai {label_proba:0%} {label_pred}"
    st.write(output)


    
