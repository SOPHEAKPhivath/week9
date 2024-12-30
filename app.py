# 1. import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 2. add button in streamlit
cylinders = st.number_input(label='cylinders', values = 8.0)
displacement = st.number_input(label='displacement', values = 307.0)
horsepower = st.number_input(label='horsepower', values = 130.0)
weight = st.number_input(label='weight', values = 3504.0)
acceleration = st.number_input(label='acceleration', values = 12.0)
model_year = st.number_input(label='model year', values = 70.0)

origin = st.selectbox(
    label = 'Origin',
    option = ['USA', 'JAPAN', 'EUROPE'],
    placeholder = 'Select Origin'
)

# 3. Convert feature 
X_num = np.array(
    object=[
        [
            cylinders,
            displacement,
            horsepower,
            weight,
            acceleration,
            model_year,
            origin
        ]
    ],
    dtype=np.float32
)

# 4. import our pre-trained model
with open(file='scaler.pkl', mode='rb') as scaler:
    scaler = pickle.load(file=scaler)

with open(file='encode.pkl', mode='rb') as encode:
    encoder = pickle.load(file=encode)

with open(file='model_lr.pkl', mode='rb') as lr:
    lr = pickle.load(file=lr)

# 5. Pre-proccessing
X1 = scaler.transform(X_num)
X_cat = np.array(object=[origin], dtype=np.float32)
X_raw = np.concat([X1, X_cat.reshape(-1,1)], axis=1)

# 6. Prediction
y = lr.predict(X)
y_raw = 1 / y

# 7. Convert to dataframe
data = np.concat([X_raw, y_raw.reshape(-1,1)], axis=1)
df = pd.DataFrame(data=data,
                    columns=[
                        'cylinders',
                        'displacement',
                        'horsepower',
                        'weight',
                        'acceleration',
                        'model_year',
                        'origin',
                        'mpg'
                    ])
st.write(df)