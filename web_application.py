#!/usr/bin/env python

import streamlit as st
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re

st.title("Phishing Website Detection Tool")
st.write("The tool harnesses the power of supervised machine learning to analyze and interpret the HTML elements of websites, distinguishing between legitimate and phishing sites with remarkable accuracy. It is developed for educational purposes only. Please select Random Forest for best results.")

choice = st.selectbox("Please select the machine learning algorithm",
                      [
                          'Support Vector Machine',
                          'Decision Tree',
                          'Gaussian Naïve Bayes',
                          'Random Forest',
                          'k-Nearest Neighbors'
                      ]
                      )

if choice == 'Support Vector Machine':
    model = ml.svm_model
elif choice == 'Decision Tree':
    model = ml.dt_model
elif choice == 'Gaussian Naïve Bayes':
    model = ml.gnb_model
elif choice == 'Random Forest':
    model = ml.rf_model
else:
    model = ml.knn_model

url = st.text_input("Please enter the URL")

if st.button('Check'):
    try:
        response = re.get(url, verify=False, timeout=4)
        if response.status_code != 200:
            print("Connection was not successful for the URL ", url)
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            vector = [fe.create_vector(soup)]
            result = model.predict(vector)
            if result[0] == 0:
                st.success("The URL seems to be legitimate!")
                st.balloons()
            else:
                st.warning("Warning! The URL is a potential phishing link!")
                st.snow()

    except re.exceptions.RequestException as e:
        print(" --> ", e)
