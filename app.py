import streamlit as st 
import pandas as pd
import numpy as np


df= pd.read_csv("/Users/mak/Desktop/Code_With_Me/Defaulting/Loan_Default.csv")

def highlight_survived(s):
    return ['background-color: green']*len(s) if s.Status ==0   else ['background-color: red']*len(s)

st.table(df.head(20).style.apply(highlight_survived, axis=1))