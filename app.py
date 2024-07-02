import streamlit as st 
import pandas as pd
import numpy as np


def model():
    ...





def main():
    
    
    #################
    #Streamlit setup:
    #page config:
    st.set_page_config(
        page_title="Default",
        page_icon="/Users/mak/Desktop/Code_With_Me/Defaulting/piggy-bank.png",
        layout="wide",
        initial_sidebar_state="expanded",

    )


    #################
    
    #main

    df= pd.read_csv("/Users/mak/Desktop/Code_With_Me/Defaulting/Loan_Default.csv")

    def highlight_survived(s):
        return ['background-color: #0b9101' ]*len(s) if s.Status ==0   else ['background-color: #e8665f']*len(s)

    st.dataframe(df.head(20).style.apply(highlight_survived, axis=1), )
    
    


    
    
    



    with st.container():

        col1 , col2, col3 = st.columns([2,1,2])
    
        with col1:
            title = st.text_input("Movie title", "Life of Brian")
            ho = st.text_input(key=2, label="hi")
            butt= st.button("Check")
        
        with col2:
            st.write('')
            
        with col3:
                if  butt:
                    st.write("#")

                    st.header('heelo')
    
    


if __name__ == "__main__":
    main()