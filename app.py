import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer 
from sklearn.preprocessing import OneHotEncoder, scale, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle
from st_social_media_links import SocialMediaIcons




def model(data):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('ohe.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    cat_vars = []
    num_vars = []
    for i in data.columns:
        if(data[i].dtype == "object"):
            cat_vars.append(i)
        else:
            num_vars.append(i)

    cat_data = data[cat_vars]
    num_data = data[num_vars]
    
    encoded_cat = encoder.transform(cat_data)
    transformed_ohe = pd.DataFrame(
    data=encoded_cat,
    columns=encoder.get_feature_names_out(cat_vars),
    index=data.index,
    )

    test_data = pd.concat([num_data, transformed_ohe], axis=1)
    
    scaled_data = scaler.transform(test_data)
    
    predictions = model.predict(scaled_data)
    
    return predictions

def model_input(input_):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('ohe.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    with open(r'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    

    data = pd.DataFrame([input_])
    
    data.columns=['loan_amount', 'rate_of_interest', 'Interest_rate_spread',
       'Upfront_charges', 'term', 'property_value', 'income', 'Credit_Score',
       'LTV', 'dtir1', 'loan_limit', 'Gender', 'approv_in_adv', 'loan_type',
       'loan_purpose', 'Credit_Worthiness', 'open_credit',
       'business_or_commercial', 'Neg_ammortization', 'interest_only',
       'lump_sum_payment', 'occupancy_type', 'total_units', 'credit_type',
       'co-applicant_credit_type', 'age', 'submission_of_application',
       'Region']
    
    
    
    cat_vars = []
    num_vars = []
    for i in data.columns:
        if(data[i].dtype == "object"):
            cat_vars.append(i)
        else:
            num_vars.append(i)
            
    cat_data = data[cat_vars]
    num_data = data[num_vars]

    encoded_cat = encoder.transform(cat_data)
    transformed_ohe = pd.DataFrame(
    data=encoded_cat,
    columns=encoder.get_feature_names_out(cat_vars),
    index=data.index,
    )

    test_data = pd.concat([num_data, transformed_ohe], axis=1)
    
    scaled_data = scaler.transform(test_data)
    
    predictions = model.predict(scaled_data)
    
    proba = model.predict_proba(scaled_data)
    
    return predictions , proba
    

    


def main():
    
    
    #################
    #Streamlit setup:
    #page config:
    st.set_page_config(
        page_title="Default",
        page_icon="piggy-bank.png",
        layout="wide",
        initial_sidebar_state="expanded",

    )



    #################
    
    #main

    st.title('💳 Defaulting')
    st.write('This application aims to identify bank customers whom might have a probability to default their loan payement.')
    st.write('You can upload your customers database and check them all at once, or insert a single customer data and check them up.')
    st.write('Made by Mohamed Aziz Khezami') 
    social_media_links = [
    "https://www.linkedin.com/in/mohamed-aziz-khezami-160523252/",

    "https://www.github.com/MohamedAziz-Khezami",
    
    "https://www.instagram.com/khezamim.a/"
    ]

    social_media_icons = SocialMediaIcons(social_media_links)

    social_media_icons.render(justify_content='start')

    st.divider()

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file:

        df= pd.read_csv(uploaded_file)
        
        predictions = pd.DataFrame(model(df), columns=['Status'])
        
        
        show_df = pd.merge(df, predictions,left_index=True,right_index=True, how='inner')
        

        def highlight_survived(s):
            return ['background-color: #1c7501' ]*len(s) if s.Status ==0   else ['background-color: #870c03']*len(s)

        st.dataframe(show_df.style.apply(highlight_survived, axis=1), )
    else:
        st.write("No file uploaded")
    
    
    
    st.divider()


    
    
    



    with st.container():

        col1 , col2= st.columns([2,2])
   
        with col1:
            
            try:
                a = float(st.text_input("Loan Amount ($):", value=236500))
                if a <= 0:
                    st.error("Check input")
                    raise ValueError  # Raise exception to prevent further calculations
            except (ValueError, TypeError):
                a = None  # Set a to None if invalid input

            try:
                b = float(st.text_input("Rate of interest:", value=3.6666666666666665))
                if b <= 0:
                    st.error("Check input")
                    raise ValueError  # Raise exception to prevent further calculations
            except (ValueError, TypeError):
                b = None  # Set a to None if invalid input

            try:
                c = float(st.text_input("Interest rate spread", value=0.5148666666666667))
                if c <= 0:
                    st.error("Check input")
                    raise ValueError  # Raise exception to prevent further calculations
            except (ValueError, TypeError):
                c = None  # Set a to None if invalid input

            try:
                d = float(st.text_input("Upfront chargs", value=86.970))
                if d <= 0:
                    st.error("Check input")
                    raise ValueError  # Raise exception to prevent further calculations
            except (ValueError, TypeError):
                d = None  # Set a to None if invalid input
            try:
                e = float(st.text_input("Term", value=360))
                if e <= 0:
                    st.error("Check input")
                    raise ValueError  # Raise exception to prevent further calculations
            except (ValueError, TypeError):
                e = None  # Set a to None if invalid input

            try:
                f = float(st.text_input("Property Value", value=238000))
                if f <= 0:
                    st.error("Check input")
                    raise ValueError  # Raise exception to prevent further calculations
            except (ValueError, TypeError):
                f = None  # Set a to None if invalid input

            try:
                g = float(st.text_input("Income", value=8640))
                if g <= 0:
                    st.error("Check input")
                    raise ValueError  # Raise exception to prevent further calculations
            except (ValueError, TypeError):
                g = None  # Set a to None if invalid input
            try:
                h = float(st.text_input("Credit Score", value=848))
                if h <= 0:
                    st.error("Check input")
                    raise ValueError  # Raise exception to prevent further calculations
            except (ValueError, TypeError):
                h = None  # Set a to None if invalid input
            try:
                i = float(st.text_input("LTV", value=99.3697479))
                if i <= 0:
                    st.error("Check input")
                    raise ValueError  # Raise exception to prevent further calculations
            except (ValueError, TypeError):
                i = None  # Set a to None if invalid input
            try:
                j = float(st.text_input("dtir1", value=41))
                if j <= 0:
                    st.error("Check input")
                    raise ValueError  # Raise exception to prevent further calculations
            except (ValueError, TypeError):
                j = None  # Set a to None if invalid input

            ja = st.selectbox('loan limit', ('cf','ncf'))
            k = st.selectbox('Gender', ('Male','Female','Joint','Sex Not Available'))
            
            l = st.selectbox('Approved in Advance', ('nopre','pre'))
            
            m= st.selectbox('Loan type', ('type1','type2','type3'))
        
            
            butt = st.button('Check')

            


            
        with col2:
        
            n = st.selectbox('Loan purpose', ('p1','p2','p3','p4'))
            o = st.selectbox('Credit worthiness', ('l1','l2'))
                        
            p = st.selectbox('Open Credit', ('nopc','opc'))
            q = st.selectbox('Business or Commercial', ('nob/c','b/c'))
            r = st.selectbox('Neg Ammortization', ('not_neg','neg_amm'))
            s = st.selectbox('Interest Only', ('not_int','int_only'))
            t = st.selectbox('Lump sum payement', ('not_lpsm','lpsm'))
            u = st.selectbox('Occupancy type', ('pr','ir','sr'))
            v = st.selectbox('Total units', ('1U','2U','3U','4U'))
            w = st.selectbox('Credit Type', ('CIB','EXP','CRIF','EQUI'))
            x = st.selectbox('Co applicant credit type', ('CIB','EXP','CRIF','EQUI'))
            y = st.selectbox('Age', ('45-54','35-44','55-64','65-74','25-34','>74','<25'))
            z = st.selectbox('Submission of application', ('to_inst','not_inst'))
            za = st.selectbox('Region', ('North','south','central','North-East'))

    
        
        input_ = [a,b,c,d,e,f,g,h,i,j,ja,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,za]
        check = 0
        for i in input_:
            if i =='' or i==None:
                i=None
                check +=1
                break
        if check>0:
            st.write("Check input")

        else:
            if  butt:
                pre , prob = model_input(input_)
                if pre[0] == 1:
                    st.markdown(f'<h1 style="background-color:#870c03;font-size:24px;">{"🛑This client is likely to default! with a probability of:🛑"}</h1>', unsafe_allow_html=True)
                    st.write("with a probability of:",type(prob))
                else:
                    st.markdown(f'<h1 style="background-color:#1c7501;font-size:24px;">{"This client is not likely to default"}</h1>', unsafe_allow_html=True)
                    st.write("with a probability of:",prob)
                
    
              
    


if __name__ == "__main__":
    main()