import pandas as pd
import streamlit as st
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np 
import sklearn
import warnings 
warnings.filterwarnings('ignore')
import joblib
from sklearn.linear_model import LinearRegression


data = pd.read_csv('startUpData.csv')

# load the model
model = joblib.load(open('LinearReg.pkl', 'rb'))

#--------------- streamlit development starts -------------------
st.markdown("<h1 style = ' color: #053B50'>START UP BUSINESS PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h6 style = 'top-margin: 0rem; color: #BB2525'>BUILT BY Gomycode Yellow Orange Beast</h1>", unsafe_allow_html=True)

# st.title('START UP BUISNESS PREDICTOR')
# st.write('built by Gomycode yellow orange Beast')

st.write('PLEASE ENTER YOUR USERNAME')
USERNAME = st.text_input(' PLEASE ENTER YOUR USERNAME:')
if st.button('SUBMIT NAME'):
    st.success(f"WELCOME {USERNAME}.please enjoy usage")
    
st.markdown("<br> <br>", unsafe_allow_html= True)
st.markdown("<h2 style = 'top-margin: 0rem;text-align: center; color: #A2C579'>Project Introduction</h1>", unsafe_allow_html=True)

st.write("In the dynamic landscape of entrepreneurship, understanding and predicting the profitability of startup firms is crucial for investors, founders, and stakeholders alike. The project at hand employs a Linear Regression model to analyze and forecast the profitability of startup companies. Leveraging historical data on various startups, this study aims to unravel the key factors that influence a startup's success and provide valuable insights for making informed investment decisions. By delving into the intricate relationship between variables such as R&D expenditure, marketing spend, location, and industry, this project endeavors to shed light on the critical drivers of startup profitability, contributing to a more data-driven approach in the world of business.")

heat_map = plt.figure(figsize = (14,7)) #---------------------------------------------------create a heat map plot
correlation_data = data[['R&D Spend',	'Administration',	'Marketing Spend', 'Profit']] #----sele data fr corelation

sns.heatmap(correlation_data.corr(), annot = True, cmap = 'BuPu')
st.write(heat_map)
data.drop('Unnamed: 0', axis = 1, inplace = True)
st.write(data.sample(10))

st.sidebar.image('pngwing.com (1).png', width= 300 , caption = f"WELCOME {USERNAME}",use_column_width= True)

st.markdown("<br>", unsafe_allow_html= True)
  

# picture = st.camera_input('Take a selfie')
# if picture:
#     st.sidebar.image(picture, use_column_width= True, caption = f"welcome  {USERNAME}")

st.sidebar.write('please decide your variable input type')
input_style = st.sidebar.selectbox('pick your preferred input', ['Slider input','Number input'])
if input_style == 'Slider input':
    research = st.sidebar.slider('R & D Spending', data['R&D Spend'].min(), data['R&D Spend'].max() )
    admin = st.sidebar.slider('Administration', data['Administration'].min(), data['Administration'].max())
    marketting = st.sidebar.slider('Marketting Expense', data['Marketing Spend'].min(), data['Marketing Spend'].max()) 
    states = data['State'].unique()
    state =  st.sidebar.select_slider('Select Your State Model', states)

else:
    research = st.sidebar.number_input('R & D Spending', data['R&D Spend'].min(), data['R&D Spend'].max() )
    admin = st.sidebar.number_input('Administration', data['Administration'].min(), data['Administration'].max())
    marketting = st.sidebar.number_input('Marketting Expense', data['Marketing Spend'].min(), data['Marketing Spend'].max()) 
    states = data['State'].unique()
    state =  st.sidebar.selectbox('Select Your State Model', states)

st.subheader("Your Inputted Data")
input_var = pd.DataFrame([{'R&D Spend': research, 'Administration': admin, 'Marketing Spend': marketting}])
st.write(input_var)

st.markdown("<br>", unsafe_allow_html= True)
tab1, tab2 = st.tabs(["Prediction Pane", "Intepretation Pane"])

with tab1:
    if st.button('PREDICT'):

        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_var)
        st.write("Predicted Profit is :", prediction)
    else:
        st.write('Pls press the predict button for prediction')

with tab2:
    st.subheader('Model Interpretation')
    st.write(f"Profit = {model.intercept_.round(2)} + {model.coef_[0].round(2)} R&D Spend + {model.coef_[1].round(2)} Administration + {model.coef_[2].round(2)} Marketing Spend")

    st.markdown("<br>", unsafe_allow_html= True)

    st.markdown(f"- The expected Profit for a startup is {model.intercept_}")

    st.markdown(f"- For every additional 1 dollar spent on R&D Spend, the expected profit is expected to increase by ${model.coef_[0].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Administration Expense, the expected profit is expected to decrease by ${model.coef_[1].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Marketting Expense, the expected profit is expected to increase by ${model.coef_[2].round(2)}  ")