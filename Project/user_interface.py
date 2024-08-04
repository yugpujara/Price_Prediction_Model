import pandas as pd
import streamlit as st
import joblib
import xgboost as xgb

def main():
    html_temp = """
        <div style = "background-color:orange"; padding:16px>
        <h2 style = "color:blue; text-align:center;">Total Retail Price Prediction</h2>
        </div>  
    """
    model = xgb.XGBRegressor()
    model.load_model('xgb_model.json')

    st.markdown(html_temp,unsafe_allow_html=True)
    st.write('')
    st.write('')
    
    st.markdown("##### Are you willing to know the selling price of the product?\n###### So let's try evaluating the price : ")
    s1 = st.selectbox("What is your Customer Status?", ('silver','gold','platinum'))
    if s1 == 'silver':
        p1 = 0
    elif s1 == 'gold':
        p1 = 1
    elif s1 == 'platinum':
        p1 = 2
    
    p2 = st.number_input('What is the quantity of product you want to order : ', 1, step=1)

    p3 = st.number_input('What is the Cost per unit of the product : ', 0.0, step=0.0, format = "%.2f")

    user_data = pd.DataFrame({
        'Customer Status' : p1,
        'Quantity Ordered' : p2,
        'Cost Price Per Unit' : p3
    },index=[0])

    if st.button('Predict'):
        pred = model.predict(user_data)
        st.balloons()
        st.success('The Total Retail Price predicted is {:.2f}'.format(pred[0]))    

if __name__ == '__main__':
    main()