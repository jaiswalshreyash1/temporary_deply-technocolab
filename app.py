import streamlit as st
import numpy as np
import pickle 
import joblib as jb 

LRmodel = jb.load('LR model.pkl')

def predict_loan_status(Age,Interest,LoanDuration,NoOfPreviousLoansBeforeLoan):
 input =  np.array([[Age,Interest,LoanDuration,NoOfPreviousLoansBeforeLoan]]).astype(np.float64)  
 prediction =LRmodel.predict_proba(input)

 pred= '{0:{1}f}'.format(prediction[0][0],2)
 return float(pred) 

def main():
    st.title("Loan status ")
html_temp = """
    <div style= "background-color: #025246"; padding:10x>
    <h2 style="color:white; text-align:center;">Loan status prediction ML app </h2>
    </div>
  """

st.markdown(html_temp, unsafe_allow_html =True)
Age = st.text_input("Age","Type Here")

Interest = st.text_input("Interest","Type Here")

LoanDuration = st.text_input("LoanDuration","Type Here")

NoOfPreviousLoansBeforeLoan = st.text_input("NoOfPreviousLoansBeforeLoan","Type Here")

safe_html = """
<div style="background-color:#F4D03F"; padding:10px>
  <h2 style="color:white; text-align:Center;">Loan Approved </h2>
</div>
"""

danger_html = """
<div style="background-color:#F08080"; padding:10px>st
  <h2 style="color:black; text-align:Center;">Loan Not Approved </h2>
</div>
"""

if st.button("Predict"):
    output =predict_loan_status(Age,Interest,LoanDuration,NoOfPreviousLoansBeforeLoan)
    st.success('The Probability of loan getting approved is {}'.format(output))
    
    if output > 0.5:
        st.markdown(safe_html, unsafe_allow_html=True)
    else:
        st.markdown(danger_html, unsafe_allow_html=True)
        
        if __name__ == '__main__':
            main()
            
