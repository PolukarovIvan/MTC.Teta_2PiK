import streamlit as st
import pandas as pd
import base64
import joblib
from random import randint
from prep import FeatureSelector, FeatureGemerator
import shap


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="answers.csv">Download answers.csv</a>'

def to_df(ID,Age,Exp,Incm,Fam,CCAvg,education,Mort,sa,cd,online,credit):
	eds = { 'Undergrad':0, 'Graduate':1,'Advanced/Professional':2}
	df = pd.DataFrame(index=[0])
	df['ID']=ID
	df['Age'] = Age
	df['Experience'] = Exp
	df['Income'] = Incm
	df['ZIP Code'] = randint(0,100)
	df['CCAvg'] = CCAvg
	df['Education'] = eds[education]
	df['Mortgage'] = Mort
	df['Family'] = Fam
	df['Securities Account'] = int(sa)
	df['CD Account'] = int(cd)
	df['Online'] = int(online)
	df['CreditCard'] = int(credit)
	return df


model = joblib.load('models/cb_model.pkl')
target_name = ["won't accept",'will accept']

st.title('Demo of personal loan prediction model')
with st.form('text'):
	c1,c2,c3 = st.columns([1,1,1])
	c4,c5,c6 = st.columns([1,1,1])
	ID=randint(0,300)
	Age = c1.number_input("Enter client's age", min_value = 21)
	Exp = c2.number_input("Enter client's proffesional experience", min_value = 0)
	Incm = c3.number_input("Enter client's annual income($000)", value = 30, step = 5)
	Fam = c4.number_input("Enter the client's size family", min_value = 1)
	CCAvg = c5.number_input("Enter client's average spending on credit cards per month ($000)", min_value = 0.0)
	Mort = c6.number_input("Enter the mortrage size (if client has it)", min_value = 0.0, step = 50.0)
	education = st.selectbox("Select client's education level", ( 'Undergrad', 'Graduate','Advanced/Professional'))
	sa = st.checkbox('Client has a securities account with the bank')
	cd = st.checkbox('Client has a certificate of deposit account with the bank')
	online = st.checkbox('Client use internet banking facilities')
	credit = st.checkbox('Client use a credit card issued by the bank')
	button = st.form_submit_button('Click here to predict one person')
	if button:
		test_df = to_df(ID,Age,Exp,Incm,Fam,CCAvg,education,Mort,sa,cd,online,credit)
		pipeline = joblib.load('data_engeneering_pipeline.pkl')
		ready_to_predict = pipeline.fit_transform(test_df)
		pred = model.predict(ready_to_predict)
		prob = model.predict_proba(ready_to_predict)
		explainer = shap.TreeExplainer(model)
		shap_values = explainer.shap_values(ready_to_predict)
		name = f'Сlient {target_name[pred[0]]} offer with {prob[0][pred[0]]:.3f} probability' 
		st.write(name)
		st.pyplot(shap.force_plot(explainer.expected_value[pred[0]], shap_values[pred[0]],
		 ready_to_predict.iloc[0], show=False, matplotlib=True, out_names='probability of prediction'))



st.write('OR')

with st.form('table'):
	uploaded_file = st.file_uploader("Upload a csv file", ["csv"])
	file_button = st.form_submit_button('Predict labels')
	if uploaded_file:
		file = pd.read_csv(uploaded_file)