import os
import streamlit as st
import docx2txt
import pdfplumber
import streamlit as st 
import streamlit.components as stc
import imblearn
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import pickle
model = pickle.load(open('ML_Model1.pkl', 'rb'))
Final_max = pickle.load(open('X_max.pkl', 'rb'))
Final_min = pickle.load(open('X_min.pkl', 'rb'))
Final_max = pd.read_pickle('X_max.pkl')
Final_min = pd.read_pickle('X_min.pkl')

Final=pd.DataFrame()


def data_eda(bank_application):
	null_percent = pd.DataFrame(((bank_application.isnull().sum()/len(bank_application.index))*100)).reset_index() # Create the column-wise null percentages from applicaton_data
	null_percent.columns = ['Column Name', 'Null Percentage']
	empty_cells= [ 'SK_ID_CURR','EMERGENCYSTATE_MODE', 'WALLSMATERIAL_MODE','TOTALAREA_MODE', 'HOUSETYPE_MODE', 'FONDKAPREMONT_MODE','NONLIVINGAREA_MEDI', 
           'NONLIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI','LIVINGAPARTMENTS_MEDI', 'LANDAREA_MEDI', 'FLOORSMIN_MEDI','FLOORSMAX_MEDI',
           'ENTRANCES_MEDI', 'ELEVATORS_MEDI','COMMONAREA_MEDI', 'YEARS_BUILD_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
           'BASEMENTAREA_MEDI', 'APARTMENTS_MEDI', 'NONLIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LANDAREA_MODE', 'FLOORSMIN_MODE', 'FLOORSMAX_MODE', 'ENTRANCES_MODE', 'ELEVATORS_MODE', 'COMMONAREA_MODE', 'YEARS_BUILD_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'BASEMENTAREA_MODE', 'APARTMENTS_MODE', 'NONLIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LANDAREA_AVG', 'FLOORSMIN_AVG', 'FLOORSMAX_AVG', 'ENTRANCES_AVG', 'ELEVATORS_AVG', 'COMMONAREA_AVG', 'YEARS_BUILD_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'BASEMENTAREA_AVG', 'APARTMENTS_AVG', 'EXT_SOURCE_1', 'OWN_CAR_AGE']
	flag_columns= [ 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7', 
           'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
           'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
           'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
	bank_application['TO_sub_doc']=(bank_application['FLAG_DOCUMENT_2']+bank_application['FLAG_DOCUMENT_3']+bank_application['FLAG_DOCUMENT_4']+bank_application['FLAG_DOCUMENT_5']+bank_application['FLAG_DOCUMENT_6']+
								 bank_application['FLAG_DOCUMENT_7']+bank_application['FLAG_DOCUMENT_8']+bank_application['FLAG_DOCUMENT_9']+bank_application['FLAG_DOCUMENT_10']+bank_application['FLAG_DOCUMENT_11']+bank_application['FLAG_DOCUMENT_12']+
								 bank_application['FLAG_DOCUMENT_13']+bank_application['FLAG_DOCUMENT_14']+bank_application['FLAG_DOCUMENT_15']+bank_application['FLAG_DOCUMENT_16']+bank_application['FLAG_DOCUMENT_17']+
								 bank_application['FLAG_DOCUMENT_18']+bank_application['FLAG_DOCUMENT_19']+bank_application['FLAG_DOCUMENT_20']+bank_application['FLAG_DOCUMENT_21'])
	flag_contact_cols = ['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE',
                 'FLAG_EMAIL']
	ext_source_cols = ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]
	unuse_cols = ["REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY","DAYS_LAST_PHONE_CHANGE"]
	non_important_col=empty_cells + flag_columns + flag_contact_cols + ext_source_cols + unuse_cols 
	#Dropping nulvalues col having more than 40% and other non-important cols
	bank_application.drop(labels=non_important_col,axis=1,inplace=True)
	#Imputing Missing values in the following columns with median which is 0.
	fill_list = ['AMT_REQ_CREDIT_BUREAU_HOUR',
	'AMT_REQ_CREDIT_BUREAU_DAY',
	'AMT_REQ_CREDIT_BUREAU_WEEK',
	'AMT_REQ_CREDIT_BUREAU_MON',
	'AMT_REQ_CREDIT_BUREAU_QRT',
	'AMT_REQ_CREDIT_BUREAU_YEAR',
	'OBS_30_CNT_SOCIAL_CIRCLE',
	'DEF_30_CNT_SOCIAL_CIRCLE',
	'OBS_60_CNT_SOCIAL_CIRCLE',
	'DEF_60_CNT_SOCIAL_CIRCLE']
	for x in fill_list:
		bank_application[x].fillna(0.0,inplace=True)
	#Here there are two blank values in CNT_FAM_MEMBERS  count of family members cannot have fractions or decimals, so we can fill the two blank values with Median which is 2
	bank_application['CNT_FAM_MEMBERS'] = bank_application['CNT_FAM_MEMBERS'].fillna(2)
	#As standard deviation is quite high, we can use Median value for imputation , Filling missing values with meadian
	bank_application['AMT_GOODS_PRICE']=bank_application['AMT_GOODS_PRICE'].fillna(450000.0)
	#categorical feature imputing with mode
	bank_application['NAME_TYPE_SUITE']=bank_application['NAME_TYPE_SUITE'].fillna('Unaccompanied')
	# Imputing variable 'OCCUPATION_TYPE', which has high null percentage(31.35%), with a new category as assigning it to any existing category might influence the analysis.
	bank_application.loc[pd.isnull(bank_application['OCCUPATION_TYPE']),['OCCUPATION_TYPE']]='Unknown'
	#As standard deviation is quite high, we can use Median value for imputation , Filling missing values with meadian
	bank_application['AMT_ANNUITY']=bank_application['AMT_ANNUITY'].fillna(24997.5)

	#Standarizing Values
	#In the application dataset some data are negative value like DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, these are days and we know that days cannot be negative.
	#convert negative days to positive days
	bank_application['DAYS_BIRTH']=bank_application['DAYS_BIRTH'].apply(lambda x: abs(x))
	bank_application['DAYS_EMPLOYED']=bank_application['DAYS_EMPLOYED'].apply(lambda x: abs(x))
	bank_application['DAYS_REGISTRATION']=bank_application['DAYS_REGISTRATION'].apply(lambda x: abs(x))
	bank_application['DAYS_ID_PUBLISH']=bank_application['DAYS_ID_PUBLISH'].apply(lambda x: abs(x))
	# Converting days birth into year
	bank_application['DAYS_BIRTH'] = bank_application['DAYS_BIRTH']/365
	#Binning the birth date 
	bank_application['AGE_GROUP'] = pd.cut(bank_application['DAYS_BIRTH'],bins=[20,30,40,50,60,70] , labels = ["20Y-30Y","30Y-40Y","40Y-50Y","50Y-60Y","60Y-70Y"])
	#Converting days employed into year
	bank_application['DAYS_EMPLOYED'] = bank_application['DAYS_EMPLOYED']/365
	bins=[0,5,10,20,30,40,50,60,100]
	slots=['0-5','5-10','10-20','20-30','30-40','40-50','50-60','Above 60']
	bank_application['EMPLOYMENT_GROUP']=pd.cut(bank_application['DAYS_EMPLOYED'],bins=bins,labels=slots)
	bank_application['AMT_INCOME_TOTAL']=bank_application.AMT_INCOME_TOTAL/100000
	bank_application['AMT_CREDIT']=bank_application.AMT_CREDIT/100000
	
	#converting the feature values with two unique values into categorical feature
	bank_application['REG_REGION_NOT_LIVE_REGION'].replace((0, 1), ('N', 'Y'), inplace=True)
	bank_application['REG_REGION_NOT_WORK_REGION'].replace((0, 1), ('N', 'Y'), inplace=True)
	bank_application['LIVE_REGION_NOT_WORK_REGION'].replace((0, 1), ('N', 'Y'), inplace=True)
	bank_application['REG_REGION_NOT_LIVE_REGION'].replace((0, 1), ('N', 'Y'), inplace=True)
	bank_application['REG_CITY_NOT_LIVE_CITY'].replace((0, 1), ('N', 'Y'), inplace=True)
	bank_application['REG_CITY_NOT_WORK_CITY'].replace((0, 1), ('N', 'Y'), inplace=True)
	bank_application['LIVE_CITY_NOT_WORK_CITY'].replace((0, 1), ('N', 'Y'), inplace=True)
	
	#Changing the name of the value to 'Secondary' from 'Secondary / secondary special'
	
	bank_application['NAME_EDUCATION_TYPE'] = bank_application['NAME_EDUCATION_TYPE'].apply(lambda x: 'Secondary' if x=='Secondary / secondary special' else x)
	
	#Changing the name of the value to 'Single' from 'Single / not married'
	
	bank_application['NAME_FAMILY_STATUS'] = bank_application['NAME_FAMILY_STATUS'].apply(lambda x: 'Single' if x=='Single / not married' else x)
	
	#Imputing undefined values with the max
	bank_application['CODE_GENDER'].replace(('XNA'), ('F'), inplace=True)
	bank_application['ORGANIZATION_TYPE'].replace(('XNA'), ('Business Entity Type 3'), inplace=True)
	
	#Lets create the parameters required for binning.
	bins=[0,1,2,3,4,5,6,7,8,9,10,50]
	slots=['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k','800k-900k','900k-1M', 'Above 1M']
	bank_application['AMT_INCOME_TOTAL_RANGE']=pd.cut(bank_application['AMT_INCOME_TOTAL'],bins=bins,labels=slots)
	bins=[0,1,2,3,4,5,6,7,8,9,10,50]
	slots=['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k','800k-900k','900k-1M', 'Above 1M']
	bank_application['AMT_CREDIT_RANGE']=pd.cut(bank_application['AMT_CREDIT'],bins=bins,labels=slots)


	return bank_application

# Utils
import base64 
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
import pandas as pd 
from PIL import Image


def main():
	st.title("Loan Default Prediction")

	menu = ["Home","Dataset"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")

	elif choice == "Dataset":
		st.subheader("Dataset")


	return choice

# Include PIL, load_image before main()

def load_image(image_file):
	img = Image.open(image_file)
	return img

class FileDownloader(object):
	
	def __init__(self, data,filename='myfile',file_ext='txt'):
		super(FileDownloader, self).__init__()
		self.data = data
		self.filename = filename
		self.file_ext = file_ext

	def download(self):
		b64 = base64.b64encode(self.data.encode()).decode()
		new_filename = "{}_{}_.{}".format(self.filename,timestr,self.file_ext)
		st.markdown("#### Download File ###")
		href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Click Here!!</a>'
		st.markdown(href,unsafe_allow_html=True)

# def main():
# 	menu = ["Text","CSV"]

# 	choice = st.sidebar.selectbox("Menu",menu)

# 	if choice == "Text":
# 		st.subheader("Text")
# 		my_text = st.text_area("Your Message")
# 		if st.button("Save"):
# 			st.write(my_text)
# 			download = FileDownloader(my_text).download()

# 	elif choice == "CSV":
# 		df = pd.read_csv("iris.csv")
# 		st.dataframe(df)
# 		download = FileDownloader(df.to_csv(),file_ext='csv').download()

if __name__ == '__main__':
	choice = main()



if choice == "Home":
		"""In this project we need to predict whether to sanction or reject the loan. Using the data 
given, we must ensure that honest clients are not rejected from giving the loans by 
building a machine learning model.
This an important problem to solve, as the loan companies grant a loan after an 
intensive process of verification and validation. However, they still don’t have assurance 
if the applicant is able to repay the loan with no difficulties and they do follow a manual 
procedure to determine whether a loan can be approved for an applicant based on 
results, manual procedures were effective but normally time consuming if the data size 
is high and deciding would take a long time.
In this Project, we’ll build a predictive model to predict whether to sanction or reject the 
loan, which can be used to make decision quick and effective even if the data size huge."""

elif choice == "Dataset":
		st.subheader("Dataset")
		data_file = st.file_uploader("Upload CSV",type=["csv"])
		if data_file is not None:
			file_details = {"filename":data_file.name, "filetype":data_file.type,
                            "filesize":data_file.size}
			
			st.write(file_details)
			bank_application = pd.read_csv(data_file)
			st.dataframe(bank_application)
			Input_data=data_eda(bank_application)
			 
			# Generating dummy variables for rest of the nominal variables
			
			Input_data=pd.get_dummies(Input_data)


			# Maintaining the same order of columns as it was during the model training
			cols=['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'CNT_FAM_MEMBERS', 'HOUR_APPR_PROCESS_START', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'TO_sub_doc', 'NAME_CONTRACT_TYPE_Cash loans', 'NAME_CONTRACT_TYPE_Revolving loans', 'CODE_GENDER_F', 'CODE_GENDER_M', 'FLAG_OWN_CAR_N', 'FLAG_OWN_CAR_Y', 'FLAG_OWN_REALTY_N', 'FLAG_OWN_REALTY_Y', 'NAME_TYPE_SUITE_Children', 'NAME_TYPE_SUITE_Family', 'NAME_TYPE_SUITE_Group of people', 'NAME_TYPE_SUITE_Other_A', 'NAME_TYPE_SUITE_Other_B', 'NAME_TYPE_SUITE_Spouse, partner', 'NAME_TYPE_SUITE_Unaccompanied', 'NAME_INCOME_TYPE_Businessman', 'NAME_INCOME_TYPE_Commercial associate', 'NAME_INCOME_TYPE_Maternity leave', 'NAME_INCOME_TYPE_Pensioner', 'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Student', 'NAME_INCOME_TYPE_Working', 'NAME_EDUCATION_TYPE_Academic degree', 'NAME_EDUCATION_TYPE_Higher education', 'NAME_EDUCATION_TYPE_Incomplete higher', 'NAME_EDUCATION_TYPE_Lower secondary', 'NAME_EDUCATION_TYPE_Secondary', 'NAME_FAMILY_STATUS_Civil marriage', 'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Separated', 'NAME_FAMILY_STATUS_Single', 'NAME_FAMILY_STATUS_Unknown', 'NAME_FAMILY_STATUS_Widow', 'NAME_HOUSING_TYPE_Co-op apartment', 'NAME_HOUSING_TYPE_House / apartment', 'NAME_HOUSING_TYPE_Municipal apartment', 'NAME_HOUSING_TYPE_Office apartment', 'NAME_HOUSING_TYPE_Rented apartment', 'NAME_HOUSING_TYPE_With parents', 'OCCUPATION_TYPE_Accountants', 'OCCUPATION_TYPE_Cleaning staff', 'OCCUPATION_TYPE_Cooking staff', 'OCCUPATION_TYPE_Core staff', 'OCCUPATION_TYPE_Drivers', 'OCCUPATION_TYPE_HR staff', 'OCCUPATION_TYPE_High skill tech staff', 'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Laborers', 'OCCUPATION_TYPE_Low-skill Laborers', 'OCCUPATION_TYPE_Managers', 'OCCUPATION_TYPE_Medicine staff', 'OCCUPATION_TYPE_Private service staff', 'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff', 'OCCUPATION_TYPE_Secretaries', 'OCCUPATION_TYPE_Security staff', 'OCCUPATION_TYPE_Unknown', 'OCCUPATION_TYPE_Waiters/barmen staff', 'WEEKDAY_APPR_PROCESS_START_FRIDAY', 'WEEKDAY_APPR_PROCESS_START_MONDAY', 'WEEKDAY_APPR_PROCESS_START_SATURDAY', 'WEEKDAY_APPR_PROCESS_START_SUNDAY', 'WEEKDAY_APPR_PROCESS_START_THURSDAY', 'WEEKDAY_APPR_PROCESS_START_TUESDAY', 'WEEKDAY_APPR_PROCESS_START_WEDNESDAY', 'REG_REGION_NOT_LIVE_REGION_N', 'REG_REGION_NOT_LIVE_REGION_Y', 'REG_REGION_NOT_WORK_REGION_N', 'REG_REGION_NOT_WORK_REGION_Y', 'LIVE_REGION_NOT_WORK_REGION_N', 'LIVE_REGION_NOT_WORK_REGION_Y', 'REG_CITY_NOT_LIVE_CITY_N', 'REG_CITY_NOT_LIVE_CITY_Y', 'REG_CITY_NOT_WORK_CITY_N', 'REG_CITY_NOT_WORK_CITY_Y', 'LIVE_CITY_NOT_WORK_CITY_N', 'LIVE_CITY_NOT_WORK_CITY_Y', 'ORGANIZATION_TYPE_Advertising', 'ORGANIZATION_TYPE_Agriculture', 'ORGANIZATION_TYPE_Bank', 'ORGANIZATION_TYPE_Business Entity Type 1', 'ORGANIZATION_TYPE_Business Entity Type 2', 'ORGANIZATION_TYPE_Business Entity Type 3', 'ORGANIZATION_TYPE_Cleaning', 'ORGANIZATION_TYPE_Construction', 'ORGANIZATION_TYPE_Culture', 'ORGANIZATION_TYPE_Electricity', 'ORGANIZATION_TYPE_Emergency', 'ORGANIZATION_TYPE_Government', 'ORGANIZATION_TYPE_Hotel', 'ORGANIZATION_TYPE_Housing', 'ORGANIZATION_TYPE_Industry: type 1', 'ORGANIZATION_TYPE_Industry: type 10', 'ORGANIZATION_TYPE_Industry: type 11', 'ORGANIZATION_TYPE_Industry: type 12', 'ORGANIZATION_TYPE_Industry: type 13', 'ORGANIZATION_TYPE_Industry: type 2', 'ORGANIZATION_TYPE_Industry: type 3', 'ORGANIZATION_TYPE_Industry: type 4', 'ORGANIZATION_TYPE_Industry: type 5', 'ORGANIZATION_TYPE_Industry: type 6', 'ORGANIZATION_TYPE_Industry: type 7', 'ORGANIZATION_TYPE_Industry: type 8', 'ORGANIZATION_TYPE_Industry: type 9', 'ORGANIZATION_TYPE_Insurance', 'ORGANIZATION_TYPE_Kindergarten', 'ORGANIZATION_TYPE_Legal Services', 'ORGANIZATION_TYPE_Medicine', 'ORGANIZATION_TYPE_Military', 'ORGANIZATION_TYPE_Mobile', 'ORGANIZATION_TYPE_Other', 'ORGANIZATION_TYPE_Police', 'ORGANIZATION_TYPE_Postal', 'ORGANIZATION_TYPE_Realtor', 'ORGANIZATION_TYPE_Religion', 'ORGANIZATION_TYPE_Restaurant', 'ORGANIZATION_TYPE_School', 'ORGANIZATION_TYPE_Security', 'ORGANIZATION_TYPE_Security Ministries', 'ORGANIZATION_TYPE_Self-employed', 'ORGANIZATION_TYPE_Services', 'ORGANIZATION_TYPE_Telecom', 'ORGANIZATION_TYPE_Trade: type 1', 'ORGANIZATION_TYPE_Trade: type 2', 'ORGANIZATION_TYPE_Trade: type 3', 'ORGANIZATION_TYPE_Trade: type 4', 'ORGANIZATION_TYPE_Trade: type 5', 'ORGANIZATION_TYPE_Trade: type 6', 'ORGANIZATION_TYPE_Trade: type 7', 'ORGANIZATION_TYPE_Transport: type 1', 'ORGANIZATION_TYPE_Transport: type 2', 'ORGANIZATION_TYPE_Transport: type 3', 'ORGANIZATION_TYPE_Transport: type 4', 'ORGANIZATION_TYPE_University']
			for i in cols:
				if i in Input_data:
					Final[i]=Input_data[i]
				else:
					Final[i]=0 


			Final_input=(Final-Final_min)/(Final_max-Final_min)


			if st.button("Submit"):
				prediction = model.predict(Final_input)
				lc = [str(i) for i in prediction]
				ans = int("".join(lc))
				if ans == 1:
					st.error(
                'According to our Calculations, you will not get the loan from Bank'
				)
				else:
					st.success(
                'Congratulations!! you will get the loan from Bank'
				)








