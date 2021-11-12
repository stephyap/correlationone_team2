import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# resources: https://docs.streamlit.io/knowledge-base/tutorials/databases/tableau
# https://towardsdatascience.com/coding-ml-tools-like-you-code-ml-models-ddba3357eace
# https://towardsdatascience.com/embedding-tableau-in-streamlit-a9ce290b932b
# https://docs.streamlit.io/library/get-started/installation#prerequisites


######### IMPORT DATA #########
@st.cache(suppress_st_warning=True)
def import_data():
    nsduh = pd.read_csv('./Data/nsduh_data_cleaned.csv')
    return nsduh


df = import_data()

# figure out difference for local vs cloud


######### SIDE BAR #########
st.sidebar.markdown("""
# Analyzing Susceptibility to Mental Health Issues
""")
# Outline Options for Sidebar
section = st.sidebar.selectbox("Outline", ("General Overview", "Datasets",
                               "Exploratory Data Analysis", "Methodology",
                                           "Findings and Recommendation", "Resources"))

st.sidebar.markdown("""
# DS4A Women 2021 - Team 2
# Whitney Brooks (Executive)
# Catherine Greenman (Practitioner)
# Margot Herman (Practitioner)
# Michell Li (Practitioner)
# Chiu-Feng (Steph) Yap (Practitioner)
""")


######### EXECUTIVE SUMMARY #########
if section == "General Overview":

    st.title('Analyzing Susceptibility to Mental Health Issues')
    st.write('''
	# Problem Statement
	While the stigma around mental health has decreased over the years, many providers have seen a spike in cases related to “diseases of despair.” These include anxiety and depression, which often go untreated or lead sufferers to “self-medicate” with substances like drugs and alcohol. According to the Tufts Medical Center and One Mind at Work, depression alone accounts for about $44 billion in losses to workplace productivity. In 2019, national spending on mental health services totaled $225.1 billion and accounted for 5.5% of all health spending1. Furthermore, approximately 40% of Americans live in a designated mental health provider shortage area, which exacerbates the problem. Across the US, each state has discretionary funding allocated specifically for mental health. Sufficient funds and effective resource allocation are necessary for the diagnosis and treatment of mental health issues. Mental health issues are pervasive and, now more than ever, need to be better understood to address their causes and impacts in a meaningful way.

	# Objective
	The goal of this project is to identify factors that make individuals more susceptible to mental health issues, based on self-administered substance use, demographics, and geographic information from the National Survey on Drug Use and Health (NSDUH).
	''')


######### DATASETS #########
if section == "Datasets":

    st.title('Datasets')

    st.write('''
    ## Datasets Used
    1. National Survey of Drug Use and Health (NSDUH) - individual level
    2. Health Professional Shortage Areas (HPSA) - county level
    3. Health Resources and Services Administration (HRSA) - county level
    ''')

    st.write('''
	## Outcome Variable
	The outcome variable is captured in the data as a binary indicator of 1 (Yes) for ‘Past Month Serious Psychological Distress Indicator’ which is derived from a series of six questions from NSDUH, asking adults respondents how frequently they experienced the following symptoms in the past 30 days:

		* How often did you feel nervous?
		* How often did you feel hopeless?
		* How often did you feel restless or fidgety?
		* How often did you feel so sad/depressed that nothing could cheer you up?
		* How often did you feel that everything was an effort?
		* How often did you feel down on yourself, no good or worthless?

	Questions are asked on a likert scale of 1-5, with a sum greater than 13 being the threshold for the outcome variable.
	''')

    st.write(''' ### Top 10 rows of NSDUH dataset ''')
    st.dataframe(df.head(10))

# All columns in df = 'Id', 'Year', 'Inpatient_Past_Year', 'Outpatient_Past_Year', 'Prescription_Treatment_Past_Year', 'Any_Treatment_Past_Year', 'Treatment_Type_Past_Year', 'Perceived_Unmet_Need', 'Received_Treatment_At_Clinic_Or_Center', 'Received_Treatment_At_Private_Therapist', 'Received_Treatment_At_NonClinic_Doctor', 'Received_Treatment_At_Medical_Clinic', 'Received_Treatment_At_Day_Hospital', 'Received_Treatment_At_School', 'Received_Treatment_Other', 'Self_Paid', 'Non_Household_Member_Paid', 'Private_Health_Insurance_Paid', 'Medicare_Paid', 'Medicaid_Paid', 'Rehab_Paid', 'Employer_Paid', 'Military_Paid', 'Other_Public_Source_Paid', 'Other_Private_Source_Paid', 'Nobody_Paid', 'No_Treatment_Could_Not_Afford', 'No_Treatment_Feared_Neighbors_Opinion', 'No_Treatment_Feared_Effect_On_Job', 'No_Treatment_Insurance_Not_Covered', 'No_Treatment_Insurance_Not_Covered_Enough', 'No_Treatment_Where_To_Go', 'No_Treatment_Confidentiality_Concerns', 'No_Treatment_Fear_Of_Being_Committed', 'No_Treatment_Didnt_Think_Needed', 'No_Treatment_Handle_Problem_Without', 'No_Treatment_Didnt_Think_Would_Help', 'No_Treatment_Didnt_Have_Time', 'No_Treatment_Didnt_Want_Others_To_Know', 'No_Treatment_No_Transport_Inconvenient', 'No_Treatment_Other', 'Num_Weeks_Mental_Health_Difficulties', 'Num_Days_Past_Year_Unable_To_Work', 'Serious_Psychological_Distress_Indicator_Past_Month', 'Psychological_Distress_Level_Worst_Month', 'Worst_Psychological_Distress_Level', 'Serious_Psychological_Distress_Indicator_Past_Year', 'Serious_Suicidal_Thoughts_Past_Year', 'Suicide_Plan_Past_Year', 'Suicide_Attempt_Past_Year', 'Serious_Or_Moderate_Mental_Illness_Indicator_Past_Year', 'Moderate_Mental_Illness_Indicator_Past_Year', 'Mild_Mental_Illness_Indicator_Past_Year', 'Low_Or_Moderate_Mental_Illness_Indicator_Past_Year', 'Categorical_Mental_Illness_Indicator', 'Serious_Mental_Illness_And_Substance_Abuse', 'Any_Mental_Illness_And_Substance_Abuse', 'Low_Or_Moderate_Mental_Illness_And_Substance_Abuse', 'Adult_Lifetime_Major_Depressive_Episode', 'Adult_Past_Year_Major_Depressive_Episode', 'Adult_Received_Counseling_Or_Meds_For_Depressive_Feelings_Past_Year', 'Gender', 'Age_Category', 'Age_Category_Two_Levels', 'Age_Category_Three_Levels', 'Age_Category_Six_Levels', 'Gender_Age_Category', 'Race_Ethnicity', 'Race_Sex', 'Education_Category', 'Overall_Health', 'Work_Situation_Past_Week', 'Num_Days_Skipped_Work_Past_30_Days', 'EAP_Offered', 'Adult_Employment_Status', 'Has_Medicare', 'Has_Medicaid_Or_CHIP', 'Has_Military_Benefit', 'Has_Private_Health_Insurance', 'Has_Other_Health_Insurance', 'Covered_By_Any_Health_Insurance', 'Covered_By_Any_Health_Insurance_Imputation_Revised', 'Family_Receives_Social_Security', 'Family_Receives_SSI', 'Family_Receives_Food_Stamps', 'Family_Receives_Public_Assistance', 'Family_Receives_Welfare_JobPlacement_Childcare', 'Months_On_Welfare', 'Total_Income_Respondent', 'Total_Income_Family', 'Participated_In_One_Or_More_Government_Assistance_Programs', 'Total_Income_Family_Recode', 'Poverty_Level', 'PDEN10', 'County_Metro_NonMetro_Status'


######### EXPLORATORY DATA ANALYSIS #########
if section == "Exploratory Data Analysis":

    st.title('Exploratory Data Analysis')


######### METHODOLOGY #########
if section == "Methodology":

    st.title('Methodology')


######### FINDINGS AND RECOMMENDATION #########
if section == "Findings and Recommendation":

    st.title('Findings and Recommendation')


######### RESOURCES #########
if section == "Resources":

    st.title('Resources')
