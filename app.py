import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model (ensure that model.pkl is in the same directory or adjust the path)
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Household Total Expense Prediction")
st.markdown("Enter your household details below:")

# -------------------------
# State Mapping: Indian states mapped 1..29 (alphabetical order)
state_mapping = {
    "Andhra Pradesh": 1,
    "Arunachal Pradesh": 2,
    "Assam": 3,
    "Bihar": 4,
    "Chhattisgarh": 5,
    "Goa": 6,
    "Gujarat": 7,
    "Haryana": 8,
    "Himachal Pradesh": 9,
    "Jharkhand": 10,
    "Karnataka": 11,
    "Kerala": 12,
    "Madhya Pradesh": 13,
    "Maharashtra": 14,
    "Manipur": 15,
    "Meghalaya": 16,
    "Mizoram": 17,
    "Nagaland": 18,
    "Odisha": 19,
    "Punjab": 20,
    "Rajasthan": 21,
    "Sikkim": 22,
    "Tamil Nadu": 23,
    "Telangana": 24,
    "Tripura": 25,
    "Uttar Pradesh": 26,
    "Uttarakhand": 27,
    "West Bengal": 28,
    "Other": 29
}

# -------------------------
# Household-level Inputs
Sector = st.number_input("Sector", value=1, step=1)
state_name = st.selectbox("Select your State", options=list(state_mapping.keys()))
State = state_mapping[state_name]
nss_region = st.number_input("NSS-Region", value=213, step=1)
District = st.number_input("District", value=6, step=1)
Household_Type = st.number_input("Household Type", value=6, step=1)
Religion = st.number_input("Religion of the head of the household", value=1, step=1)
Social_Group = st.number_input("Social Group of the head of the household", value=1, step=1)
HH_Size = st.number_input("HH Size (For FDQ)", value=2, step=1)
NCO_3D = st.number_input("NCO_3D", value=963.0)
NIC_5D = st.number_input("NIC_5D", value=1110.0)

# -------------------------
# Online Purchase Features (Binary: use 0 for No, 1 for Yes)
st.markdown("### Online Purchase Behavior (Last 365 Days)")
Is_online_Clothing = st.selectbox("Is online Clothing Purchased Last365?", options=[0, 1], index=0)
Is_online_Footwear = st.selectbox("Is online Footwear Purchased Last365?", options=[0, 1], index=0)
Is_online_Furniture = st.selectbox("Is online Furniture fixtures Purchased Last365?", options=[0, 1], index=0)
Is_online_Mobile = st.selectbox("Is online Mobile Handset Purchased Last365?", options=[0, 1], index=0)
Is_online_Personal_Goods = st.selectbox("Is online Personal Goods Purchased Last365?", options=[0, 1], index=0)
Is_online_Recreation = st.selectbox("Is online Recreation Goods Purchased Last365?", options=[0, 1], index=0)
Is_online_Household_Appliances = st.selectbox("Is online Household Appliances Purchased Last365?", options=[0, 1], index=0)
Is_online_Crockery = st.selectbox("Is online Crockery Utensils Purchased Last365?", options=[0, 1], index=0)
Is_online_Sports = st.selectbox("Is online Sports Goods Purchased Last365?", options=[0, 1], index=0)
Is_online_Medical = st.selectbox("Is online Medical Equipment Purchased Last365?", options=[0, 1], index=0)
Is_online_Bedding = st.selectbox("Is online Bedding Purchased Last365?", options=[0, 1], index=0)

# -------------------------
# Household Assets (Binary: use 0 for No, 1 for Yes)
st.markdown("### Household Assets")
Is_HH_Have_Television = st.selectbox("Does the household have a Television?", options=[0, 1], index=1)
Is_HH_Have_Radio = st.selectbox("Does the household have a Radio?", options=[0, 1], index=0)
Is_HH_Have_Laptop_PC = st.selectbox("Does the household have a Laptop/PC?", options=[0, 1], index=0)
Is_HH_Have_Mobile_handset = st.selectbox("Does the household have a Mobile handset?", options=[0, 1], index=1)
Is_HH_Have_Bicycle = st.selectbox("Does the household have a Bicycle?", options=[0, 1], index=0)
Is_HH_Have_Motorcycle_scooter = st.selectbox("Does the household have a Motorcycle/Scooter?", options=[0, 1], index=0)
Is_HH_Have_Motorcar = st.selectbox("Does the household have a Motorcar/Jeep/Van?", options=[0, 1], index=0)
Is_HH_Have_Trucks = st.selectbox("Does the household have Trucks?", options=[0, 1], index=0)
Is_HH_Have_Animal_cart = st.selectbox("Does the household have an Animal cart?", options=[0, 1], index=0)
Is_HH_Have_Refrigerator = st.selectbox("Does the household have a Refrigerator?", options=[0, 1], index=1)
Is_HH_Have_Washing_machine = st.selectbox("Does the household have a Washing Machine?", options=[0, 1], index=0)
Is_HH_Have_Airconditioner = st.selectbox("Does the household have an Airconditioner/Aircooler?", options=[0, 1], index=0)

# -------------------------
# Aggregated Person-level Features
st.markdown("### Person-Level Aggregated Features")
count_adults = st.number_input("Count of adults (Age 18-64)", value=3, step=1)
adults_mean_age = st.number_input("Adults mean age", value=35.0)
count_children = st.number_input("Count of children (under 18)", value=1, step=1)
children_mean_age = st.number_input("Children mean age", value=10.0)
count_elders = st.number_input("Count of elders (65+)", value=0, step=1)
elders_mean_age = st.number_input("Elders mean age", value=0.0)
count_males = st.number_input("Count of males", value=2, step=1)
count_females = st.number_input("Count of females", value=2, step=1)
# Optionally, you can either let the user input gender_ratio manually or compute it.
gender_ratio = st.number_input("Gender ratio (females/males)", value=1.0)
mean_age = st.number_input("Mean age", value=30.0)

# Compute dependency ratio automatically: (children + elders) / adults (if adults > 0)
if count_adults > 0:
    dependency_ratio = (count_children + count_elders) / count_adults
else:
    dependency_ratio = 0.0
st.write(f"Computed Dependency Ratio: {dependency_ratio:.2f}")

count_married = st.number_input("Count of married persons", value=2, step=1)
max_education = st.number_input("Maximum educational level attained", value=4, step=1)
count_internet_users = st.number_input("Count of internet users", value=1, step=1)
total_meals_school = st.number_input("Total meals from school", value=0.0)
total_meals_employer = st.number_input("Total meals from employer", value=0.0)
total_meals_payment = st.number_input("Total meals on payment", value=0.0)
total_meals_home = st.number_input("Total meals at home", value=180.0)
total_meals_others = st.number_input("Total meals from others", value=0.0)

# -------------------------
# Assemble the features into a DataFrame.
# The columns and their names must match those used during model training.
input_data = pd.DataFrame({
    "Sector": [Sector],
    "State": [State],
    "NSS-Region": [nss_region],
    "District": [District],
    "Household Type": [Household_Type],
    "Religion of the head of the household": [Religion],
    "Social Group of the head of the household": [Social_Group],
    "HH Size (For FDQ)": [HH_Size],
    "NCO_3D": [NCO_3D],
    "NIC_5D": [NIC_5D],
    "Is_online_Clothing_Purchased_Last365": [Is_online_Clothing],
    "Is_online_Footwear_Purchased_Last365": [Is_online_Footwear],
    "Is_online_Furniture_fixturesPurchased_Last365": [Is_online_Furniture],
    "Is_online_Mobile_Handset_Purchased_Last365": [Is_online_Mobile],
    "Is_online_Personal_Goods_Purchased_Last365": [Is_online_Personal_Goods],
    "Is_online_Recreation_Goods_Purchased_Last365": [Is_online_Recreation],
    "Is_online_Household_Appliances_Purchased_Last365": [Is_online_Household_Appliances],
    "Is_online_Crockery_Utensils_Purchased_Last365": [Is_online_Crockery],
    "Is_online_Sports_Goods_Purchased_Last365": [Is_online_Sports],
    "Is_online_Medical_Equipment_Purchased_Last365": [Is_online_Medical],
    "Is_online_Bedding_Purchased_Last365": [Is_online_Bedding],
    "Is_HH_Have_Television": [Is_HH_Have_Television],
    "Is_HH_Have_Radio": [Is_HH_Have_Radio],
    "Is_HH_Have_Laptop_PC": [Is_HH_Have_Laptop_PC],
    "Is_HH_Have_Mobile_handset": [Is_HH_Have_Mobile_handset],
    "Is_HH_Have_Bicycle": [Is_HH_Have_Bicycle],
    "Is_HH_Have_Motorcycle_scooter": [Is_HH_Have_Motorcycle_scooter],
    "Is_HH_Have_Motorcar_jeep_van": [Is_HH_Have_Motorcar],
    "Is_HH_Have_Trucks": [Is_HH_Have_Trucks],
    "Is_HH_Have_Animal_cart": [Is_HH_Have_Animal_cart],
    "Is_HH_Have_Refrigerator": [Is_HH_Have_Refrigerator],
    "Is_HH_Have_Washing_machine": [Is_HH_Have_Washing_machine],
    "Is_HH_Have_Airconditioner_aircooler": [Is_HH_Have_Airconditioner],
    "count_adults": [count_adults],
    "adults_mean_age": [adults_mean_age],
    "count_children": [count_children],
    "children_mean_age": [children_mean_age],
    "count_elders": [count_elders],
    "elders_mean_age": [elders_mean_age],
    "count_males": [count_males],
    "count_females": [count_females],
    "gender_ratio": [gender_ratio],
    "mean_age": [mean_age],
    "dependency_ratio": [dependency_ratio],
    "count_married": [count_married],
    "max_education": [max_education],
    "count_internet_users": [count_internet_users],
    "total_meals_school": [total_meals_school],
    "total_meals_employer": [total_meals_employer],
    "total_meals_payment": [total_meals_payment],
    "total_meals_home": [total_meals_home],
    "total_meals_others": [total_meals_others]
})

# -------------------------
# Prediction Button
if st.button("Predict Total Expense"):
    # The model was trained on log-transformed target values,
    # so we need to inverse-transform the predictions using np.expm1.
    y_pred_log = model.predict(input_data)
    y_pred = np.expm1(y_pred_log)
    st.success(f"Predicted Total Expense: {y_pred[0]:.2f}")
