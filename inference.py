import pandas as pd
import numpy as np
import lightgbm as lgb


def make_predictions(hh_data_df, pp_data_df):
    hh_data_df['NCO_3D'].fillna(963, inplace=True)
    pp_data_df['NIC_5D'].fillna(1110, inplace=True)

    hh_data_df.fillna(0, inplace=True)
    pp_data_df.fillna(0, inplace=True)

    # Group the person-level dataset by HH_ID and compute the aggregated features.
    person_features = pp_data_df.groupby('HH_ID').apply(compute_features).reset_index()

    # Merge the computed features into the household dataset using HH_ID.
    master_df = pd.merge(hh_data_df, person_features, on='HH_ID', how='left')

    # fill missing values of person features with 0
    master_df.fillna(0, inplace=True)

    # Optional: Save the master dataset to a new CSV file.
    # master_df.to_csv("aggregated_data_test.csv", index=False)
    
    # Drop the first column
    master_df = master_df.drop(columns=['HH_ID'])

    # if master data contains TotalExpense column, drop it
    if 'TotalExpense' in master_df.columns:
        master_df = master_df.drop(columns=['TotalExpense'])

    # Load the trained model from the file
    model = lgb.Booster(model_file='mpce_model.txt')
    
    # Make predictions
    y_pred = model.predict(master_df)
    y_pred = np.expm1(y_pred)

    return y_pred

    
def compute_features(group):
    # Count of adults: Age 18-64 (inclusive)
    count_adults = group[(group['Age(in years)'] >= 18) & (group['Age(in years)'] <= 64)].shape[0]
    adults_mean_age = group[(group['Age(in years)'] >= 18) & (group['Age(in years)'] <= 64)]['Age(in years)'].mean() 

    
    # Count of children: Age below 18
    count_children = group[group['Age(in years)'] < 18].shape[0]
    children_mean_age = group[group['Age(in years)'] < 18]['Age(in years)'].mean()
    
    # Count of elders: Age 65 and above
    count_elders = group[group['Age(in years)'] >= 65].shape[0]
    elders_mean_age = group[group['Age(in years)'] >= 65]['Age(in years)'].mean()
    
    # Gender ratio: Count females divided by count males
    # Adjust the gender codes if needed. Here we assume: 1=Male, 2=Female.
    count_males = group[group['Gender'] == 1].shape[0]
    count_females = group[group['Gender'] == 2].shape[0]
    gender_ratio = count_females / (count_males + 1) # Add 1 to avoid division by zero
    
    # Mean age
    mean_age = group['Age(in years)'].mean()
    
    # Dependency ratio: (Count of children + Count of elders) / Count of adults
    dependency_ratio = (count_children + count_elders) / (count_adults + 1) # Add 1 to avoid division by zero
    
    # Count of married: assuming Marital Status code 2 means currently married
    count_married = group[group['Marital Status (code)'] == 2].shape[0]
    
    # Maximum educational level attained (assuming higher code means higher education)
    max_education = group['Highest educational level attained (code)'].max()
    
    # Count of internet users: assuming '1' indicates "Yes"
    count_internet_users = group[group['Whether used internet from any location during last 30 days'] == 1].shape[0]
    
    # Total meals from various sources. Check if column exists before summing.
    total_meals_school = group['No. of meals taken during last 30 days from school, balwadi etc.'].sum() if 'No. of meals taken during last 30 days from school, balwadi etc.' in group.columns else np.nan
    total_meals_employer = group['No. of meals taken during last 30 days from employer as perquisites or part of wage'].sum() if 'No. of meals taken during last 30 days from employer as perquisites or part of wage' in group.columns else np.nan
    total_meals_payment = group['No. of meals taken during last 30 days on payment'].sum() if 'No. of meals taken during last 30 days on payment' in group.columns else np.nan
    total_meals_home = group['No. of meals taken during last 30 days at home'].sum() if 'No. of meals taken during last 30 days at home' in group.columns else np.nan
    total_meals_others = group['No. of meals taken during last 30 days  others'].sum() if 'No. of meals taken during last 30 days  others' in group.columns else np.nan
    
    return pd.Series({
        'count_adults': count_adults,
        'adults_mean_age': adults_mean_age,
        'count_children': count_children,
        'children_mean_age': children_mean_age,
        'count_elders': count_elders,
        'elders_mean_age': elders_mean_age,
        'count_males': count_males,
        'count_females': count_females,
        'gender_ratio': gender_ratio,
        'mean_age': mean_age,
        'dependency_ratio': dependency_ratio,
        'count_married': count_married,
        'max_education': max_education,
        'count_internet_users': count_internet_users,
        'total_meals_school': total_meals_school,
        'total_meals_employer': total_meals_employer,
        'total_meals_payment': total_meals_payment,
        'total_meals_home': total_meals_home,
        'total_meals_others': total_meals_others,
    })


household_data = pd.read_csv('HH_Test_Data.csv')
person_data = pd.read_csv('Person_Test_Data.csv')

# Get predictions
predictions = make_predictions(household_data, person_data)