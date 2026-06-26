import streamlit as st
import pandas as pd
# import pickle
import joblib
# -----------------------------
# Load Trained Model
# -----------------------------
import os
# import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "Model", "best_model.pkl")

model = joblib.load(MODEL_PATH)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction System")
st.write("Enter the house details below to predict the selling price.")

# -----------------------------
# User Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality", 1, 10, 5)
    gr_liv_area = st.number_input("Ground Living Area (sq ft)", 500, 6000, 1500)
    garage_cars = st.slider("Garage Capacity", 0, 5, 2)
    garage_area = st.number_input("Garage Area", 0, 1500, 500)
    total_bsmt_sf = st.number_input("Basement Area", 0, 4000, 900)

with col2:
    year_built = st.number_input("Year Built", 1872, 2025, 2000)
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)
    bedroom = st.slider("Bedrooms", 1, 8, 3)
    lot_area = st.number_input("Lot Area", 1000, 50000, 9000)
    first_floor = st.number_input("1st Floor Area", 300, 4000, 1200)

# -----------------------------
# Default Values for Remaining Features
# -----------------------------
sample = {
    'Id':1,
    'MSSubClass':20,
    'MSZoning':'RL',
    'LotFrontage':70,
    'LotArea':lot_area,
    'Street':'Pave',
    'Alley':'NA',
    'LotShape':'Reg',
    'LandContour':'Lvl',
    'Utilities':'AllPub',
    'LotConfig':'Inside',
    'LandSlope':'Gtl',
    'Neighborhood':'NAmes',
    'Condition1':'Norm',
    'Condition2':'Norm',
    'BldgType':'1Fam',
    'HouseStyle':'1Story',
    'OverallQual':overall_qual,
    'OverallCond':5,
    'YearBuilt':year_built,
    'YearRemodAdd':year_built,
    'RoofStyle':'Gable',
    'RoofMatl':'CompShg',
    'Exterior1st':'VinylSd',
    'Exterior2nd':'VinylSd',
    'MasVnrType':'None',
    'MasVnrArea':0,
    'ExterQual':'TA',
    'ExterCond':'TA',
    'Foundation':'PConc',
    'BsmtQual':'TA',
    'BsmtCond':'TA',
    'BsmtExposure':'No',
    'BsmtFinType1':'Unf',
    'BsmtFinSF1':0,
    'BsmtFinType2':'Unf',
    'BsmtFinSF2':0,
    'BsmtUnfSF':total_bsmt_sf,
    'TotalBsmtSF':total_bsmt_sf,
    'Heating':'GasA',
    'HeatingQC':'Ex',
    'CentralAir':'Y',
    'Electrical':'SBrkr',
    '1stFlrSF':first_floor,
    '2ndFlrSF':0,
    'LowQualFinSF':0,
    'GrLivArea':gr_liv_area,
    'BsmtFullBath':1,
    'BsmtHalfBath':0,
    'FullBath':full_bath,
    'HalfBath':0,
    'BedroomAbvGr':bedroom,
    'KitchenAbvGr':1,
    'KitchenQual':'TA',
    'TotRmsAbvGrd':6,
    'Functional':'Typ',
    'Fireplaces':1,
    'FireplaceQu':'Gd',
    'GarageType':'Attchd',
    'GarageYrBlt':year_built,
    'GarageFinish':'RFn',
    'GarageCars':garage_cars,
    'GarageArea':garage_area,
    'GarageQual':'TA',
    'GarageCond':'TA',
    'PavedDrive':'Y',
    'WoodDeckSF':0,
    'OpenPorchSF':50,
    'EnclosedPorch':0,
    '3SsnPorch':0,
    'ScreenPorch':0,
    'PoolArea':0,
    'PoolQC':'NA',
    'Fence':'NA',
    'MiscFeature':'NA',
    'MiscVal':0,
    'MoSold':6,
    'YrSold':2010,
    'SaleType':'WD',
    'SaleCondition':'Normal'
}

input_df = pd.DataFrame([sample])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict House Price"):

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted House Price: ${prediction:,.2f} USD")