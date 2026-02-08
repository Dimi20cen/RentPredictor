import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Set Page Config
st.set_page_config(page_title="Swiss Rent Predictor", layout="centered")

# --- 1. Load Resources ---
@st.cache_resource
def load_resources():
    # Load Model
    with open('models/xgb_rent_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load Zip Encoder
    with open('models/zip_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    # Load feature columns used during training
    with open('models/feature_columns.json', 'r', encoding='utf-8') as f:
        feature_columns = json.load(f)
        
    # Load Cleaned Data (for dropdown options only)
    # We only need unique Zips, Cantons, and Tax info
    df = pd.read_pickle('data/processed/02_featured_data.pkl')
    
    return model, encoder, feature_columns, df

model, encoder_zip, model_feature_names, df_ref = load_resources()

# --- 2. Helper Functions (Recreating NB 02 Logic) ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

HUBS = {
    'Zurich_HB': (47.378177, 8.540192),
    'Geneva_Cornavin': (46.210226, 6.142456),
    'Basel_SBB': (47.547412, 7.589556),
    'Bern_HB': (46.948833, 7.439122),
    'Lausanne_Gare': (46.516777, 6.629095)
}

# --- 3. UI Layout ---
st.title("🇨🇭 Swiss Rental Price Predictor")
st.markdown("Estimate the fair market value of an apartment in Switzerland using Machine Learning (XGBoost).")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Living Area (m²)", min_value=10, max_value=500, value=65)
    rooms = st.number_input("Rooms", min_value=1.0, max_value=10.0, step=0.5, value=2.5)
    floor = st.number_input("Floor", min_value=-1, max_value=20, value=2)

with col2:
    # Canton Selection
    cantons = sorted(df_ref['Canton'].unique())
    selected_canton = st.selectbox("Canton", cantons, index=cantons.index('ZH') if 'ZH' in cantons else 0)
    
    # Filter Zips by Canton
    canton_zips = sorted(df_ref[df_ref['Canton'] == selected_canton]['Zip'].unique())
    selected_zip = st.selectbox("Zip Code", canton_zips)
    
    # SubType
    subtypes = sorted(df_ref['SubType'].unique())
    selected_type = st.selectbox("Property Type", subtypes, index=subtypes.index('FLAT') if 'FLAT' in subtypes else 0)

st.markdown("### ✨ Extras")
c1, c2, c3 = st.columns(3)
with c1: has_lake = st.checkbox("Lake View")
with c2: is_new = st.checkbox("New Building")
with c3: is_quiet = st.checkbox("Quiet Area")

# --- 4. Prediction Logic ---
if st.button("Predict Rent", type="primary"):
    
    # A. Get Reference Data for the selected Zip
    # We need Lat, Lon, and Tax Rate for the selected Zip
    # We take the median Lat/Lon/Tax of existing listings in that Zip
    ref_row = df_ref[df_ref['Zip'] == selected_zip].iloc[0]
    lat, lon = ref_row['Lat'], ref_row['Lon']
    tax_rate = ref_row['tax_rate']
    
    # B. Calculate Distances
    dists = {}
    for hub, coords in HUBS.items():
        dists[f'dist_to_{hub}'] = haversine_distance(lat, lon, coords[0], coords[1])
    min_dist = min(dists.values())
    
    # C. Construct Input DataFrame
    # Must match the columns XGBoost expects (except One-Hot columns which we handle manually)
    input_data = {
        'Rooms': [rooms],
        'Area_m2': [area],
        'Floor': [floor],
        'Lat': [lat],
        'Lon': [lon],
        'dist_to_Zurich_HB': [dists['dist_to_Zurich_HB']],
        'dist_to_Geneva_Cornavin': [dists['dist_to_Geneva_Cornavin']],
        'dist_to_Basel_SBB': [dists['dist_to_Basel_SBB']],
        'dist_to_Bern_HB': [dists['dist_to_Bern_HB']],
        'dist_to_Lausanne_Gare': [dists['dist_to_Lausanne_Gare']],
        'dist_to_closest_hub': [min_dist],
        'tax_rate': [tax_rate],
        'is_rent_estimated': [0], # User input is "real"
        'year_built_is_missing': [1], # Assume unknown
        'is_renovated': [1 if is_new else 0],
        'Balcony': [1 if has_lake else 0], # Proxy
        'Elevator': [1],
        'Parking': [0],
        'View': [1 if has_lake else 0],
        'Fireplace': [0],
        'Child_Friendly': [0],
        'CableTV': [0],
        'New_Building': [1 if is_new else 0],
        'Minergie': [0],
        'Wheelchair': [0],
        'has_lake_view': [1 if has_lake else 0],
        'is_attic': [0],
        'is_quiet': [1 if is_quiet else 0],
        'is_sunny': [0],
        'Zip': [selected_zip] # For Encoder
    }
    
    X_input = pd.DataFrame(input_data)
    
    # D. Encoding
    # 1. Target Encode Zip
    X_input['Zip_encoded'] = encoder_zip.transform(X_input['Zip'])
    X_input = X_input.drop(columns=['Zip'])
    
    # 2. One-Hot Encode (Manual to match training columns)
    # We need to create the exact columns the model saw during training
    # e.g., 'Canton_ZH', 'SubType_LOFT'
    # Some XGBoost pickles do not preserve booster feature names.
    # We load canonical feature names from models/feature_columns.json.
    
    # Create empty df with all model columns initialized to 0
    X_final = pd.DataFrame(0, index=[0], columns=model_feature_names, dtype=float)
    
    # Fill in the numeric data we have
    for col in X_input.columns:
        if col in X_final.columns:
            X_final[col] = X_input[col]
            
    # Set the One-Hot columns
    canton_col = f"Canton_{selected_canton}"
    if canton_col in X_final.columns:
        X_final[canton_col] = 1
        
    subtype_col = f"SubType_{selected_type}"
    if subtype_col in X_final.columns:
        X_final[subtype_col] = 1
        
    # E. Predict
    prediction = model.predict(X_final)[0]
    
    # F. Display
    st.success(f"### Estimated Rent: CHF {int(prediction):,}")
    st.info(f"📍 Location: {selected_zip} (Tax Index: {tax_rate}) | 📏 Distance to Hub: {int(min_dist)} km")
