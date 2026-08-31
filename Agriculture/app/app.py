import streamlit as st
import pandas as pd
import numpy as np
import joblib 
page_bg ="""
<style>
[data-testid= "stAppViewContainer"]{
    background: linear-gradient(180deg, #2E7D32 0%, #43A047 50%, #66BB6A 100%);}
    [data-testid="stHeader"]{
        background-color: rgba(0,0,0,0);
        }
        h1, h2, h3,label,p{
            color:white
        }
        </style>
"""
st.markdown(page_bg, unsafe_allow_html=True)
st.set_page_config(page_title= 'AgriInsight Kenya', page_icon='🌾',layout='wide')
@st.cache_resource
def load_artifacts():
    return(joblib.load('models/agri_model.pkl'),
           joblib.load('models/agri_scaler.pkl'),
           joblib.load('models/feature_names.pkl'))
model,scaler, feature_names = load_artifacts()
st.title('🌾 Agri Insights Kenya - Crop Yield Predictor')
st.markdown('Predict crop yield (tonnes/hectare) by county, season and inputs.')
st.divider()
col1, col2, col3 = st.columns(3)
COUNTIES =[   'Uasin Gishu',
    'Trans Nzoia',
    'Nakuru',
    'Nandi',
    'Bungoma', 
    'Kakamega',
    'Kisii', 
    'Nyamira',
    'Vihiga', 
    'Bomet',
    'Kericho',
    'Nyeri', 
    'Muranga',
    'Kiambu', 
    'Kirinyaga',
    'Embu', 
    'Meru',
    'Tharaka-Nithi', 
    'Nyandarua',
    'Nairobi',
    'Machakos',
    'Makueni',
    'Kitui',
    'Mombasa',
    'Kilifi',
    'Kwale',
    'Tana River',
    'Lamu',
    'Taita Taveta',
    'Garissa',
    'Wajir',
    'Mandera',
    'Turkana',
    'West Pokot',
    'Samburu',
    'Baringo',
    'Laikipia',
    'Kisumu',
    'Siaya',
    'Homa Bay',
    'Migori',
    'Kajiado',
    'Narok',
    'Elgeyo-Marakwet',
    'Busia',
    'Isiolo',
     'Marsabit']
COUNTY_AVG = {'Uasin Gishu' :3.528,
    'Trans Nzoia' :4.069,
    'Nakuru' :3.973,
    'Nandi' :2.798,
    'Bungoma' :2.502, 
    'Kakamega' :2.346,
    'Kisii' :2.552, 
    'Nyamira' :2.398,
    'Vihiga' :1.579, 
    'Bomet' :3.983,
    'Kericho' :2.773,
    'Nyeri' :1.242, 
    'Muranga' :1.371,
    'Kiambu' :0.987, 
    'Kirinyaga' :1.215,
    'Embu' :1.367, 
    'Meru' :1.854,
    'Tharaka-Nithi' :1.881, 
    'Nyandarua' :3.861,
    'Nairobi' :0.496,
    'Machakos' :0.89,
    'Makueni' :0.48,
    'Kitui' : 0.455,
    'Mombasa' :0.266,
    'Kilifi' :0.661,
    'Kwale' :0.678,
    'Tana River' :0.341 ,
    'Lamu' :0.339,
    'Taita Taveta' :0.628,
    'Garissa' :0.251,
    'Wajir' :0.239,
    'Mandera' :0.215,
    'Turkana' :0.216,
    'West Pokot' :1.081,
    'Samburu' :0.34,
    'Baringo' :0.617,
    'Laikipia' :1.385,
    'Kisumu' :1.472,
    'Siaya' :1.511,
    'Homa Bay' :1.053,
    'Migori' :1.369,
    'Kajiado' :0.583,
    'Narok' :1.391,
    'Elgeyo-Marakwet' :1.444,
    'Busia' :1.602,
    'Isiolo':0.243,
     'Marsabit' :0.212
                 }
with col1:
    st.subheader('Location and Season')
    county =st.selectbox('county', COUNTIES)
    season = st.selectbox('Season', ['Long Rains', 'Short Rains'])
    year = st.slider('Year', 2020,2030,2024)
    crop = st.selectbox('Crop', ['Maize', 'Beans', 'Wheat','Sorghum','Potatoes'])
with col2:
    st.subheader('Climate Conditions')
    rainfall =st.slider('Seasonal Rainfall (mm)', 50, 1800, 750)
    temp = st.slider('Avg Temperature (°C)', 10.0, 38.0,20.0,0.5)
with col3:
    st.subheader('Farm Inputs')
    area = st.number_input('Area (hectares)', 0.5, 5000.0, 2.0,0.5)
    fertiliser = st.slider('Fertiliser (kg/ha)', 0,150, 30)
    seed = st.selectbox('Seed Variety',['Hybrid', 'Improved Traditional','Traditional'])

if st.button('Predict Yield', type='primary' ,use_container_width=True):
    crop_yield_mult ={'Maize' :1.0, 'Beans': 0.4,'Wheat':1.2,'Sorghum':0.8,'Potatoes':8.5}
    seed_mult = {'Hybrid': 1.0, 'Improved Traditional':0.0, 'Traditional':0.0}
    row = {f: 0 for f in feature_names}
    row.update({
        'year' :year,
        'rainfall_mm' :rainfall,
        'avg_temp_c' :temp,
        'fertiliser_kg_ha': fertiliser,
        'log_area' : np.log1p(area),
        'season_enc': 0 if season =='Long Rains' else 1,
        'temp_stress': int(temp > 30 or temp < 12),
        'input_intensity': fertiliser / area,
        'county_avg_yield': COUNTY_AVG.get(county, 1.8),
        f'crop_{crop}': 1 if f'crop_{crop}' in feature_names else 0,
        f'seed_variety_Hybrid': 1 if seed == 'Hybrid' else 0,
   }) 
    X = pd.DataFrame([row])[feature_names]
    pred = model.predict(scaler.transform(X))[0]
    pred = max (0.1, pred)
    total = pred * area
    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Predict Yield', f'{pred:.2f} t/ha')
    c2.metric('Total Production', f'{total:.1f} tonnes')
    c3.metric('vs National Average', f'{pred - 1.84:.2f}t/ha')
    rating = 'Excellent' if pred > 3 else 'Good' if pred > 2 else 'Average' if pred > 1.2 else 'Poor'
    c4.metric('Season Rating', rating)
    if fertiliser < 20:
        st.info('Low fertiliser use detected. Increase to 30-50 kg/ha could '
               'improve yields by 15-25% based on historical data.')
    if seed == 'Traditional':
        st.info('Switching to certified hybrid seed could increase yields by 30-40%.')

  #Revenue estimate
    maize_price_kes = 4500
    revenue = total * maize_price_kes
    st.success(f'Estimated Revenue: KES{revenue:,.0f} (at KES{maize_price_kes}/tonne)')
       