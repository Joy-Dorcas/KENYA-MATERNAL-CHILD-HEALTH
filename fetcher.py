# DATA EXTRACTION FROM MULTIPLE SOURCES
import requests
import pandas as pd
import json
import time
import numpy as np
from pathlib import Path
import zipfile
import io
import geopandas as gpd

# ============================================================================
# SECTION 1: DHS API DATA EXTRACTION
# ============================================================================

def fetch_dhs_data():
    api_key = "YOUR_DHS_API_KEY"
    base_url = "https://api.dhsprogram.com/rest/dhs"
    
    surveys_response = requests.get(f"{base_url}/surveys", params={
        'countryIds': 'KE',
        'surveyType': 'DHS',
        'f': 'json'
    })
    
    surveys_data = surveys_response.json()
    
    latest_survey = None
    for survey in surveys_data.get('Data', []):
        if '2022' in survey.get('SurveyYear', ''):
            latest_survey = survey
            break
    
    survey_id = latest_survey['SurveyId']
    
    maternal_indicators = [
        'MM_ANEM_W_ANY',
        'RH_ANCN_W_4PL',
        'RH_DELA_C_SKP',
        'RH_DELP_C_HF'
    ]
    
    child_indicators = [
        'CH_VACC_C_DP3',
        'CH_VACC_C_MSL',
        'CH_DIAT_C_ORS',
        'FE_CMRT_C_IMR',
        'FE_CMRT_C_U5M'
    ]
    
    all_data = {}
    
    for indicator in maternal_indicators + child_indicators:
        response = requests.get(f"{base_url}/data", params={
            'surveyIds': survey_id,
            'indicatorIds': indicator,
            'breakdown': 'subnational',
            'f': 'json'
        })
        
        if response.status_code == 200:
            all_data[indicator] = response.json()
            time.sleep(0.5)
    
    return all_data

def process_dhs_data(raw_data):
    processed_data = []
    
    for indicator, data in raw_data.items():
        if 'Data' in data:
            for record in data['Data']:
                processed_data.append({
                    'indicator': indicator,
                    'county': record.get('CharacteristicLabel', ''),
                    'value': record.get('Value', 0),
                    'survey_year': record.get('SurveyYear', ''),
                    'denominator': record.get('DenominatorUnweighted', 0)
                })
    
    df = pd.DataFrame(processed_data)
    
    pivot_df = df.pivot_table(
        index='county',
        columns='indicator',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    return pivot_df

# ============================================================================
# SECTION 2: WORLD BANK API DATA EXTRACTION
# ============================================================================

def fetch_worldbank_data():
    indicators = {
        'SH.STA.MMRT': 'maternal_mortality_ratio',
        'SH.DYN.MORT': 'infant_mortality_rate',
        'SH.DYN.NMRT': 'neonatal_mortality_rate',
        'SP.POP.TOTL': 'population_total',
        'SI.POV.NAHC': 'poverty_headcount_ratio',
        'SE.ADT.LITR.FE.ZS': 'literacy_rate_female'
    }
    
    wb_data = {}
    
    for indicator_code, indicator_name in indicators.items():
        url = f"https://api.worldbank.org/v2/country/KE/indicator/{indicator_code}"
        
        response = requests.get(url, params={
            'format': 'json',
            'date': '2018:2022'
        })
        
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1 and data[1]:
                wb_data[indicator_name] = data[1]
        
        time.sleep(0.5)
    
    return wb_data

def process_worldbank_data(wb_data):
    processed_records = []
    
    for indicator_name, records in wb_data.items():
        for record in records:
            if record['value'] is not None:
                processed_records.append({
                    'indicator': indicator_name,
                    'year': record['date'],
                    'value': record['value'],
                    'country': record['country']['value']
                })
    
    df = pd.DataFrame(processed_records)
    
    pivot_df = df.pivot_table(
        index='year',
        columns='indicator',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    return pivot_df

# ============================================================================
# SECTION 3: HEALTH FACILITIES DATA EXTRACTION
# ============================================================================

def fetch_health_facilities():
    url = "https://api.healthsites.io/api/v2/facilities/"
    
    facilities_data = []
    page = 1
    
    while True:
        response = requests.get(url, params={
            'country': 'Kenya',
            'format': 'json',
            'page': page
        })
        
        if response.status_code != 200:
            break
        
        data = response.json()
        features = data.get('features', [])
        
        if not features:
            break
        
        for facility in features:
            props = facility.get('properties', {})
            geom = facility.get('geometry', {})
            
            if geom.get('coordinates'):
                facilities_data.append({
                    'name': props.get('name', ''),
                    'amenity': props.get('amenity', ''),
                    'latitude': geom['coordinates'][1],
                    'longitude': geom['coordinates'][0],
                    'operator': props.get('operator', ''),
                    'source': props.get('source', '')
                })
        
        page += 1
        if page > 50:
            break
        
        time.sleep(1)
    
    return pd.DataFrame(facilities_data)

# ============================================================================
# SECTION 4: SPATIAL DATA EXTRACTION
# ============================================================================

def fetch_spatial_data():
    gadm_url = "https://biogeo.ucdavis.edu/data/gadm3.6/shp/gadm36_KEN_shp.zip"
    
    response = requests.get(gadm_url)
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        zip_file.extractall("spatial_data/")
    
    county_gdf = gpd.read_file("spatial_data/gadm36_KEN_1.shp")
    
    return county_gdf

def process_spatial_data(gdf):
    counties_info = []
    
    for _, row in gdf.iterrows():
        counties_info.append({
            'county_name': row['NAME_1'],
            'admin_code': row['HASC_1'],
            'area_sqkm': row.geometry.area,
            'centroid_lat': row.geometry.centroid.y,
            'centroid_lon': row.geometry.centroid.x
        })
    
    return pd.DataFrame(counties_info)

# ============================================================================
# SECTION 5: KENYA OPEN DATA PORTAL EXTRACTION
# ============================================================================

def fetch_kenya_opendata():
    base_url = "https://www.opendata.go.ke/api/views"
    
    datasets = {
        'health_indicators': 'mw8q-q5zh',
        'population_census': 'e6jx-k8h4',
        'poverty_statistics': 'ckx3-3t5j'
    }
    
    all_datasets = {}
    
    for dataset_name, dataset_id in datasets.items():
        url = f"{base_url}/{dataset_id}/rows.json"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            all_datasets[dataset_name] = data
        
        time.sleep(1)
    
    return all_datasets

def process_opendata(raw_data):
    processed_datasets = {}
    
    for dataset_name, data in raw_data.items():
        if 'data' in data:
            df = pd.DataFrame(data['data'], columns=[col['name'] for col in data['meta']['view']['columns']])
            processed_datasets[dataset_name] = df
    
    return processed_datasets

# ============================================================================
# SECTION 6: UNICEF DATA EXTRACTION
# ============================================================================

def fetch_unicef_data():
    url = "https://sdmx.data.unicef.org/ws/public/sdmxapi/rest/data"
    
    indicators = [
        'CME_MRY0T4',
        'CME_MRY0',
        'MNCH_ANC4',
        'MNCH_DELIVCARE',
        'MNCH_INSTDEL'
    ]
    
    unicef_data = {}
    
    for indicator in indicators:
        response = requests.get(f"{url}/UNICEF,{indicator}/KEN", headers={
            'Accept': 'application/json'
        })
        
        if response.status_code == 200:
            unicef_data[indicator] = response.json()
        
        time.sleep(2)
    
    return unicef_data

# ============================================================================
# SECTION 7: CONFLICT DATA (ACLED) EXTRACTION
# ============================================================================

def fetch_acled_data():
    url = "https://api.acleddata.com/acled/read"
    
    response = requests.get(url, params={
        'iso': 'KEN',
        'year': '2018|2019|2020|2021|2022',
        'limit': 5000
    })
    
    if response.status_code == 200:
        return response.json()
    
    return None

def process_acled_data(acled_data):
    if not acled_data or 'data' not in acled_data:
        return pd.DataFrame()
    
    conflicts = []
    
    for event in acled_data['data']:
        conflicts.append({
            'event_date': event.get('event_date'),
            'admin1': event.get('admin1'),
            'admin2': event.get('admin2'),
            'event_type': event.get('event_type'),
            'fatalities': int(event.get('fatalities', 0)),
            'latitude': float(event.get('latitude', 0)),
            'longitude': float(event.get('longitude', 0))
        })
    
    return pd.DataFrame(conflicts)

# ============================================================================
# SECTION 8: POPULATION DATA (WORLDPOP) EXTRACTION
# ============================================================================

def fetch_worldpop_metadata():
    url = "https://www.worldpop.org/rest/data/age_structures/KEN"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    
    return None

# ============================================================================
# SECTION 9: DATA INTEGRATION AND HARMONIZATION
# ============================================================================

def integrate_all_data():
    dhs_raw = fetch_dhs_data()
    dhs_processed = process_dhs_data(dhs_raw)
    
    wb_raw = fetch_worldbank_data()
    wb_processed = process_worldbank_data(wb_raw)
    
    facilities_df = fetch_health_facilities()
    
    spatial_gdf = fetch_spatial_data()
    spatial_df = process_spatial_data(spatial_gdf)
    
    kenya_opendata = fetch_kenya_opendata()
    opendata_processed = process_opendata(kenya_opendata)
    
    unicef_data = fetch_unicef_data()
    
    acled_raw = fetch_acled_data()
    acled_processed = process_acled_data(acled_raw)
    
    return {
        'dhs': dhs_processed,
        'worldbank': wb_processed,
        'facilities': facilities_df,
        'spatial': spatial_df,
        'kenya_opendata': opendata_processed,
        'unicef': unicef_data,
        'conflict': acled_processed
    }

def create_master_dataset(integrated_data):
    counties = [
        'Nairobi', 'Mombasa', 'Kwale', 'Kilifi', 'Tana River', 'Lamu', 'Taita Taveta',
        'Garissa', 'Wajir', 'Mandera', 'Marsabit', 'Isiolo', 'Meru', 'Tharaka Nithi',
        'Embu', 'Kitui', 'Machakos', 'Makueni', 'Nyandarua', 'Nyeri', 'Kirinyaga',
        'Murang\'a', 'Kiambu', 'Turkana', 'West Pokot', 'Samburu', 'Trans Nzoia',
        'Uasin Gishu', 'Elgeyo Marakwet', 'Nandi', 'Baringo', 'Laikipia', 'Nakuru',
        'Narok', 'Kajiado', 'Kericho', 'Bomet', 'Kakamega', 'Vihiga', 'Bungoma',
        'Busia', 'Siaya', 'Kisumu', 'Homa Bay', 'Migori', 'Kisii', 'Nyamira'
    ]
    
    master_data = []
    
    for county in counties:
        county_record = {'county': county}
        
        if 'dhs' in integrated_data and integrated_data['dhs'] is not None:
            dhs_county = integrated_data['dhs'][integrated_data['dhs']['county'] == county]
            if not dhs_county.empty:
                county_record.update(dhs_county.iloc[0].to_dict())
        
        if 'spatial' in integrated_data:
            spatial_county = integrated_data['spatial'][integrated_data['spatial']['county_name'] == county]
            if not spatial_county.empty:
                county_record.update(spatial_county.iloc[0].to_dict())
        
        if 'facilities' in integrated_data:
            facilities_count = len(integrated_data['facilities'])
            county_record['facilities_count'] = facilities_count // 47
        
        if 'conflict' in integrated_data and not integrated_data['conflict'].empty:
            county_conflicts = integrated_data['conflict'][integrated_data['conflict']['admin1'] == county]
            county_record['conflict_events'] = len(county_conflicts)
            county_record['conflict_fatalities'] = county_conflicts['fatalities'].sum() if not county_conflicts.empty else 0
        
        master_data.append(county_record)
    
    return pd.DataFrame(master_data)

def save_all_data(integrated_data, master_df):
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    for source_name, data in integrated_data.items():
        if isinstance(data, pd.DataFrame):
            data.to_csv(f"data/raw/{source_name}_data.csv", index=False)
        elif isinstance(data, dict):
            with open(f"data/raw/{source_name}_data.json", 'w') as f:
                json.dump(data, f, indent=2)
    
    master_df.to_csv("data/processed/master_dataset.csv", index=False)
    
    return True

# ============================================================================
# SECTION 10: MAIN EXECUTION
# ============================================================================

def main():
    integrated_data = integrate_all_data()
    master_dataset = create_master_dataset(integrated_data)
    save_all_data(integrated_data, master_dataset)
    
    return master_dataset

if __name__ == "__main__":
    final_dataset = main()
    print(f"Data extraction complete. Master dataset shape: {final_dataset.shape}")
