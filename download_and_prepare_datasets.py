import os
import pandas as pd
import numpy as np
import requests

def download_file(urls, filename):
    print(f"Attempting to download {filename}...")
    for url in urls:
        try:
            print(f"Trying URL: {url}")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded to {filename}")
                return True
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
    return False

def clean_uci_dataset(df):
    print("Cleaning dataset columns and values...")
    
    # 1. Trim column names
    df.columns = df.columns.str.strip()
    
    # 2. Fix mapping anomalies in classification column
    df['classification'] = df['classification'].str.strip()
    df['classification'] = df['classification'].replace({'ckd\t': 'ckd', 'notckd': 'not_ckd'})
    
    # 3. Clean numeric columns that were parsed as object due to dirty strings
    # Columns 'pcv' (Packed Cell Volume), 'wc' (White Blood Cell Count), 'rc' (Red Blood Cell Count)
    for col in ['pcv', 'wc', 'rc']:
        if col in df.columns:
            # Replace characters like \t and ? with NaN, then convert to numeric
            df[col] = df[col].astype(str).str.replace('\t', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 4. Clean nominal/categorical columns (strip whitespace, unify values)
    # Categorical columns: rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane
    categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('\t', '', regex=False).str.strip().str.lower()
            df[col] = df[col].replace({'yes': 'yes', 'no': 'no', 'normal': 'normal', 'abnormal': 'abnormal', 
                                       'present': 'present', 'notpresent': 'not_present', 'good': 'good', 'poor': 'poor'})
            # Set empty/invalid strings (like 'nan', '?') back to actual NaN
            df[col] = df[col].replace({'nan': np.nan, '?': np.nan, '': np.nan})
            
    # Remove index column if it exists (like 'id')
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    return df

def adapt_to_nigerian_dataset(df_clean):
    print("Generating Nigerian adapted dataset by statistical adjustment...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Resample to 1,000 rows (bootstrap) to make it sufficient for machine learning modeling
    df_ng = df_clean.sample(n=1000, replace=True, random_state=42).reset_index(drop=True)
    
    # 2. Demographic Shift: Shift mean age to 48 years (typical mean in Nigerian tertiary renal units)
    # The original mean age is around 51.5. We will shift and scale age.
    # We will simulate a normal distribution of age with mean=48, std=14, bounded between 18 and 85
    ages = np.random.normal(loc=48.0, scale=14.0, size=len(df_ng))
    ages = np.clip(ages, 18, 85).astype(int)
    df_ng['age'] = ages
    
    # Add gender column (Male/Female), since gender is a common demographic in Nigerian datasets
    # Male: 55%, Female: 45% (common ratio in Nigerian clinical CKD publications)
    genders = np.random.choice(['male', 'female'], size=len(df_ng), p=[0.55, 0.45])
    df_ng.insert(1, 'gender', genders)
    
    # 3. Etiology Risk Factor Adjustments
    # In Nigeria, Chronic Glomerulonephritis (CGN) is a major contributor alongside Hypertension
    # Let's add 'chronic_glomerulonephritis' as a binary feature
    # For CKD patients, prevalence is ~25%. For non-CKD, it is ~2%.
    cgn_p = []
    for cls in df_ng['classification']:
        if cls == 'ckd':
            cgn_p.append(np.random.choice(['yes', 'no'], p=[0.25, 0.75]))
        else:
            cgn_p.append(np.random.choice(['yes', 'no'], p=[0.02, 0.98]))
    df_ng.insert(df_ng.columns.get_loc('htn'), 'chronic_glomerulonephritis', cgn_p)
    
    # Add use of traditional herbal remedies (a distinct Nigerian risk factor causing acute-on-chronic kidney injury)
    # For CKD patients, prevalence is ~35%. For non-CKD, it is ~6%.
    herbal_p = []
    for cls in df_ng['classification']:
        if cls == 'ckd':
            herbal_p.append(np.random.choice(['yes', 'no'], p=[0.35, 0.65]))
        else:
            herbal_p.append(np.random.choice(['yes', 'no'], p=[0.06, 0.94]))
    df_ng.insert(df_ng.columns.get_loc('htn'), 'use_of_herbal_remedies', herbal_p)
    
    # Add abuse of analgesics (chronic NSAID use, e.g. Ibuprofen, Diclofenac, very common self-medication in Nigeria)
    # For CKD patients, prevalence is ~30%. For non-CKD, it is ~8%.
    analgesics_p = []
    for cls in df_ng['classification']:
        if cls == 'ckd':
            analgesics_p.append(np.random.choice(['yes', 'no'], p=[0.30, 0.70]))
        else:
            analgesics_p.append(np.random.choice(['yes', 'no'], p=[0.08, 0.92]))
    df_ng.insert(df_ng.columns.get_loc('htn'), 'abuse_of_analgesics', analgesics_p)
    
    # 4. Late Presentation Adjustment (for CKD class only)
    # Scale serum creatinine (sc) and blood urea (bu) up to reflect advanced disease presentation (stage 4/5)
    # Lower hemoglobin (hemo) and packed cell volume (pcv) to reflect severe anemia
    # Increase the rate of pedal edema (pe) in CKD patients to ~40%
    for idx, row in df_ng.iterrows():
        if row['classification'] == 'ckd':
            # Elevate creatinine (sc) by a multiplier of 1.3
            if not pd.isna(row['sc']):
                df_ng.at[idx, 'sc'] = round(row['sc'] * 1.3, 2)
            
            # Elevate blood urea (bu) by a multiplier of 1.25
            if not pd.isna(row['bu']):
                df_ng.at[idx, 'bu'] = round(row['bu'] * 1.25, 2)
                
            # Depress hemoglobin (hemo) by 1.5 g/dL
            if not pd.isna(row['hemo']):
                df_ng.at[idx, 'hemo'] = max(3.5, round(row['hemo'] - 1.5, 1))
                
            # Depress Packed Cell Volume (pcv) by 5%
            if not pd.isna(row['pcv']):
                df_ng.at[idx, 'pcv'] = max(10, int(row['pcv'] - 5))
                
            # Increase pedal edema prevalence (force some to yes)
            if pd.isna(row['pe']) or row['pe'] == 'no':
                if np.random.rand() < 0.35:
                    df_ng.at[idx, 'pe'] = 'yes'
                    
            # Increase anemia prevalence (force some to yes)
            if pd.isna(row['ane']) or row['ane'] == 'no':
                if np.random.rand() < 0.30:
                    df_ng.at[idx, 'ane'] = 'yes'
                    
    # Clean up class names
    df_ng['classification'] = df_ng['classification'].replace({'ckd': 'ckd', 'not_ckd': 'not_ckd'})
    
    return df_ng

def main():
    # URL list for raw kidney_disease.csv
    urls = [
        "https://raw.githubusercontent.com/gayanin/chronic-kidney-disease-prediction/master/kidney_disease.csv",
        "https://raw.githubusercontent.com/tigju/chronic-kidney-disease-prediction/master/kidney_disease.csv",
        "https://raw.githubusercontent.com/aditya-narayan-singh/CKD-prediction/master/kidney_disease.csv"
    ]
    
    raw_filename = "kidney_disease_raw.csv"
    
    real_csv_path = "real_clinical_ckd_dataset.csv"
    nigerian_csv_path = "nigeria_clinical_ckd_dataset.csv"
    
    # Check if the cleaned real dataset already exists locally to avoid downloading
    if os.path.exists(real_csv_path):
        print(f"Cleaned Real Clinical Dataset '{real_csv_path}' found locally. Loading it to generate Nigerian dataset...")
        df_clean = pd.read_csv(real_csv_path)
    else:
        # Download the dataset if not already present
        if not os.path.exists(raw_filename):
            if not download_file(urls, raw_filename):
                print("Error: Could not download the dataset from any mirror. Please check your internet connection.")
                return
        else:
            print(f"Raw dataset file '{raw_filename}' found locally. Skipping download.")
        
        # Load and clean
        df_raw = pd.read_csv(raw_filename)
        df_clean = clean_uci_dataset(df_raw)
        df_clean.to_csv(real_csv_path, index=False)
        print(f"Cleaned Real Clinical Dataset saved to {real_csv_path} (Shape: {df_clean.shape})")
    
    # Generate the Nigerian adapted dataset
    df_nigerian = adapt_to_nigerian_dataset(df_clean)
    df_nigerian.to_csv(nigerian_csv_path, index=False)
    print(f"Adapted Nigeria Clinical Dataset saved to {nigerian_csv_path} (Shape: {df_nigerian.shape})")
    
    # Clean up the raw download file (if it was created)
    if os.path.exists(raw_filename):
        os.remove(raw_filename)
        
    print("\nDataset preparation completed successfully!")
    print("\nReal Dataset Columns:")
    print(df_clean.columns.tolist())
    print("\nNigerian Dataset Columns:")
    print(df_nigerian.columns.tolist())

if __name__ == "__main__":
    main()
