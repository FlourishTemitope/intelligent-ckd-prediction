import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Intelligent CKD Prediction System",
    layout="wide"
)

# Custom balanced clinical styling (Deep Teal, slate blue, soft green/red indicators)
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* Headings and Titles */
    h1 {
        color: #004D40; /* Clinical Deep Teal */
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    h3 {
        color: #00796B;
        margin-top: 15px;
        border-bottom: 2px solid #E0F2F1;
        padding-bottom: 5px;
    }
    
    /* Predict Button Customization */
    div.stButton > button:first-child {
        background-color: #00796B;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0, 121, 107, 0.15);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #004D40;
        box-shadow: 0 6px 12px rgba(0, 77, 64, 0.25);
    }
    
    /* Info/Warning custom container cards */
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #00796B;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    
    .healthy-card {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #2E7D32;
        color: #1B5E20;
        font-weight: 500;
    }
    
    .ckd-card {
        background-color: #FFEBEE;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #C62828;
        color: #B71C1C;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model pipelines
@st.cache_resource
def load_model(model_name, dataset_type):
    model_key = model_name.lower().replace(" ", "_")
    dataset_key = dataset_type.lower().replace(" ", "_")
    filename = f"{model_key}_{dataset_key}.joblib"
    if os.path.exists(filename):
        return joblib.load(filename)
    return None

def main():
    # Header Section
    st.markdown("<h1>Intelligent Chronic Kidney Disease Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #555; font-size: 15px; margin-top:-5px;'>Clinical Decision Support Tool for Screening & Early Diagnostic Prediction of CKD</p>", unsafe_allow_html=True)
    
    st.divider()

    # Column Layout for Settings
    col1, col2 = st.columns(2)
    with col1:
        dataset_type = st.selectbox(
            "Select Diagnostic Target Dataset",
            options=["Nigerian Clinical", "UCI Clinical"],
            help="Choose the model context: 'Nigerian Clinical' is statistically adjusted for local demographics, risk factors, and late-stage uremic presentation in sub-Saharan Africa. 'UCI Clinical' represents the global baseline dataset."
        )
    with col2:
        model_name = st.selectbox(
            "Select Supervised Classifier",
            options=["Random Forest", "Logistic Regression", "Support Vector Machine"],
            index=0,  # Default to Random Forest as requested
            help="Select the machine learning algorithm to run the prediction. Random Forest provides the highest overall accuracy."
        )
        
    # Load selected model pipeline
    pipeline = load_model(model_name, dataset_type)
    
    if pipeline is None:
        st.error(f"Error: Model file `{model_name.lower().replace(' ', '_')}_{dataset_type.lower().replace(' ', '_')}.joblib` was not found. Please train the models first.")
        return

    st.markdown("### Patient Vitals & Clinical Laboratory Input Panel")
    st.markdown("<p style='color: #666; font-size: 13px;'>Fill in the patient's demographics, clinical history, and laboratory test markers below. Tooltips (❓) are provided to explain medical parameters and typical ranges.</p>", unsafe_allow_html=True)
    
    # Organize inputs into tabs or columns
    tab_demo, tab_urinalysis, tab_blood, tab_history = st.tabs([
        "1. Demographics & Vitals", 
        "2. Urinalysis Parameters", 
        "3. Blood Chemistry & Lab Markers", 
        "4. Patient Medical History"
    ])
    
    inputs = {}
    
    with tab_demo:
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs['age'] = st.number_input(
                "Patient Age (Years)", 
                min_value=1, max_value=120, value=48, step=1,
                help="Age of the patient in years. If you do not know the patient's exact age, you can leave it at the default value of 48 years (which is the typical age of presentation in chronic kidney disease clinics)."
            )
        with c2:
            # Dynamic input: Gender is only used in the Nigerian dataset model
            if dataset_type == "Nigerian Clinical":
                gender = st.selectbox(
                    "Gender", 
                    options=["Male", "Female"],
                    help="Biological sex of the patient. If you do not know it, leave it at the default 'Male' (which represents approximately 55% of admissions in documented renal clinics)."
                ).lower()
                inputs['gender'] = gender
            else:
                gender = None
        with c3:
            inputs['bp'] = st.number_input(
                "Diastolic Blood Pressure (mm/Hg)", 
                min_value=50, max_value=220, value=80, step=5,
                help="The diastolic (lower number) blood pressure reading in mm/Hg. Normal diastolic blood pressure is around 80 mm/Hg. High blood pressure (above 90 mm/Hg) is a leading cause of chronic kidney damage. If you don't know this value, keep the default value of 80."
            )
            
    with tab_urinalysis:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            inputs['sg'] = st.selectbox(
                "Urine Specific Gravity (sg)", 
                options=[1.005, 1.010, 1.015, 1.020, 1.025],
                index=3,
                help="A urinalysis test measuring urine concentration, showing how well your kidneys dilute/concentrate fluids. Healthy range: 1.015 to 1.025. Impaired kidneys lose this filtering concentration capacity, resulting in a fixed low specific gravity (~1.010). If you don't have this urinalysis result, keep the default value of 1.020 (representing normal concentration)."
            )
        with c2:
            inputs['al'] = st.selectbox(
                "Urine Albumin (Protein) Level (al)", 
                options=[0, 1, 2, 3, 4, 5],
                index=0,
                help="Measures albumin (protein) leaking into the urine (proteinuria). Healthy kidneys filter out protein, so the normal level is 0. Higher values (1 to 5) indicate progressive kidney damage. If you don't have this urinalysis result, keep the default value of 0 (normal, no protein leak)."
            )
        with c3:
            inputs['su'] = st.selectbox(
                "Urine Sugar Level (su)", 
                options=[0, 1, 2, 3, 4, 5],
                index=0,
                help="Measures sugar/glucose leaking into the urine (glucosuria). Normal is 0, meaning healthy kidneys keep all sugar in the bloodstream. Elevated levels (1 to 5) indicate high blood sugar or diabetes. If you don't have this urinalysis result, keep the default value of 0 (normal, no sugar leak)."
            )
        with c4:
            inputs['rbc'] = st.selectbox(
                "Red Blood Cells in Urine (rbc)", 
                options=["normal", "abnormal"],
                help="Detects blood cells in urine (hematuria). Normal is 'normal' (no blood cells). 'Abnormal' indicates blood cells are leaking through damaged kidney filters or urinary tracts. If you don't have this result, keep the default selection 'normal'."
            )
            
        c5, c6, c7 = st.columns(3)
        with c5:
            inputs['pc'] = st.selectbox(
                "Pus Cells in Urine (pc)", 
                options=["normal", "abnormal"],
                help="White blood cells in urine. Normal is 'normal' (no significant white cells). 'Abnormal' indicates white cells are present, pointing to kidney inflammation or a urinary tract infection. If you don't have this result, keep the default selection 'normal'."
            )
        with c6:
            inputs['pcc'] = st.selectbox(
                "Pus Cell Clumps in Urine (pcc)", 
                options=["present", "not_present"],
                index=1,
                help="Indicates if white blood cells are clumping together in urine. Normal is 'not_present'. 'Present' indicates a severe bacterial infection in the urinary tract. If you don't have this result, keep the default selection 'not_present'."
            )
        with c7:
            inputs['ba'] = st.selectbox(
                "Bacteria in Urine (ba)", 
                options=["present", "not_present"],
                index=1,
                help="Detects visible bacteria in the urine. Normal is 'not_present'. 'Present' confirms an active bacterial infection. If you don't have this result, keep the default selection 'not_present'."
            )
            
    with tab_blood:
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs['bgr'] = st.number_input(
                "Random Blood Glucose (bgr - mg/dL)", 
                min_value=50.0, max_value=500.0, value=120.0, step=5.0,
                help="Measures the sugar levels in your blood at any random time. Normal range is 70 to 140 mg/dL. Levels above 200 mg/dL suggest diabetes, which is a major cause of kidney damage. If you don't have this blood test result, keep the default of 120.0 (representing a healthy level)."
            )
        with c2:
            inputs['bu'] = st.number_input(
                "Blood Urea (bu - mg/dL)", 
                min_value=5.0, max_value=400.0, value=35.0, step=1.0,
                help="Measures the amount of urea nitrogen (a waste product from protein digestion) in your blood. Healthy kidneys filter this waste out, keeping normal levels between 7 and 20 mg/dL, though values up to 40 mg/dL can occur. Elevated levels suggest impaired kidney filtering. If you don't have this blood test result, keep the default value of 35.0."
            )
        with c3:
            inputs['sc'] = st.number_input(
                "Serum Creatinine (sc - mg/dL)", 
                min_value=0.1, max_value=25.0, value=1.2, step=0.1,
                help="Measures creatinine, a chemical waste product of muscle metabolism. Healthy kidneys filter it completely, keeping normal levels at 0.6 to 1.2 mg/dL. Values above 1.5 mg/dL are the single strongest blood marker of moderate-to-severe loss of kidney filter capacity. If you don't have this blood test result, keep the default of 1.2 (representing a normal upper baseline)."
            )
            
        c4, c5, c6 = st.columns(3)
        with c4:
            inputs['sod'] = st.number_input(
                "Serum Sodium (sod - mEq/L)", 
                min_value=100.0, max_value=170.0, value=138.0, step=1.0,
                help="An important blood electrolyte that regulates water balance in the body. Normal range is 135 to 145 mEq/L. Abnormally low sodium can occur in advanced kidney disease due to water retention. If you don't have this blood test result, keep the default value of 138.0 (representing a healthy sodium level)."
            )
        with c5:
            inputs['pot'] = st.number_input(
                "Serum Potassium (pot - mEq/L)", 
                min_value=1.5, max_value=10.0, value=4.0, step=0.1,
                help="An essential blood electrolyte critical for heart and muscle function. Normal range is 3.5 to 5.0 mEq/L. High potassium (above 5.5 mEq/L) is dangerous and common in advanced kidney failure. If you don't have this blood test result, keep the default value of 4.0 (representing a healthy potassium level)."
            )
        with c6:
            inputs['hemo'] = st.number_input(
                "Hemoglobin (hemo - g/dL)", 
                min_value=3.0, max_value=20.0, value=13.0, step=0.1,
                help="The iron-rich protein in red blood cells that carries oxygen throughout your body. Normal range is 12.0 to 17.5 g/dL. Low hemoglobin indicates anemia, a very common complication of kidney disease because damaged kidneys fail to make enough erythropoietin (the hormone that tells your body to make red blood cells). If you don't have this blood test result, keep the default of 13.0 (representing a healthy hemoglobin level)."
            )
            
        c7, c8, c9 = st.columns(3)
        with c7:
            inputs['pcv'] = st.number_input(
                "Packed Cell Volume (pcv %)", 
                min_value=10.0, max_value=60.0, value=40.0, step=1.0,
                help="The percentage of your blood volume made up of red blood cells. Normal range is 36% to 50%. A low percentage indicates anemia, which occurs as kidney function declines. If you don't have this blood test result, keep the default of 40.0 (representing a healthy baseline)."
            )
        with c8:
            inputs['wc'] = st.number_input(
                "White Blood Cell Count (wc - cells/cumm)", 
                min_value=1000.0, max_value=30000.0, value=7500.0, step=100.0,
                help="Measures the immune cells that fight infections in the body. Normal range is 4,000 to 11,000 cells/cumm. High levels indicate an active infection or inflammation. If you don't have this blood test result, keep the default value of 7,500.0 (representing a normal, non-infected baseline)."
            )
        with c9:
            inputs['rc'] = st.number_input(
                "Red Blood Cell Count (rc - million cells/mcL)", 
                min_value=1.0, max_value=10.0, value=4.5, step=0.1,
                help="The total number of red blood cells in your blood. Normal range is 4.2 to 6.1 million cells/cmm. Low counts indicate anemia. If you don't have this blood test result, keep the default value of 4.5 (representing a healthy red blood cell count)."
            )
            
    with tab_history:
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs['htn'] = st.selectbox(
                "History of Hypertension (htn)", 
                options=["yes", "no"],
                index=1,
                help="Has the patient been diagnosed with chronic high blood pressure (hypertension)? High blood pressure is both a primary cause of kidney disease and a symptom of it. If unknown, keep the default selection 'no'."
            )
        with c2:
            inputs['dm'] = st.selectbox(
                "History of Diabetes Mellitus (dm)", 
                options=["yes", "no"],
                index=1,
                help="Has the patient been diagnosed with diabetes? Over time, high blood sugar damages the millions of tiny filtering units (nephrons) inside the kidneys. If unknown, keep the default selection 'no'."
            )
        with c3:
            inputs['cad'] = st.selectbox(
                "History of Coronary Artery Disease (cad)", 
                options=["yes", "no"],
                index=1,
                help="Has the patient been diagnosed with coronary heart disease (narrowed heart arteries or past heart attacks)? Heart and kidney health are closely linked. If unknown, keep the default selection 'no'."
            )
            
        c4, c5, c6 = st.columns(3)
        with c4:
            inputs['appet'] = st.selectbox(
                "Patient Appetite (appet)", 
                options=["good", "poor"],
                help="Does the patient currently have a 'good' or 'poor' appetite? When kidneys fail, waste products (urea) build up in the blood, causing nausea, a metallic taste in the mouth, and poor appetite. If unknown, keep the default selection 'good'."
            )
        with c5:
            inputs['pe'] = st.selectbox(
                "Presence of Pedal Edema (pe)", 
                options=["yes", "no"],
                index=1,
                help="Does the patient have swelling/fluid retention in their feet or ankles? Healthy kidneys remove excess fluids; when they fail, fluid builds up in the lower extremities. If unknown, keep the default selection 'no'."
            )
        with c6:
            inputs['ane'] = st.selectbox(
                "Presence of Anemia (ane)", 
                options=["yes", "no"],
                index=1,
                help="Does the patient show clinical symptoms of anemia (pale skin, chronic fatigue, weakness)? Kidney disease reduces red blood cell production, causing anemia. If unknown, keep the default selection 'no'."
            )
            
        # Dynamic inputs: Local risk factors are ONLY shown and used for the Nigerian Clinical Dataset model
        if dataset_type == "Nigerian Clinical":
            st.markdown("<p style='color: #00796B; font-weight: bold; margin-top: 15px;'>Local Nigerian Etiology & Risk Factors</p>", unsafe_allow_html=True)
            c7, c8, c9 = st.columns(3)
            with c7:
                inputs['chronic_glomerulonephritis'] = st.selectbox(
                    "Chronic Glomerulonephritis", 
                    options=["yes", "no"],
                    index=1,
                    help="Has the patient had chronic inflammation of the kidney's filtering units (glomerulonephritis)? This is one of the most common causes of kidney failure in West Africa. If unknown, keep the default selection 'no'."
                )
            with c8:
                inputs['use_of_herbal_remedies'] = st.selectbox(
                    "Use of Traditional Herbal Remedies", 
                    options=["yes", "no"],
                    index=1,
                    help="Does the patient have a history of regularly consuming traditional native herbs, concoctions, or wood ash infusions? Many traditional mixtures contain substances that are directly toxic to the kidneys in sub-Saharan Africa. If unknown, keep the default selection 'no'."
                )
            with c9:
                inputs['abuse_of_analgesics'] = st.selectbox(
                    "Self-Medication / Abuse of Analgesics", 
                    options=["yes", "no"],
                    index=1,
                    help="Has the patient regularly self-medicated with over-the-counter NSAID painkillers (such as ibuprofen, diclofenac, or Alabukun powder) over a long period? Chronic overuse of these drugs reduces blood flow to the kidneys, causing analgesic nephropathy. If unknown, keep the default selection 'no'."
                )
                
    st.divider()
    
    # Run Prediction Section
    predict_col, result_col = st.columns([1, 2])
    
    with predict_col:
        st.markdown("<br>", unsafe_allow_html=True)
        run_prediction = st.button(
            "Generate Diagnostic Prediction",
            help="Click here to run the selected Machine Learning model on the inputs above and predict CKD status."
        )
        
        # Display information card
        st.markdown(
            f"""
            <div class="metric-card">
                <strong>Current Pipeline Selection:</strong><br>
                • Dataset: {dataset_type}<br>
                • Classifier: {model_name}<br>
                <span style="font-size: 11px; color:#666;">Inputs will be preprocessed automatically using the fitted pipeline transformer.</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with result_col:
        if run_prediction:
            # Create a dataframe from inputs matching the column sequence expected by the pipeline
            df_input = pd.DataFrame([inputs])
            
            # Map strings to standard representations just in case
            categorical_cols = df_input.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                df_input[col] = df_input[col].astype(str).str.strip().str.lower()
                
            try:
                # Reorder columns to match the exact feature sequence from training
                if hasattr(pipeline, "feature_names_in_"):
                    df_input = df_input[pipeline.feature_names_in_]
                
                # Predict
                prediction = pipeline.predict(df_input)[0]
                probabilities = pipeline.predict_proba(df_input)[0]
                ckd_probability = probabilities[1]
                
                # Display Results
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="ckd-card">
                            <h3 style="color:#B71C1C; margin-top:0px; border-bottom:none;">CRITICAL: Chronic Kidney Disease Detected</h3>
                            <p style="font-size: 16px; margin-bottom: 0px;">
                                The patient is predicted to have <strong>Chronic Kidney Disease (CKD)</strong>.<br>
                                • Risk Probability: <strong>{ckd_probability*100:.2f}%</strong><br>
                                • Recommendation: Immediate nephrology consultation, kidney function tracking (eGFR), and blood pressure management are strongly advised.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="healthy-card">
                            <h3 style="color:#1B5E20; margin-top:0px; border-bottom:none;">DIAGNOSIS: Normal / No CKD Detected</h3>
                            <p style="font-size: 16px; margin-bottom: 0px;">
                                The patient is predicted to have <strong>Normal Renal Function</strong>.<br>
                                • Risk Probability: <strong>{(1-ckd_probability)*100:.2f}% Probability of Normal Function</strong> (CKD Risk: {ckd_probability*100:.2f}%)<br>
                                • Recommendation: Maintain healthy diet, standard blood pressure, and periodic screening.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}. Check if features match the model expected columns.")
        else:
            st.info("Click the 'Generate Diagnostic Prediction' button on the left to run classification.")

if __name__ == "__main__":
    main()
