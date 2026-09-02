import os
import json
from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import shap

from preprocessing import ClevelandPreprocessor

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model    = joblib.load(os.path.join(BASE_DIR, 'models/best_model.pkl'))
scaler   = joblib.load(os.path.join(BASE_DIR, 'models/scaler.pkl'))
features = joblib.load(os.path.join(BASE_DIR, 'models/selected_features.pkl'))
preprocessor = joblib.load(os.path.join(BASE_DIR, 'models/preprocessor.pkl'))
with open(os.path.join(BASE_DIR, 'models/model_metadata.json')) as metadata_file:
    model_metadata = json.load(metadata_file)

# Build explainer background from full training data
df_bg    = pd.read_csv(os.path.join(BASE_DIR, 'data/heart_cleveland_upload.csv'))
X_bg     = preprocessor.transform(df_bg.drop('condition', axis=1))
X_bg_sel = X_bg[features]
X_bg_sc  = scaler.transform(X_bg_sel)
explainer = shap.LinearExplainer(model, X_bg_sc)

FEATURE_LABELS = {
    'age':         'Age',
    'sex':         'Sex',
    'trestbps':    'Resting Blood Pressure',
    'chol':        'Serum Cholesterol',
    'fbs':         'Fasting Blood Sugar',
    'thalach':     'Max Heart Rate',
    'exang':       'Exercise Angina',
    'oldpeak':     'ST Depression',
    'ca':          'Major Vessels (ca)',
    'cp_2':        'Non-Anginal Pain',
    'cp_3':        'Asymptomatic',
    'restecg_2':   'LV Hypertrophy',
    'thal_2':      'Reversible Defect',
    'slope_1':     'Flat ST Slope',
    'hr_bp_ratio': 'Heart Rate / BP Ratio',
}

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/assess', methods=['GET', 'POST'])
def assess():
    if request.method == 'POST':
        try:
            form = request.form

            age      = float(form['age'])
            sex      = float(form['sex'])
            trestbps = float(form['trestbps'])
            chol     = float(form['chol'])
            fbs      = float(form['fbs'])
            thalach  = float(form['thalach'])
            exang    = float(form['exang'])
            oldpeak  = float(form['oldpeak'])
            ca       = float(form['ca'])

            raw_input = pd.DataFrame([{
                'age': age, 'sex': sex, 'cp': int({'typical': 0, 'atypical': 1, 'non': 2, 'asymptomatic': 3}[form['cp']]),
                'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
                'restecg': int({'normal': 0, 'stt': 1, 'lv': 2}[form['restecg']]),
                'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
                'slope': int({'up': 0, 'flat': 1, 'down': 2}[form['slope']]),
                'ca': ca, 'thal': int({'normal': 0, 'fixed': 1, 'reversible': 2}[form['thal']]),
            }])
            x_features = preprocessor.transform(raw_input)[features]
            x_scaled = scaler.transform(x_features)

            prob  = model.predict_proba(x_scaled)[0][1]
            pred  = int(prob >= 0.5)
            risk  = 'High' if prob > 0.65 else 'Moderate' if prob > 0.35 else 'Low'
            color = '#e74c3c' if risk == 'High' else '#f39c12' if risk == 'Moderate' else '#27ae60'

            shap_vals  = explainer.shap_values(x_scaled)[0]
            shap_pairs = sorted(
                zip(features, shap_vals),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:6]

            return render_template('result.html',
                prob=round(prob * 100, 1),
                pred=pred,
                risk=risk,
                color=color,
                shap_pairs=shap_pairs,
                feature_labels=FEATURE_LABELS,
                model_metadata=model_metadata)

        except Exception as e:
            return render_template('assess.html', error=str(e))

    return render_template('assess.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)