from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import shap

app = Flask(__name__)

model    = joblib.load('models/best_model.pkl')
scaler   = joblib.load('models/scaler.pkl')
features = joblib.load('models/selected_features.pkl')

# Build explainer background from full training data
df_bg    = pd.read_csv('data/heart_cleveland_upload.csv')
X_bg     = df_bg.drop('condition', axis=1)
X_bg     = pd.get_dummies(X_bg, columns=['cp','restecg','thal','slope'], drop_first=True)
X_bg['hr_bp_ratio'] = X_bg['thalach'] / X_bg['trestbps']
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

            cp        = form['cp']
            cp_2      = 1.0 if cp == 'non'          else 0.0
            cp_3      = 1.0 if cp == 'asymptomatic' else 0.0

            restecg   = form['restecg']
            restecg_2 = 1.0 if restecg == 'lv'       else 0.0

            thal      = form['thal']
            thal_2    = 1.0 if thal == 'reversible'  else 0.0

            slope     = form['slope']
            slope_1   = 1.0 if slope == 'flat'       else 0.0

            hr_bp_ratio = thalach / trestbps

            feature_map = {
                'age':         age,
                'sex':         sex,
                'trestbps':    trestbps,
                'chol':        chol,
                'fbs':         fbs,
                'thalach':     thalach,
                'exang':       exang,
                'oldpeak':     oldpeak,
                'ca':          ca,
                'cp_2':        cp_2,
                'cp_3':        cp_3,
                'restecg_2':   restecg_2,
                'thal_2':      thal_2,
                'slope_1':     slope_1,
                'hr_bp_ratio': hr_bp_ratio,
            }

            vals     = [feature_map[f] for f in features]
            x_raw    = np.array(vals).reshape(1, -1)
            x_scaled = scaler.transform(x_raw)

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
                feature_labels=FEATURE_LABELS)

        except Exception as e:
            return render_template('assess.html', error=str(e))

    return render_template('assess.html')

if __name__ == '__main__':
    app.run(debug=True)