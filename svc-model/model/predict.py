import flask
import joblib
import numpy as np
import os

app = flask.Flask(__name__)

# Load model and scaler
MODEL_PATH = '/opt/ml/model/'
svc_model = joblib.load(os.path.join(MODEL_PATH, 'svc_model.joblib'))
scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.joblib'))

@app.route('/ping', methods=['GET'])
def ping():
    return flask.Response(response='\n', status=200)

@app.route('/invocations', methods=['POST'])
def prediction():
    # Parse input data
    input_data = flask.request.json
    X = np.array(input_data['instances'])
    
    # Preprocess and predict
    X_scaled = scaler.transform(X)
    predictions = svc_model.predict(X_scaled)
    proba = svc_model.predict_proba(X_scaled)
    
    return flask.jsonify({
        'predictions': predictions.tolist(),
        'probabilities': proba.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)