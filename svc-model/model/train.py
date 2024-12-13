import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

def train_model():
    # Load your dataset (replace with actual data loading)
    X = pd.read_csv('train_features.csv')
    y = pd.read_csv('train_labels.csv')
    
    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train SVC model
    svc = SVC(kernel='rbf', probability=True)
    svc.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    joblib.dump(svc, '/opt/ml/model/svc_model.joblib')
    joblib.dump(scaler, '/opt/ml/model/scaler.joblib')

if __name__ == '__main__':
    train_model()