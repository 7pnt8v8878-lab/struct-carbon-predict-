# struct-carbon-predict-import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Simulate Structural Sensor Data for a Bridge Network
np.random.seed(42)
n_samples = 5000

data = {
    'vibration_hz': np.random.normal(15.0, 2.5, n_samples),
    'load_stress_mpa': np.random.normal(45.0, 8.0, n_samples),
    'temp_delta_c': np.random.normal(10.0, 4.0, n_samples),
    'crack_width_mm': np.random.exponential(0.5, n_samples)
}

df = pd.DataFrame(data)

# Tuned failure risk rules to reflect realistic structural fatigue
df['failure_risk'] = (
    ((df['vibration_hz'] > 17.0) & (df['load_stress_mpa'] > 50.0)) | 
    (df['crack_width_mm'] > 1.0)
).astype(int)

# Introduce realistic, minimal telemetry noise (1% of dataset)
noise_idx = np.random.choice(df.index, size=50, replace=False)
df.loc[noise_idx, 'failure_risk'] = 1 - df.loc[noise_idx, 'failure_risk']

# 2. Train a Predictive Maintenance Model
X = df[['vibration_hz', 'load_stress_mpa', 'temp_delta_c', 'crack_width_mm']]
y = df['failure_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n--- Model Performance ---")
print(classification_report(y_test, y_pred))

# 3. The Carbon Impact Engine
def calculate_carbon_offset(predicted_failures):
    avg_repair_emissions_tons = 12.0     
    avg_rebuild_emissions_tons = 450.0   
    
    total_rebuild_avoided = predicted_failures * avg_rebuild_emissions_tons
    total_repair_cost = predicted_failures * avg_repair_emissions_tons
    
    net_carbon_saved = total_rebuild_avoided - total_repair_cost
    return net_carbon_saved

total_caught_failures = y_pred.sum()
carbon_saved = calculate_carbon_offset(total_caught_failures)

print(f"\n--- Sustainability Impact Analysis ---")
print(f"Total Structural Failures Predicted & Prevented: {total_caught_failures}")
print(f"Estimated Embodied Carbon Saved: {carbon_saved:,.2f} metric tons of CO2e")
