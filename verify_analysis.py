import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os

# Create directory for plots
if not os.path.exists('verification_plots'):
    os.makedirs('verification_plots')

# Set plot style (optional replacement for sns.set)
plt.style.use('ggplot')
file_path = 'arrivals (1).csv'
df = pd.read_csv(file_path)

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Filter for 'ALL' countries
df_total = df[df['country'] == 'ALL'].copy()
df_total = df_total.sort_values('date').set_index('date')
print(f"Data loaded: {len(df_total)} rows.")

# --- GROUND TRUTH ---
print("Defining Ground Truth...")
start_covid = '2020-03-01'
end_covid = '2022-04-01'
df_total['ground_truth'] = 0
df_total.loc[(df_total.index >= start_covid) & (df_total.index <= end_covid), 'ground_truth'] = 1
print(f"Ground Truth Anomalies: {df_total['ground_truth'].sum()} months.")

# --- DBSCAN ---
print("\nRunning DBSCAN...")
scaler = StandardScaler()
X_dbscan = scaler.fit_transform(df_total[['arrivals']])
dbscan = DBSCAN(eps=0.5, min_samples=3)
df_total['dbscan_labels'] = dbscan.fit_predict(X_dbscan)
# Convert -1 to 1 (Anomaly), others to 0
df_total['dbscan_anomaly'] = df_total['dbscan_labels'].apply(lambda x: 1 if x == -1 else 0)

# Plot DBSCAN
plt.figure(figsize=(14, 6))
plt.plot(df_total.index, df_total['arrivals'], label='Arrivals', color='blue', alpha=0.6)
plt.scatter(df_total[df_total['dbscan_anomaly'] == 1].index, 
            df_total[df_total['dbscan_anomaly'] == 1]['arrivals'], 
            color='orange', label='DBSCAN Anomaly', s=50, zorder=5)
plt.title('DBSCAN Anomaly Detection')
plt.legend()
plt.savefig('verification_plots/dbscan_results.png')
print("DBSCAN plot saved.")

# --- ISOLATION FOREST ---
print("\nRunning Isolation Forest...")
X_iso = df_total[['arrivals']]
iso_forest = IsolationForest(contamination=0.4, random_state=42)
df_total['iso_labels'] = iso_forest.fit_predict(X_iso)
# Convert -1 to 1 (Anomaly)
df_total['iso_anomaly'] = df_total['iso_labels'].apply(lambda x: 1 if x == -1 else 0)

# Plot Isolation Forest
plt.figure(figsize=(14, 6))
plt.plot(df_total.index, df_total['arrivals'], label='Arrivals', color='blue', alpha=0.6)
plt.scatter(df_total[df_total['iso_anomaly'] == 1].index, 
            df_total[df_total['iso_anomaly'] == 1]['arrivals'], 
            color='purple', label='Isolation Forest Anomaly', s=50, zorder=5)
plt.title('Isolation Forest Anomaly Detection')
plt.legend()
plt.savefig('verification_plots/isolation_forest_results.png')
print("Isolation Forest plot saved.")

# --- EVALUATION ---
print("\n" + "="*30)
print("EVALUATION RESULTS")
print("="*30)

print("\n--- DBSCAN Evaluation ---")
print(classification_report(df_total['ground_truth'], df_total['dbscan_anomaly'], target_names=['Normal', 'Anomaly']))
print("Confusion Matrix:\n", confusion_matrix(df_total['ground_truth'], df_total['dbscan_anomaly']))

print("\n--- Isolation Forest Evaluation ---")
print(classification_report(df_total['ground_truth'], df_total['iso_anomaly'], target_names=['Normal', 'Anomaly']))
print("Confusion Matrix:\n", confusion_matrix(df_total['ground_truth'], df_total['iso_anomaly']))

# --- QUALITATIVE SUMMARY ---
db_f1 = f1_score(df_total['ground_truth'], df_total['dbscan_anomaly'])
iso_f1 = f1_score(df_total['ground_truth'], df_total['iso_anomaly'])

print("\nSummary:")
print(f"DBSCAN F1-Score: {db_f1:.2f}")
print(f"Isolation Forest F1-Score: {iso_f1:.2f}")

if iso_f1 > db_f1:
    print("Conclusion: Isolation Forest performed better on this dataset.")
elif db_f1 > iso_f1:
    print("Conclusion: DBSCAN performed better on this dataset.")
else:
    print("Conclusion: Both models performed similarly.")
