import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Load and prepare data
df = pd.read_csv('arrivals (1).csv')
df['date'] = pd.to_datetime(df['date'])
df_total = df[df['country'] == 'ALL'].copy()
df_total = df_total.sort_values('date').set_index('date')

# --- GROUND TRUTH ---
covid_start = pd.to_datetime('2020-03-01')
covid_end = pd.to_datetime('2022-04-01')

df_total['ground_truth'] = 0
df_total.loc[(df_total.index >= covid_start) & (df_total.index <= covid_end), 'ground_truth'] = 1

# --- SPLIT DATA (SEMI-SUPERVISED / NOVELTY DETECTION) ---
# Train on ALL Normal data (Pre-COVID AND Post-COVID)
# Since we have labels, we use all known "Normal" periods to define the boundary
train_data = df_total[df_total['ground_truth'] == 0].copy()

# Test on everything (to see if it correctly flags the COVID period as different)
X_train = train_data[['arrivals']].values
X_full = df_total[['arrivals']].values

print(f"Training Data (Normal periods): {len(train_data)}")
print(f"Full Data: {len(df_total)} periods")

# --- MODEL 1: ISOLATION FOREST (NOVELTY DETECTION) ---
print("\n" + "="*50)
print("Isolation Forest (Semi-Supervised / Novelty Mode)")
print("="*50)

# Contamination is now the expected % of outliers IN THE TRAINING SET.
# Since we chose "Normal" data, contamination should be very low (e.g. 0.01 or 0.05)
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_train)

# Predict on full data
# -1 = Anomaly, 1 = Normal
iso_preds = iso_forest.predict(X_full)
df_total['iso_novelty'] = np.where(iso_preds == -1, 1, 0)

f1_iso = f1_score(df_total['ground_truth'], df_total['iso_novelty'])
print(f"Isolation Forest F1-Score: {f1_iso:.3f}")
print(classification_report(df_total['ground_truth'], df_total['iso_novelty'], target_names=['Normal', 'Anomaly']))

# --- MODEL 2: ONE-CLASS SVM ---
print("\n" + "="*50)
print("One-Class SVM")
print("="*50)

# OCSVM is excellent for learning a boundary around normal data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_full_scaled = scaler.transform(X_full)

# nu approaches the upper bound on the fraction of training errors and lower bound of the fraction of support vectors.
oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
oc_svm.fit(X_train_scaled)

svm_preds = oc_svm.predict(X_full_scaled)
df_total['svm_novelty'] = np.where(svm_preds == -1, 1, 0)

f1_svm = f1_score(df_total['ground_truth'], df_total['svm_novelty'])
print(f"One-Class SVM F1-Score: {f1_svm:.3f}")
print(classification_report(df_total['ground_truth'], df_total['svm_novelty'], target_names=['Normal', 'Anomaly']))

# --- VISUALIZATION ---
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Plot 1: Isolation Forest
ax1 = axes[0]
ax1.plot(df_total.index, df_total['arrivals'], 'b-', alpha=0.6, label='Arrivals')
# Highlight Training Period
ax1.axvspan(df_total.index.min(), covid_start, alpha=0.1, color='green', label='Training Data (Normal)')

novelty_iso = df_total[df_total['iso_novelty'] == 1]
ax1.scatter(novelty_iso.index, novelty_iso['arrivals'], color='red', s=50, label='Predicted Anomaly')
ax1.set_title(f'Semi-Supervised Isolation Forest (F1: {f1_iso:.3f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: One-Class SVM
ax2 = axes[1]
ax2.plot(df_total.index, df_total['arrivals'], 'b-', alpha=0.6, label='Arrivals')
# Highlight Training Period
ax2.axvspan(df_total.index.min(), covid_start, alpha=0.1, color='green', label='Training Data (Normal)')

novelty_svm = df_total[df_total['svm_novelty'] == 1]
ax2.scatter(novelty_svm.index, novelty_svm['arrivals'], color='purple', s=50, label='Predicted Anomaly')
ax2.set_title(f'Semi-Supervised One-Class SVM (F1: {f1_svm:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('verification_plots/semi_supervised_results.png', dpi=300)
print(f"\nPlots saved to 'verification_plots/semi_supervised_results.png'")

# --- SUMMARY ---
print("\n" + "="*50)
print("SUMMARY Comparison")
print("="*50)
print(f"Isolation Forest (Novelty) F1: {f1_iso:.3f}")
print(f"One-Class SVM F1: {f1_svm:.3f}")

if f1_svm > f1_iso:
    print("✓ One-Class SVM is the winner")
else:
    print("✓ Isolation Forest is the winner")
