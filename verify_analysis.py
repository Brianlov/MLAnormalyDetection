import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('arrivals (1).csv')
df['date'] = pd.to_datetime(df['date'])
df_total = df[df['country'] == 'ALL'].copy()
df_total = df_total.sort_values('date').set_index('date')

# --- IMPROVED GROUND TRUTH DEFINITION ---
# COVID-19 period (adjust based on your domain knowledge)
covid_start = pd.to_datetime('2020-03-01')
covid_end = pd.to_datetime('2022-04-01')

# Create ground truth
df_total['ground_truth'] = 0
df_total.loc[(df_total.index >= covid_start) & (df_total.index <= covid_end), 'ground_truth'] = 1

print(f"Total periods: {len(df_total)}")
print(f"Anomaly periods (COVID): {df_total['ground_truth'].sum()}")
print(f"Normal periods: {len(df_total) - df_total['ground_truth'].sum()}")
print(f"Anomaly ratio: {df_total['ground_truth'].mean():.2%}")

# --- IMPROVED DBSCAN ---
print("\n" + "="*50)
print("DBSCAN with parameter tuning")
print("="*50)

# Try different scalers and parameters
results_dbscan = []

# Parameter grid to try
param_grid = [
    {'eps': 0.3, 'min_samples': 3, 'scaler': StandardScaler()},
    {'eps': 0.2, 'min_samples': 3, 'scaler': RobustScaler()},
    {'eps': 0.25, 'min_samples': 4, 'scaler': StandardScaler()},
    {'eps': 0.15, 'min_samples': 5, 'scaler': RobustScaler()},
]

for i, params in enumerate(param_grid):
    print(f"\n--- DBSCAN Configuration {i+1} ---")
    print(f"eps={params['eps']}, min_samples={params['min_samples']}, scaler={params['scaler'].__class__.__name__}")
    
    # Scale data
    scaler = params['scaler']
    X_scaled = scaler.fit_transform(df_total[['arrivals']])
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    labels = dbscan.fit_predict(X_scaled)
    
    # Convert to anomaly labels (-1 -> 1, else -> 0)
    anomaly_labels = np.where(labels == -1, 1, 0)
    
    # Calculate metrics
    f1 = f1_score(df_total['ground_truth'], anomaly_labels)
    anomaly_count = anomaly_labels.sum()
    
    results_dbscan.append({
        'config': i+1,
        'eps': params['eps'],
        'min_samples': params['min_samples'],
        'scaler': params['scaler'].__class__.__name__,
        'f1_score': f1,
        'anomalies_detected': anomaly_count,
        'labels': anomaly_labels.copy()
    })
    
    print(f"  F1-Score: {f1:.3f}")
    print(f"  Anomalies detected: {anomaly_count}")
    print(f"  Actual anomalies: {df_total['ground_truth'].sum()}")

# Select best DBSCAN configuration
best_dbscan = max(results_dbscan, key=lambda x: x['f1_score'])
df_total['dbscan_best'] = best_dbscan['labels']

print(f"\n✓ Best DBSCAN Configuration:")
print(f"  Config {best_dbscan['config']}: eps={best_dbscan['eps']}, "
      f"min_samples={best_dbscan['min_samples']}, scaler={best_dbscan['scaler']}")
print(f"  F1-Score: {best_dbscan['f1_score']:.3f}")

# --- IMPROVED ISOLATION FOREST ---
print("\n" + "="*50)
print("Isolation Forest with parameter tuning")
print("="*50)

# Use actual anomaly ratio for contamination
actual_contamination = df_total['ground_truth'].mean()
print(f"Actual anomaly ratio in data: {actual_contamination:.2%}")

results_iso = []

# Try different contamination values
raw_contamination_values = [
    actual_contamination,  # Match actual ratio
    actual_contamination * 1.5,  # Slightly higher
    actual_contamination * 2,    # Higher
    max(0.1, actual_contamination),  # At least 10%
    0.2,  # Fixed 20%
    0.3,  # Fixed 30%
]

# Filter out invalid values for Isolation Forest (must be <= 0.5)
contamination_values = [c for c in raw_contamination_values if 0 < c <= 0.5]

if not contamination_values:
    print(f"Warning: Calculated contamination {actual_contamination:.2%} is too high for Isolation Forest (max 0.5).")
    print("Using default fixed values <= 0.5 for benchmarking.")
    contamination_values = [0.1, 0.2, 0.3, 0.4, 0.5]
else:
    print(f"Filtered {len(raw_contamination_values)-len(contamination_values)} invalid contamination values (> 0.5)")

for i, contam in enumerate(contamination_values):
    print(f"\n--- Isolation Forest Configuration {i+1} ---")
    print(f"contamination={contam:.3f}")
    
    # Apply Isolation Forest
    iso = IsolationForest(
        contamination=contam,
        random_state=42,
        n_estimators=100
    )
    
    # Fit and predict
    iso_labels = iso.fit_predict(df_total[['arrivals']].values)
    anomaly_labels = np.where(iso_labels == -1, 1, 0)
    
    # Calculate metrics
    f1 = f1_score(df_total['ground_truth'], anomaly_labels)
    anomaly_count = anomaly_labels.sum()
    
    results_iso.append({
        'config': i+1,
        'contamination': contam,
        'f1_score': f1,
        'anomalies_detected': anomaly_count,
        'labels': anomaly_labels.copy()
    })
    
    print(f"  F1-Score: {f1:.3f}")
    print(f"  Anomalies detected: {anomaly_count}")

# Select best Isolation Forest configuration
best_iso = max(results_iso, key=lambda x: x['f1_score'])
df_total['iso_best'] = best_iso['labels']

print(f"\n✓ Best Isolation Forest Configuration:")
print(f"  Config {best_iso['config']}: contamination={best_iso['contamination']:.3f}")
print(f"  F1-Score: {best_iso['f1_score']:.3f}")

# --- VISUALIZATION ---
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Plot 1: Original data with ground truth
ax1 = axes[0]
ax1.plot(df_total.index, df_total['arrivals'], 'b-', alpha=0.6, label='Arrivals')
ax1.fill_between(df_total.index, df_total['arrivals'].min(), df_total['arrivals'].max(),
                 where=(df_total['ground_truth'] == 1),
                 alpha=0.3, color='red', label='Ground Truth Anomalies')
ax1.set_title('Original Data with Ground Truth Anomalies')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: DBSCAN results
ax2 = axes[1]
ax2.plot(df_total.index, df_total['arrivals'], 'b-', alpha=0.6, label='Arrivals')
dbscan_anomalies = df_total[df_total['dbscan_best'] == 1]
ax2.scatter(dbscan_anomalies.index, dbscan_anomalies['arrivals'],
            color='orange', s=50, label='DBSCAN Anomalies', zorder=5)
ax2.set_title(f'DBSCAN Anomaly Detection (F1: {best_dbscan["f1_score"]:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Isolation Forest results
ax3 = axes[2]
ax3.plot(df_total.index, df_total['arrivals'], 'b-', alpha=0.6, label='Arrivals')
iso_anomalies = df_total[df_total['iso_best'] == 1]
ax3.scatter(iso_anomalies.index, iso_anomalies['arrivals'],
            color='purple', s=50, label='Isolation Forest Anomalies', zorder=5)
ax3.set_title(f'Isolation Forest Anomaly Detection (F1: {best_iso["f1_score"]:.3f})')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('verification_plots/comparison_results.png', dpi=300, bbox_inches='tight')

# --- FINAL EVALUATION ---
print("\n" + "="*50)
print("FINAL EVALUATION")
print("="*50)

print("\n--- Best DBSCAN Results ---")
print(classification_report(df_total['ground_truth'], df_total['dbscan_best'], 
                           target_names=['Normal', 'Anomaly']))
print("Confusion Matrix:")
print(confusion_matrix(df_total['ground_truth'], df_total['dbscan_best']))

print("\n--- Best Isolation Forest Results ---")
print(classification_report(df_total['ground_truth'], df_total['iso_best'], 
                           target_names=['Normal', 'Anomaly']))
print("Confusion Matrix:")
print(confusion_matrix(df_total['ground_truth'], df_total['iso_best']))

# --- COMPARISON SUMMARY ---
print("\n" + "="*50)
print("COMPARISON SUMMARY")
print("="*50)

print(f"\nDBSCAN Best F1-Score: {best_dbscan['f1_score']:.3f}")
print(f"Isolation Forest Best F1-Score: {best_iso['f1_score']:.3f}")

if best_iso['f1_score'] > best_dbscan['f1_score']:
    print(f"\n✓ Isolation Forest performs better by {best_iso['f1_score'] - best_dbscan['f1_score']:.3f} F1-score")
elif best_dbscan['f1_score'] > best_iso['f1_score']:
    print(f"\n✓ DBSCAN performs better by {best_dbscan['f1_score'] - best_iso['f1_score']:.3f} F1-score")
else:
    print("\n✓ Both models perform similarly")

# Check if models are better than random guessing
random_f1 = 2 * actual_contamination * (1 - actual_contamination) / (actual_contamination + (1 - actual_contamination))
print(f"\nRandom guessing F1-score (baseline): {random_f1:.3f}")

if best_dbscan['f1_score'] > random_f1:
    print("✓ DBSCAN is better than random guessing")
else:
    print("✗ DBSCAN is NOT better than random guessing")

if best_iso['f1_score'] > random_f1:
    print("✓ Isolation Forest is better than random guessing")
else:
    print("✗ Isolation Forest is NOT better than random guessing")

print("\nSaving results to CSV...")
df_total[['arrivals', 'ground_truth', 'dbscan_best', 'iso_best']].to_csv('verification_plots/anomaly_results.csv')
print("Results saved to 'verification_plots/anomaly_results.csv'")
print(f"Plots saved to 'verification_plots/comparison_results.png'")
