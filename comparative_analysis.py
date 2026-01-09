import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.neighbors import NearestNeighbors

# --- 1. DATA PREPARATION ---
df = pd.read_csv('arrivals (1).csv')
df['date'] = pd.to_datetime(df['date'])
df_total = df[df['country'] == 'ALL'].copy()
df_total = df_total.sort_values('date').set_index('date')

# Ground Truth (COVID: Mar 2020 - Apr 2022)
covid_start = pd.to_datetime('2020-03-01')
covid_end = pd.to_datetime('2022-04-01')
df_total['ground_truth'] = 0
df_total.loc[(df_total.index >= covid_start) & (df_total.index <= covid_end), 'ground_truth'] = 1

X = df_total[['arrivals']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Results storage
results = []

# --- 2. MODEL IMPLEMENTATIONS ---

# Model A: DBSCAN (Unsupervised)
# using best params from previous run: eps=0.25, min_samples=4
dbscan = DBSCAN(eps=0.25, min_samples=4)
labels_db = dbscan.fit_predict(X_scaled)
# DBSCAN: -1 is anomaly, others used to be clusters (0, 1, etc.)
# We treat -1 as Anomaly (1), everything else as Normal (0)
pred_db = np.where(labels_db == -1, 1, 0)

# Model B: Isolation Forest (Unsupervised)
# using best "unfiltered" contamination ~0.45 or limiting to 0.5.
# Let's use 0.45 as it's close to the known ratio, giving it the best fighting chance
iso_unsupervised = IsolationForest(contamination=0.45, random_state=42)
pred_iso_unsup = iso_unsupervised.fit_predict(X)
# IsoForest: -1 is anomaly, 1 is normal. Convert to 0/1
pred_iso_unsup = np.where(pred_iso_unsup == -1, 1, 0)

# Model C: K-Means (Unsupervised)
# Approach: Cluster into K=2 (Normal vs Low/COVID).
# Assumption: The smaller cluster (or the one with lower mean?) is the anomaly?
# Actually, let's treat it as "Clustering-based Anomaly Detection":
# 1. Fit K-Means
# 2. Calculate distance to nearest centroid
# 3. Threshold distances (e.g. top X%)? 
# OR simpler: Since we know the data has two regimes, K=2 might solve it perfectly 
# if we interpret the "COVID cluster" as anomalies.
kmeans = KMeans(n_clusters=2, random_state=42)
labels_km = kmeans.fit_predict(X_scaled)

# Determine which cluster is "Normal" vs "Anomaly"
# Logic: "Normal" usually has higher arrivals. Calculate mean of each cluster.
centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_means = [centers[0][0], centers[1][0]]
normal_cluster = 1 if cluster_means[1] > cluster_means[0] else 0
anomaly_cluster = 1 - normal_cluster # The other one

pred_km = np.where(labels_km == anomaly_cluster, 1, 0)

# Model D: Isolation Forest (Semi-Supervised / Novelty) -- THE "BETTER" APPROACH
# Train on Normal data only
train_data = df_total[df_total['ground_truth'] == 0].copy()
X_train = train_data[['arrivals']].values

iso_semi = IsolationForest(contamination=0.01, random_state=42) # Low contamination assumption for training set
iso_semi.fit(X_train)
pred_iso_semi_raw = iso_semi.predict(X)
pred_iso_semi = np.where(pred_iso_semi_raw == -1, 1, 0)

# Model E: DBSCAN (Semi-Supervised)
# Train on Normal data only
mask_normal = df_total['ground_truth'] == 0
X_train_scaled = X_scaled[mask_normal]

db_semi = DBSCAN(eps=0.25, min_samples=4)
db_semi.fit(X_train_scaled)

core_samples = db_semi.components_

if len(core_samples) == 0:
    pred_db_semi = np.ones(len(X_scaled)) # All anomalies if no normal clusters found
else:
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(core_samples)
    dists, _ = neigh.kneighbors(X_scaled)
    # If distance to nearest core sample > eps, it's an anomaly (1)
    pred_db_semi = np.where(dists.flatten() > db_semi.eps, 1, 0)

# --- 3. EVALUATION & SAVING ---

models = {
    'DBSCAN (Unsup)': pred_db,
    'Isolation Forest (Unsup)': pred_iso_unsup,
    'K-Means (Unsup)': pred_km,
    'IsoForest (Semi-Supervised)': pred_iso_semi,
    'DBSCAN (Semi-Supervised)': pred_db_semi
}

# Print Comparison Table
print(f"{'Model':<30} | {'F1':<6} | {'Prec':<6} | {'Rec':<6} | {'Acc':<6} | {'TN':<4} {'FP':<4} {'FN':<4} {'TP':<4}")
print("-" * 95)

results = []
for name, preds in models.items():
    # Summary Table Metrics
    f1 = f1_score(df_total['ground_truth'], preds)
    prec = precision_score(df_total['ground_truth'], preds)
    rec = recall_score(df_total['ground_truth'], preds)
    acc = np.mean(preds == df_total['ground_truth'])
    tn, fp, fn, tp = confusion_matrix(df_total['ground_truth'], preds).ravel()
    
    results.append({'Model': name, 'F1': f1, 'Precision': prec, 'Recall': rec, 'Accuracy': acc, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
    print(f"{name:<30} | {f1:.3f}  | {prec:.3f}  | {rec:.3f}  | {acc:.3f}  | {tn:<4} {fp:<4} {fn:<4} {tp:<4}")

# Print Detailed Reports
print("\n" + "="*40)
print("DETAILED CLASSIFICATION REPORTS")
print("="*40)

for name, preds in models.items():
    print(f"\n--- {name} Evaluation ---")
    print(classification_report(df_total['ground_truth'], preds))
    print("Confusion Matrix:")
    print(confusion_matrix(df_total['ground_truth'], preds))
    print("-" * 40)

# --- 4. VISUALIZATION ---
fig, axes = plt.subplots(len(models), 1, figsize=(12, 4*len(models)))
colors = {'DBSCAN (Unsup)': 'orange', 'Isolation Forest (Unsup)': 'purple', 
          'K-Means (Unsup)': 'cyan', 'IsoForest (Semi-Supervised)': 'lime',
          'DBSCAN (Semi-Supervised)': 'magenta'}

for i, (name, preds) in enumerate(models.items()):
    ax = axes[i]
    # Plot original line
    ax.plot(df_total.index, df_total['arrivals'], 'b-', alpha=0.5, label='Arrivals')
    
    # Plot ground truth background
    ax.fill_between(df_total.index, df_total['arrivals'].min(), df_total['arrivals'].max(),
                    where=(df_total['ground_truth'] == 1),
                    alpha=0.1, color='red', label='Ground Truth (COVID)')
    
    # Plot Predictions
    anomalies = df_total[preds == 1]
    ax.scatter(anomalies.index, anomalies['arrivals'], color=colors[name], s=40, label=f'Predicted Anomaly', zorder=5)
    
    # Calculate metrics for title
    metric = next(r for r in results if r['Model'] == name)
    ax.set_title(f"{name}\nF1: {metric['F1']:.3f} | Accuracy: {metric['Accuracy']:.3f}")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('verification_plots/final_comparison.png', dpi=300)
print("\nPlot saved to 'verification_plots/final_comparison.png'")

# --- 5. CONFUSION MATRICES (Unsupervised) ---
# User requested specifically for the three unsupervised methods
unsup_models = {k: v for k, v in models.items() if '(Unsup)' in k}

fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))
fig_cm.suptitle("Confusion Matrices: Unsupervised Models", fontsize=16)

for i, (name, preds) in enumerate(unsup_models.items()):
    ax = axes_cm[i]
    cm = confusion_matrix(df_total['ground_truth'], preds)
    # Labels: 0=Normal, 1=Anomaly
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    ax.set_title(name)
    ax.grid(False) # Disable grid for CM

plt.tight_layout()
plt.savefig('verification_plots/unsupervised_confusion_matrices.png', dpi=300)
print("Confusion matrices saved to 'verification_plots/unsupervised_confusion_matrices.png'")
