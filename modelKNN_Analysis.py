import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('C:/Master Thesis/Anomaly detection implementation/combined_data_breaks.csv')
labels = data['BREAKS']
features = data[['ID', 'Tardiness', 'Overall_processing_time']]

scaler = RobustScaler()
features_scaled = pd.DataFrame(
    scaler.fit_transform(features),
    columns=features.columns,
    index=features.index
)

labels = pd.Series(labels, index=features.index)

train_idx, test_idx = train_test_split(
    features_scaled.index,
    test_size=0.2,
    random_state=42
)

X_train = features_scaled.loc[train_idx]
X_test = features_scaled.loc[test_idx]
y_train = labels.loc[train_idx]
y_test = labels.loc[test_idx]

y_test = (y_test > 0).astype(int)
y_train = (y_train > 0).astype(int)

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(X_train,y_train)

distances_train, _ = knn.kneighbors(X_train)
distances_test, _ = knn.kneighbors(X_test)

anomaly_scores_train = distances_train[:, -1]
anomaly_scores_test = distances_test[:, -1]

threshold = np.percentile(anomaly_scores_train,70)  # 70th percentile for 0.3 contamination
y_pred = (anomaly_scores_test > threshold).astype(int)

comparison_df = pd.DataFrame({
    'ID': features.loc[test_idx, 'ID'],
    'Tardiness': features.loc[test_idx, 'Tardiness'],
    'Overall_processing_time': features.loc[test_idx, 'Overall_processing_time'],
    'Predicted_Anomaly': y_pred,
    'Actual_Break': y_test,
    'Correct_Prediction': y_pred == y_test,
    'Anomaly_Score': anomaly_scores_test
})

print("\nAccuracy Analysis:")
print("-" * 50)
correct_normals = comparison_df[(comparison_df['Predicted_Anomaly'] == 0) & (comparison_df['Actual_Break'] == 0)].shape[0]
correct_anomalies = comparison_df[(comparison_df['Predicted_Anomaly'] == 1) & (comparison_df['Actual_Break'] == 1)].shape[0]

total_normals = (y_test == 0).sum()
total_anomalies = (y_test == 1).sum()

print(f"\nCorrectly identified normal cases: {correct_normals}/{total_normals} " f"({correct_normals/total_normals*100:.2f}%)")
print(f"Correctly identified anomalies: {correct_anomalies}/{total_anomalies} " f"({correct_anomalies/total_anomalies*100:.2f}%)")


comparison_df.to_csv('anomaly_detection_results_knn.csv', index=False)

print("\nPerformance Metrics:")
print("-" * 50)
f1 = f1_score(y_test, y_pred, average='binary')
precision = precision_score(y_test, y_pred, average='binary', zero_division=1)
recall = recall_score(y_test, y_pred, average='binary')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Confusion Matrix:\n{conf_matrix}')


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

normal_points = X_test[y_pred == 0]
ax.scatter(normal_points.iloc[:, 0], normal_points.iloc[:, 1], normal_points.iloc[:, 2], c='blue', label='Normal')

anomalous_points = X_test[y_pred == 1]
ax.scatter(anomalous_points.iloc[:, 0], anomalous_points.iloc[:, 1], anomalous_points.iloc[:, 2], c='red', label='Anomalies')

ax.set_xlabel('ID')
ax.set_ylabel('Tardiness')
ax.set_zlabel('Overall Processing Time')
ax.set_title('Clusters and Anomalies in 3D')

ax.legend()

plt.show()