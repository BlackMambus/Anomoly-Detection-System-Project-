import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=5, size=1000)
anomalies = np.random.normal(loc=80, scale=2, size=20)
data = np.concatenate([normal_data, anomalies])

df = pd.DataFrame(data, columns=["value"])
sns.histplot(df["value"], bins=50, kde=True)
plt.title("Data Distribution")
plt.show()
from sklearn.ensemble import IsolationForest

# Fit the model
model = IsolationForest(contamination=0.02)  # 2% anomalies
df["anomaly"] = model.fit_predict(df[["value"]])

# Mark anomalies
df["is_anomaly"] = df["anomaly"] == -1

# Visualize
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df.index, y="value", hue="is_anomaly", data=df, palette={True: "red", False: "blue"})
plt.title("Anomaly Detection with Isolation Forest")
plt.show()
def detect_anomalies(new_values):
    new_df = pd.DataFrame(new_values, columns=["value"])
    new_df["anomaly"] = model.predict(new_df[["value"]])
    return new_df[new_df["anomaly"] == -1]

# Example usage
new_data = [48, 52, 81, 49, 85]
anomalies_found = detect_anomalies(new_data)
print("Detected anomalies:\n", anomalies_found)

