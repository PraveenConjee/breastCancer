from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataset = pd.read_csv("data.csv")

# data = dataset.drop(columns=["Unnamed: 32"])
data = dataset.dropna()
data = data.drop(
    columns=["id", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
             "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst",
             "fractal_dimension_worst"])

scaler = StandardScaler()
encoder = LabelEncoder()
normalizer = MinMaxScaler()

data["diagnosis"] = encoder.fit_transform(data["diagnosis"])

numericalColumns = data.drop(columns=["diagnosis"]).columns
data[numericalColumns] = scaler.fit_transform(data[numericalColumns])
data[numericalColumns] = normalizer.fit_transform(data[numericalColumns])
data.to_csv("data_refined.csv", index=False)

sns.pairplot(data, hue="diagnosis", diag_kind="kde")
plt.show()


plt.figure(figsize=(14, 12))
correlationMatrix = data.corr()
sns.heatmap(correlationMatrix, annot=True, cmap="coolwarm")
plt.show()

plt.figure(figsize=(20, 15))
dataMelted = data.melt(id_vars="diagnosis", var_name="features", value_name="value")
sns.boxplot(x="features", y="value", hue="diagnosis", data=dataMelted)
plt.xticks(rotation=90)
plt.show()
