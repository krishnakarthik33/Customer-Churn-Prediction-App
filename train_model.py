import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Churn_Modelling.csv")

# Drop unwanted columns
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# -----------------------------------
# ADD MULTIPLE COUNTRIES
# -----------------------------------
new_countries = [
    "India", "USA", "Canada", "Australia",
    "Brazil", "UK", "UAE", "Singapore"
]

for country in new_countries:
    sample = df.sample(300, random_state=42).copy()
    sample["Geography"] = country
    df = pd.concat([df, sample], ignore_index=True)

# -----------------------------------
# Encode Gender
# -----------------------------------
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])

# -----------------------------------
# One Hot Encode Geography
# -----------------------------------
df = pd.get_dummies(df, columns=["Geography"])

# Save column order
feature_columns = df.drop("Exited", axis=1).columns

# -----------------------------------
# Split
# -----------------------------------
X = df.drop("Exited", axis=1)
y = df["Exited"]

# -----------------------------------
# Scale
# -----------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------
# Train Model
# -----------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_scaled, y)

# -----------------------------------
# Save everything
# -----------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le_gender, open("encoder.pkl", "wb"))
pickle.dump(feature_columns, open("features.pkl", "wb"))

print("🔥 Advanced Model trained successfully!")
