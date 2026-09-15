import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("exactkm.csv")

# Drop numeric equivalent of the target to prevent data leakage
df = df.drop(columns=["Remaining_KM"])

tire_map = {"Critical": 0, "Worn": 1, "Moderate": 2, "Good": 3}
vibration_map = {"Low": 0, "Medium": 1, "High": 2}
accident_map = {"No": 0, "Yes": 1}

df["Tire_Condition"] = df["Tire_Condition"].map(tire_map)
df["Vibration_Level"] = df["Vibration_Level"].map(vibration_map)
df["Accident_History"] = df["Accident_History"].map(accident_map)

df = pd.get_dummies(df, columns=["Vehicle_Type"], drop_first=True)

X = df.drop(columns=["KM_Range_Before_Defect"])
y = df["KM_Range_Before_Defect"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)