from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from preprocessing import X_train, X_test, y_train, y_test, le

feature_set_2 = [
    "Vibration_Level",
    "Last_Service_Months",
    "Tire_Condition",
    "Engine_Temp",
    "Oil_Pressure",
    "Fuel_Consumption",
    "Accident_History",
    "Vehicle_Type_SUV",
    "Vehicle_Type_Sedan",
    "Vehicle_Type_Truck",
]

X_train_subset = X_train[feature_set_2]
X_test_subset = X_test[feature_set_2]

xgb_model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=7,
    random_state=42,
    eval_metric="mlogloss",
)
xgb_model.fit(X_train_subset, y_train)

y_pred = xgb_model.predict(X_test_subset)
accuracy = accuracy_score(y_test, y_pred)

print(f"XGBoost accuracy (feature set 2): {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_, xticks_rotation=45)
plt.title("Confusion Matrix - XGBoost (feature set 2)")
plt.tight_layout()
plt.savefig("confusion_matrix_xgb_set2.png")
plt.show()