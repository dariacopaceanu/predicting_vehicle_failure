from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_subset)
X_test_scaled = scaler.transform(X_test_subset)

svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_model.fit(X_train_scaled, y_train)

y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"SVM accuracy (feature set 2): {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_, xticks_rotation=45)
plt.title("Confusion Matrix - SVM (feature set 2)")
plt.tight_layout()
plt.savefig("confusion_matrix_svm_set2.png")
plt.show()