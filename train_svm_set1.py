from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from preprocessing import X_train, X_test, y_train, y_test, le

feature_set_1 = [
    "Vehicle_Age",
    "Mileage",
    "Engine_Temp",
    "Oil_Pressure",
    "Last_Service_Months",
    "Accident_History",
    "Tire_Condition",
    "Vibration_Level",
]

X_train_subset = X_train[feature_set_1]
X_test_subset = X_test[feature_set_1]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_subset)
X_test_scaled = scaler.transform(X_test_subset)

svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_model.fit(X_train_scaled, y_train)

y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"SVM accuracy (feature set 1): {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_, xticks_rotation=45)
plt.title("Confusion Matrix - SVM (feature set 1)")
plt.tight_layout()
plt.savefig("confusion_matrix_svm_set1.png")
plt.show()