from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from preprocessing import X_train, X_test, y_train, y_test, le

xgb_model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=7,
    random_state=42,
    eval_metric="mlogloss",
)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"XGBoost accuracy (all features): {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_, xticks_rotation=45)
plt.title("Confusion Matrix - XGBoost (all features)")
plt.tight_layout()
plt.savefig("confusion_matrix_xgb_all.png")
plt.show()