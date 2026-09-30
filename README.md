# Vehicle Failure Prediction

A machine learning project focused on predicting the mileage range before a vehicle defect occurs. The main goal of the project was to learn and compare two different classification approaches: **Support Vector Machines (SVM)** and **XGBoost**.

## Overview

The project treats vehicle failure prediction as a multiclass classification problem, using vehicle characteristics and sensor measurements to predict `KM_Range_Before_Defect`.

Rather than focusing only on achieving the highest accuracy, this project was also an opportunity to understand **how different machine learning algorithms work and how their underlying approaches affect their results**.

The two models were tested using three feature configurations:

- All available features
- 8 selected features
- 10 selected features

This made it possible to observe how both the **algorithm** and the **input features** influence the predictions.

## Dataset

- 12,000 samples
- 15 input features
- Target: `KM_Range_Before_Defect`
- Multiclass classification

The dataset was obtained from Kaggle:

[Car Sensor Dataset](https://www.kaggle.com/datasets/lalit7881/car-sensor-dataset)

## Data Preprocessing

The dataset is prepared in `preprocessing.py`.

The preprocessing includes:

- Removing `Remaining_KM` to prevent data leakage
- Encoding categorical variables
- One-hot encoding `Vehicle_Type`
- Encoding the target using `LabelEncoder`
- Splitting the data into 80% training and 20% testing data
- Using `random_state=42` for reproducibility

For SVM, the features are additionally standardized using `StandardScaler`.

## SVM

SVM (Support Vector Machine) works by finding decision boundaries that separate different classes. With the RBF kernel used in this project, the model can create non-linear decision boundaries when the classes cannot be separated effectively by a straight line.

One of the things I wanted to understand through this project was how **feature scaling and the distribution of the input features affect a distance-based model such as SVM**.

The model uses:

```text
kernel = RBF
C = 1.0
gamma = scale
```

## XGBoost

XGBoost takes a different approach. Instead of finding a single decision boundary, it builds an ensemble of decision trees sequentially. Each new tree attempts to correct errors made by the previous trees.

This makes XGBoost particularly interesting to compare with SVM because the two models learn patterns in fundamentally different ways.

The model uses:

```text
n_estimators = 150
learning_rate = 0.1
max_depth = 7
```

Unlike SVM, XGBoost does not require feature scaling.

## Feature Sets

Three configurations were tested to see how feature selection affects the models.

### Feature Set 1

8 features:

```text
Vehicle_Age
Mileage
Engine_Temp
Oil_Pressure
Last_Service_Months
Accident_History
Tire_Condition
Vibration_Level
```

### Feature Set 2

10 features:

```text
Vibration_Level
Last_Service_Months
Tire_Condition
Engine_Temp
Oil_Pressure
Fuel_Consumption
Accident_History
Vehicle_Type_SUV
Vehicle_Type_Sedan
Vehicle_Type_Truck
```

The purpose of using multiple feature sets was to see whether additional information actually improves the predictions, or whether a smaller set of relevant features can produce similar results.

## Results

Each model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

| Model | Features | Accuracy |
|---|---|---:|
| SVM | All | 71.96% |
| SVM | Set 1 | 68.21% |
| SVM | Set 2 | 77.04% |
| XGBoost | All | 95.58% |
| XGBoost | Set 1 | 78.75% |
| XGBoost | Set 2 |  96.42% |

The experiments show that the choice of both model and features had a significant impact on the results. XGBoost performed substantially better than SVM on this dataset, while Feature Set 2 produced the highest accuracy for both models.

## Conclusion

The results show that XGBoost achieved the highest accuracy with Feature Set 2, reaching **96.42%**, compared with **95.58%** using all features. This suggests that the features included in Feature Set 2 contain most of the information needed to distinguish between the target classes, while some of the additional features do not provide enough useful information to improve the model.

This can also be explained by how XGBoost works. It builds a sequence of decision trees, where each new tree focuses on correcting the errors made by the previous ones. This allows the model to capture non-linear relationships and interactions between features. Feature Set 2 contains variables such as `Vibration_Level`, `Last_Service_Months`, `Tire_Condition`, `Engine_Temp`, `Oil_Pressure`, `Fuel_Consumption`, `Accident_History`, and `Vehicle_Type`, which can provide useful combinations of information for the tree-based model.

The comparison with SVM was also useful for understanding the difference between the two approaches. SVM achieved **77.04%** with Feature Set 2, while XGBoost reached **96.42%**. This project helped me understand that model performance depends not only on the number of features, but also on how the algorithm is able to use relationships between those features.

## What I Learned

The main purpose of this project was not only to build a prediction model, but to better understand two fundamentally different machine learning approaches.

Through the project, I learned:

- How SVM uses decision boundaries to separate classes.
- Why feature scaling is important for SVM.
- How the RBF kernel allows SVM to model non-linear relationships.
- How XGBoost combines multiple decision trees.
- How boosting works by progressively correcting previous errors.
- Why XGBoost does not require feature scaling in the same way as SVM.
- How feature selection can affect model performance.
- How confusion matrices can reveal errors that accuracy alone does not show.
- How data leakage can lead to misleading model performance.

## Project Structure

```text
Vehicle-Failure-Prediction

	preprocessing.py

	train_svm_all.py
	train_svm_set1.py
	train_svm_set2.py

	train_xgb_all.py
	train_xgb_set1.py
	train_xgb_set2.
	
	README.md
```

## Technologies

- Python
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib