# Vehicle Failure Prediction

## Overview

	This project aims to predict the likehood of vehicle failure based on vehicle data and sensor measurements. It is a classification project that predicts the mileage range in which the vehicle is likely to experience failure.
	

## Objectives

	- Predict vehicle failures using machine-learning algorithms
	- Preprocess and clean the raw dataset
	- Compare SVM and XGBoost models
	- Understand how different classification models work and why one model may perform better than another when using additional features


## Dataset

	- 12000 samples
	- 15 input features and 1 target variable
	- target variable: KM_Range_Before_Defect

	The dataset was obtained from Kaggle: https://www.kaggle.com/datasets/lalit7881/car-sensor-dataset?resource=download


## Project structure

	- preprocessing.py  ->  prepare dataset for training and testing the models; feature engineering; categorical encoding; feature scaling

	- train_svm_all.py  ->  training SVM model with all features
	- training_svm_set1  ->  treining SVM model with 8 features
	- training_svm_set2  ->  training SVM model with 10 features

	- training_xgb_all.py  -> training XGBoost whith all features
	- training_xgb_set1  ->  treining XGBoost model with 8 features
	- training_xgb_set2  ->  training XGBoost model with 10 features

## Data Preprocessing

	