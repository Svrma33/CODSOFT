# Titanic Survival Prediction

This project is focused on predicting passenger survival on the Titanic using Logistic Regression. The dataset used for this analysis is the Titanic dataset available on [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset). The project includes various steps such as data preprocessing, data visualization, correlation analysis, and model training.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Steps](#project-steps)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Data Visualization](#2-data-visualization)
  - [3. Correlation Analysis](#3-correlation-analysis)
  - [4. Model Training & Evaluation](#4-model-training--evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

This project aims to analyze and predict survival of passengers aboard the Titanic using various Machine Learning techniques. Logistic Regression is the primary model used, and the project also features extensive data visualization and correlation analysis.

## Dataset

The dataset used in this project is the [Titanic Dataset from Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset), containing information about passengers like name, age, gender, class, fare, and whether they survived or not.

## Project Steps

### 1. Data Preprocessing
- Filling missing values in the `Age` column with the average age.
- Dropping unnecessary columns like `Cabin`, `PassengerId`, `Name`, and `Ticket`.
- Encoding categorical variables like `Sex` and `Embarked`.

### 2. Data Visualization
Several plots are used to visualize the dataset and understand the relationships between variables:
- Survival counts
- Survival counts by gender
- Survival counts by passenger class
- Histograms for numerical variables (`Age`, `Fare`)

### 3. Correlation Analysis
A correlation matrix is generated to analyze relationships between numerical variables and gain insights on how different features relate to survival.

### 4. Model Training & Evaluation
- A Logistic Regression model is trained using the preprocessed dataset.
- Accuracy scores are generated for both training and test sets.
- A confusion matrix and classification report provide detailed evaluation metrics.

## Results
The logistic regression model provides insight into the survival probabilities of passengers based on their features. The project highlights key factors such as gender, age, and class as important indicators of survival.

## Conclusion
This project demonstrates how data analysis and machine learning techniques can be applied to make predictions about passenger survival on the Titanic. The Logistic Regression model performs reasonably well, offering an accuracy score that indicates its ability to classify survival outcomes.

## Requirements
To run this project, ensure you have the following libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`