# Iris Flower Classification

This project focuses on classifying the species of iris flowers based on their physical attributes (sepal length, sepal width, petal length, petal width) using various Machine Learning models. The dataset used for this classification task is the famous Iris dataset.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Steps](#project-steps)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Correlation Analysis](#2-correlation-analysis)
  - [3. Model Training & Evaluation](#3-model-training--evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

This project aims to classify the species of iris flowers (Iris-setosa, Iris-versicolor, and Iris-virginica) using a variety of machine learning models such as Logistic Regression, K-Nearest Neighbors (KNN), and Decision Tree. The dataset contains 150 samples with 4 features (sepal length, sepal width, petal length, petal width) and a target variable specifying the species.

## Dataset

The Iris dataset is a well-known dataset in the field of machine learning and statistics. It contains 150 samples of iris flowers, with three different species: **Iris-setosa**, **Iris-versicolor**, and **Iris-virginica**. The dataset consists of the following columns:
- `sepal_length`: Sepal length in cm
- `sepal_width`: Sepal width in cm
- `petal_length`: Petal length in cm
- `petal_width`: Petal width in cm
- `species`: The species of iris flowers

## Project Steps

### 1. Exploratory Data Analysis (EDA)
In this step, we explore the dataset with the following:
- Summary statistics using `describe()` and data information using `info()`.
- Histograms for each of the four features: `sepal_length`, `sepal_width`, `petal_length`, and `petal_width`.
- Scatterplots to visualize the relationships between different features.
- Checking for missing values and handling them if needed (although the Iris dataset does not contain missing values).

### 2. Correlation Analysis
A correlation matrix is generated to analyze the relationships between the numerical features (sepal length, sepal width, petal length, and petal width) and understand their interdependencies.

### 3. Model Training & Evaluation
The dataset is split into training (70%) and testing (30%) sets, and the following machine learning models are trained and evaluated:
- **Logistic Regression**: A statistical model used for binary classification, extended here for multiclass classification.
- **K-Nearest Neighbors (KNN)**: A simple, instance-based learning algorithm that classifies a data point based on the majority class of its neighbors.
- **Decision Tree**: A tree-like model that splits the data based on feature values to make classification decisions.

Each model's accuracy is computed using the test set to evaluate its performance.

## Results
- **Logistic Regression Accuracy**: Achieved a classification accuracy of approximately **X%** on the test set.
- **K-Nearest Neighbors Accuracy**: Achieved a classification accuracy of approximately **Y%** on the test set.
- **Decision Tree Accuracy**: Achieved a classification accuracy of approximately **Z%** on the test set.

## Conclusion
The project showcases the effectiveness of simple machine learning models in classifying iris species. Decision Trees, K-Nearest Neighbors, and Logistic Regression performed well on this dataset, and the project highlights the importance of understanding feature relationships through visualization and correlation analysis.

## Requirements
To run this project, ensure you have the following libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`