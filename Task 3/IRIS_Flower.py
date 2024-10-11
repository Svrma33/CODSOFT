# Importing necessary libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
    # Load the dataset
    df = pd.read_csv('IRIS.csv')
    
    # Display basic information
    print(df.head())
    print(df.tail())
    print(df.describe())
    print(df.info())
    
    # Display the number of samples per species class
    print(df['species'].value_counts())

    # Check for null values
    print(df.isnull().sum())

    # Plot histograms
    plt.figure(figsize=(10, 6))
    df['sepal_length'].hist(color='darkred', edgecolor='white')
    plt.title('Sepal Length Distribution')
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 600), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
    plt.show()

    plt.figure(figsize=(10, 6))
    df['sepal_width'].hist(color='darkorange', edgecolor='white')
    plt.title('Sepal Width Distribution')
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 600), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
    plt.show()

    plt.figure(figsize=(10, 6))
    df['petal_length'].hist(color='darkcyan', edgecolor='white')
    plt.title('Petal Length Distribution')
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 600), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
    plt.show()

    plt.figure(figsize=(10, 6))
    df['petal_width'].hist(color='purple', edgecolor='white')
    plt.title('Petal Width Distribution')
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 600), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
    plt.show()

    # Scatterplots
    colors = ['green', 'blue', 'orange']
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    plt.figure(figsize=(10, 6))

    for i in range(3):
        x = df[df['species'] == species[i]]
        plt.scatter(x['sepal_length'], x['sepal_width'], c=colors[i], label=species[i])
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 600), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
    plt.show()

    plt.figure(figsize=(10, 6))
    for i in range(3):
        x = df[df['species'] == species[i]]
        plt.scatter(x['petal_length'], x['petal_width'], c=colors[i], label=species[i])
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend()
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 600), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
    plt.show()

    plt.figure(figsize=(10, 6))
    for i in range(3):
        x = df[df['species'] == species[i]]
        plt.scatter(x['sepal_length'], x['petal_length'], c=colors[i], label=species[i])
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')
    plt.legend()
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 600), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
    plt.show()

    plt.figure(figsize=(10, 6))
    for i in range(3):
        x = df[df['species'] == species[i]]
        plt.scatter(x['sepal_width'], x['petal_width'], c=colors[i], label=species[i])
    plt.xlabel('Sepal Width')
    plt.ylabel('Petal Width')
    plt.legend()
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 600), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
    plt.show()

    # Correlation matrix (excluding the 'species' column)
    corr = df.drop(columns=['species']).corr()

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, ax=ax, cmap='icefire')
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 600), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
    plt.show()

    # Label encoding
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    print(df.head())

    # Train-test split (70% train, 30% test)
    X = df.drop(columns=['species'])
    Y = df['species']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

    # Logistic Regression
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    print('Logistic Regression Accuracy:', model.score(X_test, Y_test) * 100)

    # K-Nearest Neighbours
    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    print('K-Nearest Neighbours Accuracy:', model.score(X_test, Y_test) * 100)

    # Decision Tree
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    print('Decision Tree Accuracy:', model.score(X_test, Y_test) * 100)

if __name__ == '__main__':
    main()
