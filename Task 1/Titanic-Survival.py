# Importing necessary libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

def load_data(file_path):
    # Read the dataset
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Data preprocessing
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df.drop(columns='Cabin', inplace=True)
    
    # Dropping unnecessary columns
    df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
    
    # Label encoding for categorical features
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'].fillna('S'))
    # Male =1 , Female =0
    return df

def visualize_data(df):
    # Visualization of data
    
    # Survival Counts
    Survived = df['Survived'].value_counts().reset_index()
    Survived.columns = ['Survived', 'Count']
    plt.figure(figsize=(8, 6))
    plt.bar(Survived['Survived'], Survived['Count'], color=['darkred', 'orange'])
    plt.xticks(Survived['Survived'])
    plt.title('Survival Counts')
    plt.xlabel('Survived')
    plt.ylabel('Count')
    plt.tight_layout()
    
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 500), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
    plt.show()

    # Survival by gender
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Survived', hue='Sex', data=df)
    plt.title('Survival Counts by Gender')
    plt.xlabel('Survived')
    plt.ylabel('Count')
    plt.tight_layout()
    
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 500), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
    plt.show()

    # Survival by class
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Survived', hue='Pclass', data=df)
    plt.title('Survival Counts by Pclass')
    plt.xlabel('Survived')
    plt.ylabel('Count')
    plt.tight_layout()

    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 500), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
    plt.show()

    # Histogram for numerical variables
    histogram_labels = ['Fare', 'Age']
    for label in histogram_labels:
        plt.figure(figsize=(8, 6))
        plt.hist(df[label], bins=30, color='darkcyan', edgecolor='black')
        plt.title(f'{label} Distribution')
        plt.xlabel(label)
        plt.ylabel('Frequency')
        plt.tight_layout()

        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 500), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
        plt.show()

def train_logistic_regression(df):
    # Preparing data for logistic regression
    X = df.drop(columns='Survived')
    Y = df['Survived']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=43)
    
    # Building logistic regression model
    logmodel = LogisticRegression(max_iter=5000, C=0.1)
    logmodel.fit(X_train, Y_train)
    
    # Prediction and evaluation
    y_pred = logmodel.predict(X_test)
    print(f"Train Accuracy: {logmodel.score(X_train, Y_train):.2f}")
    print(f"Test Accuracy: {logmodel.score(X_test, Y_test):.2f}")
    
    #Model Evaluations

    # Confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(Y_test, y_pred))

def plot_correlation_matrix(df):
    corrcolumns = df.drop(columns=['Embarked', 'Sex'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(corrcolumns.corr(), annot=True, cmap='icefire', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()

    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+{}+{}".format(int(manager.window.winfo_screenwidth()/2 - 500), 
                                           int(manager.window.winfo_screenheight()/2 - 450)))
    plt.show()

if __name__ == "__main__":
    file_path = 'Titanic-Dataset.csv'
    
    # Load and preprocess data
    df = load_data(file_path)
    df = preprocess_data(df)
    
    # Visualize data
    visualize_data(df)
    
    # Plot correlation matrix
    plot_correlation_matrix(df)
    
    # Train logistic regression model
    train_logistic_regression(df)