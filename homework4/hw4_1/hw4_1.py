import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.classification import *

def load_and_prepare_data():
    """Load and prepare the training and test datasets"""
    print("Loading data...")
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    
    # Store test PassengerId for later use
    test_passenger_ids = test['PassengerId'].copy()
    
    # Remove PassengerId and split training data
    train_data, valid_data = train_test_split(
        train.drop(['PassengerId'], axis=1),
        random_state=100,
        train_size=0.8
    )
    
    return train_data, valid_data, test, test_passenger_ids

def create_model_comparison():
    """Set up PyCaret environment and compare models"""
    # Initialize setup
    clf1 = setup(
        data=train_data,
        target='Survived',
        numeric_features=['Age', 'Fare', 'SibSp', 'Parch'],
        categorical_features=['Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked'],
        ignore_features=['Name'],
        session_id=123,
        verbose=False
    )
    
    # Display available models
    print("\nAvailable models:")
    print(models())
    
    # Define models to include
    model_list = [
        'lr',      # Logistic Regression
        'knn',     # K Neighbors Classifier
        'nb',      # Naive Bayes
        'dt',      # Decision Tree Classifier
        'svm',     # SVM - Linear Kernel
        'rbfsvm',  # SVM - Radial Kernel
        'gpc',     # Gaussian Process Classifier
        'mlp',     # MLP Classifier
        'ridge',   # Ridge Classifier
        'rf',      # Random Forest Classifier
        'qda',     # Quadratic Discriminant Analysis
        'ada',     # Ada Boost Classifier
        'gbc',     # Gradient Boosting Classifier
        'lda',     # Linear Discriminant Analysis
        'et',      # Extra Trees Classifier
        'lightgbm' # Light Gradient Boosting Machine
    ]
    
    print("\nComparing models...")
    best_model = compare_models(
        fold=5,
        include=model_list,
        verbose=True
    )
    
    return best_model

def generate_predictions(model, test_data, test_ids):
    """Generate predictions using the best model"""
    print("\nGenerating predictions...")
    predictions = predict_model(model, data=test_data)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions['prediction_label']
    })
    
    # Save predictions
    submission.to_csv('model_predictions.csv', index=False)
    print("\nPredictions saved to 'model_predictions.csv'")
    
    return predictions

def main():
    # Load and prepare data
    global train_data, valid_data, test_data, test_ids
    train_data, valid_data, test_data, test_ids = load_and_prepare_data()
    
    # Create and compare models
    best_model = create_model_comparison()
    
    # Generate predictions
    predictions = generate_predictions(best_model, test_data, test_ids)
    
    # Print model performance on validation data
    print("\nBest model performance metrics:")
    print(pull())
    
    return {
        'best_model': best_model,
        'predictions': predictions
    }

if __name__ == "__main__":
    results = main()