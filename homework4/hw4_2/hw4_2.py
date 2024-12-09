import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.classification import *

def load_and_prepare_data():
    """Load, prepare and engineer features for both train and test data"""
    print("Loading data...")
    data_train = pd.read_csv('./data/train.csv')
    data_test = pd.read_csv('./data/test.csv')
    test_ids = data_test['PassengerId'].copy()

    # Create train-validation split
    train_data, valid_data = train_test_split(
        data_train.drop(['PassengerId'], axis=1),
        random_state=100,
        train_size=0.8
    )

    def prepare_data(df):
        """Prepare and engineer features for a dataset"""
        data = df.copy()
        
        # Handle missing values
        data['Age'].fillna(data['Age'].median(), inplace=True)
        data['Fare'].fillna(data['Fare'].median(), inplace=True)
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
        data['Cabin'].fillna('Unknown', inplace=True)
        
        # Create features
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
        
        # Title extraction
        data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Ms': 'Miss', 'Lady': 'Rare', 'Sir': 'Rare',
            'Capt': 'Rare', 'Don': 'Rare', 'Jonkheer': 'Rare', 'Countess': 'Rare',
            'Mme': 'Mrs', 'Dona': 'Rare'
        }
        data['Title'] = data['Title'].map(title_mapping).fillna('Rare')
        
        # Fare binning
        data['FareBin'] = pd.qcut(data['Fare'].rank(method='first'), 
                                 q=4, 
                                 labels=['Low', 'Medium', 'High', 'VeryHigh'])
        
        # Drop unnecessary columns
        cols_to_drop = ['Name', 'Ticket']
        data.drop(cols_to_drop, axis=1, inplace=True)
        
        return data

    # Prepare all datasets
    train_processed = prepare_data(train_data)
    test_processed = prepare_data(data_test)

    return train_processed, test_processed, test_ids

def train_models(train_data):
    """Set up and train models"""
    print("\nSetting up PyCaret...")
    clf = setup(
        data=train_data,
        target='Survived',
        numeric_features=['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize'],
        categorical_features=['Pclass', 'Sex', 'Embarked', 'Title', 'FareBin'],
        ignore_features=['Cabin'],  # Ignoring Cabin for now as it's highly sparse
        session_id=123,
        verbose=False
    )

    print("\nComparing models...")
    best_model = compare_models(
        fold=5,
        include=[
            'lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 
            'ridge', 'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'lightgbm'
        ],
        sort='Accuracy'
    )

    print("\nCreating ensemble model...")
    ensemble_model = blend_models(
        estimator_list=[
            best_model,
            create_model('rf', verbose=False),
            create_model('gbc', verbose=False)
        ],
        fold=5,
        verbose=False
    )

    print("\nTuning ensemble model...")
    tuned_model = tune_model(
        ensemble_model,
        optimize='Accuracy',
        fold=5,
        n_iter=10,
        verbose=False
    )

    return tuned_model

def format_predictions(predictions, test_data):
    """Format the predictions with detailed results"""
    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.precision', 4)

    # Select columns for display
    display_cols = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
        'Embarked', 'FamilySize', 'IsAlone', 'Title', 'FareBin',
        'prediction_label', 'prediction_score'
    ]

    # Combine predictions with features
    detailed_results = predictions.copy()
    print("\nDetailed Prediction Results:")
    print(detailed_results[display_cols])

    return detailed_results[display_cols]

def main():
    # Load and prepare data
    train_data, test_data, test_ids = load_and_prepare_data()
    
    # Train models
    final_model = train_models(train_data)
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predict_model(final_model, data=test_data)
    
    # Format and display results
    detailed_results = format_predictions(predictions, test_data)
    
    # Save results
    detailed_results.to_csv('detailed_predictions.csv', index=False)
    
    # Create submission file
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions['prediction_label']
    })
    submission.to_csv('submission.csv', index=False)
    
    print("\nFiles saved:")
    print("- detailed_predictions.csv (detailed results)")
    print("- submission.csv (competition submission format)")
    
    return final_model, detailed_results

if __name__ == "__main__":
    final_model, detailed_results = main()