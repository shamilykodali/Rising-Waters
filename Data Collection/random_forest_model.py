# Project: Rising Waters - Model Building
# Milestone: Random Forest Classifier Implementation

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def random_forest(X_train, y_train, X_test, estimators=100, random_state=42):
    """
    Initializes, trains, and evaluates a Random Forest Classifier model.
    """
    print("--- 1. Initializing Random Forest Classifier ---")
    # Initialize the classifier with specified hyperparameters
    model = RandomForestClassifier(n_estimators=estimators, random_state=random_state)
    print(f"Model configured with {estimators} decision trees.")
    
    print("\n--- 2. Model Training ---")
    # Train the model on training data
    model.fit(X_train, y_train)
    print("Model training completed successfully.")
    
    print("\n--- 3. Prediction ---")
    # Generate predicted outcomes on test dataset
    y_pred = model.predict(X_test)
    print("Predictions generated on test features.")
    
    return model, y_pred

def evaluate_model(y_test, y_pred):
    """
    Calculates and prints model performance metrics.
    """
    print("\n--- 4. Model Evaluation ---")
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Display results to console
    print(f"Accuracy Score: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    return accuracy, cm, report

# Blueprint initialized for project deployment tracking
print("Random Forest machine learning template structured successfully.")

