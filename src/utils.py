import os
import sys

import numpy as np 
import pandas as pd
import dill  # (not used here, but usually for saving Python objects)
import pickle  # for saving/loading Python objects
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException  # custom exception handler

# ----------------- Function to Save an Object -----------------
def save_object(file_path, obj):
    try:
        # Get the folder path from the file path
        dir_path = os.path.dirname(file_path)

        # Create the folder if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Open file in write-binary mode and save the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # If something goes wrong, raise a custom exception with full details
        raise CustomException(e, sys)
    

# ----------------- Function to Evaluate Multiple Models -----------------
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}  # dictionary to store model name → R² score mapping

        # Loop over all models
        for i in range(len(list(models))):
            # Get the model instance
            model = list(models.values())[i]

            # Get the parameter grid for this model
            para = param[list(models.keys())[i]]

            # GridSearchCV will try different parameter combinations (cv=3 means 3-fold cross-validation)
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)  # train model with different parameters

            # Update model with the best found parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # train again with best params

            # Predict on training data
            y_train_pred = model.predict(X_train)

            # Predict on testing data
            y_test_pred = model.predict(X_test)

            # Calculate R² score for train set
            train_model_score = r2_score(y_train, y_train_pred)

            # Calculate R² score for test set
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test score in the report dictionary with model name as key
            report[list(models.keys())[i]] = test_model_score

        # Return dictionary: { "model_name": r2_score_on_test, ... }
        return report

    except Exception as e:
        raise CustomException(e, sys)
    

# ----------------- Function to Load an Object -----------------
def load_object(file_path):
    try:
        # Open file in read-binary mode and load object using pickle
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
