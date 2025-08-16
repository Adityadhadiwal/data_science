import os
import sys
from dataclasses import dataclass

# Import regressors (some from sklearn, some external libs)
from catboost import CatBoostRegressor  # type: ignore # CatBoost library for gradient boosting with categorical support
from sklearn.ensemble import (           # ensemble methods from scikit-learn # type: ignore
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression  # type: ignore # simple linear model
from sklearn.metrics import r2_score                # type: ignore # to evaluate model performance
from sklearn.neighbors import KNeighborsRegressor   # (imported but not used in this snippet)
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor  # XGBoost, another powerful gradient boosting library

from src.exception import CustomException  # your project's custom exception wrapper
from src.logger import logging             # project logger for info/error messages

from src.utils import save_object, evaluate_models  # utility functions: save model, evaluate many models


# dataclass to hold where the trained model will be saved
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    # this is the path where the chosen (best) model will be saved (pickled)


class ModelTrainer:
    def __init__(self):
        # initialize configuration for model saving
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        """
        Main method to:
         - split arrays into X and y for train/test
         - define candidate models and parameter grids
         - run hyperparameter search and evaluate all models
         - pick the best model, save it, and return its R² on the test set
        """
        try:
            logging.info("Split training and test input data")

            # Split the combined arrays into features (X) and target (y)
            # train_array and test_array are expected to have features in all columns except last,
            # and the target value in the last column.
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # all columns except last -> training features
                train_array[:, -1],   # last column -> training target
                test_array[:, :-1],   # all columns except last -> testing features
                test_array[:, -1]     # last column -> testing target
            )

            # Define the candidate models to evaluate.
            # Keys are friendly names; values are the model instances.
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define hyperparameter grids for each model (used by GridSearchCV)
            # Each key matches a model name in the `models` dict.
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # other tree params (splitter/max_features) are commented out but could be added
                },
                "Random Forest": {
                    # tune number of trees (n_estimators) as an example
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # tune learning rate, subsample ratio and number of trees
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {
                    # empty dict means no hyperparameters to search for plain linear regression
                },
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # evaluate_models tries each model with its parameter grid using cross-validated grid search,
            # then fits the best parameters and returns a report (dict) mapping model name -> test R² score
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # model_report is expected to be like: {"Random Forest": 0.82, "Linear Regression": 0.4, ...}
            # Get the best R² score across models
            best_model_score = max(sorted(model_report.values()))
            # Get the model name corresponding to that best score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            # Retrieve the actual model instance from our models dictionary
            best_model = models[best_model_name]

            # If the best model does not meet a minimum quality threshold, raise an exception
            if best_model_score < 0.6:
                # 0.6 is an arbitrary threshold chosen here — change as needed
                raise CustomException("No best model found with acceptable score")

            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            # Persist (save) the best model object to disk so it can be loaded later for inference
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Use the chosen best model to predict on the test features
            predicted = best_model.predict(X_test)

            # Compute R² on the test set and return it (this is the final performance metric)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            # Wrap any exception in your project's CustomException (keeps consistent error handling)
            raise CustomException(e, sys)
