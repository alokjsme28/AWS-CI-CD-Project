import os
import sys
from dataclasses import dataclass
import dill

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("split data into Test and Train.")
            X_train, y_train,X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression" : LinearRegression(),
                "Support Vector Machines" : SVR(),
                "Decision Tree" : DecisionTreeRegressor(),
                "K-Nearest Neighbor" : KNeighborsRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "Adaboost" : AdaBoostRegressor(),
                "XGBoost" : XGBRegressor(),
                "Catboost" : CatBoostRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor()
            }

            params = {

                "Linear Regression":{},

                "SVR" : {
                    'degree' : [2,3,4],
                    'kernel' : ['linear', 'poly', 'rbf']
                },

                "Decision Tree" : {
                    "criterion" : ['friedman_mse', 'absolute_error', 'poisson', 'squared_error'],
                    'max_depth' : [5,10,15,20],
                    'min_samples_split' : [2,5,8,10]
                },

                "K-Nearest Neighbor" : {
                    'n_neighbors' : [5,8,11],
                    'algorithm' : ['auto', 'ball_tree', 'kd_tree'],
                    'p' : [1,2]
                },

                "Random Forest" : {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },        
            }

            model_report : dict = evaluate_model(X_train,y_train,X_test,y_test,models,params)

            # To get the best model score from the report
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if(best_model_score < 0.6):
                raise CustomException("No best model found.")
            
            logging.info(f"Best found model on both training and testing dataset.")

            save_object(file_path= self.model_trainer_config.trained_model_file_path
                        , obj = best_model)
            
            predicted = best_model.predict(X_test)

            r2 = r2_score(y_test,predicted)

            return r2

        except Exception as ex:
            raise CustomException(ex,sys)
        
