import pandas as pd
import gc
from sklearn.model_selection import train_test_split
import optuna
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error 

import lightgbm as lgbm
import joblib
import warnings
warnings.filterwarnings('ignore')
from s3_utils import read_pd_from_parquet, write_pickle, read_pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna 
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna import Trial, visualization
from optuna.samplers import TPESampler


import multiprocessing
import time
import warnings
from tempfile import mkdtemp

import joblib
import mlflow
import pandas as pd

from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from psutil import virtual_memory
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from termcolor import colored

import datetime

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed


def objective(trial, X_train, X_test, y_train, y_test):
    param = {
        'metric': 'mse',
        'random_state': 42,
        'n_estimators': trial.suggest_categorical('n_estimators', [500]),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006, 0.01, 0.02, 0.05, 0.1, 0.15]),
        'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7, 8, 9, 10, 15]),
        'num_leaves': trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth': trial.suggest_int('min_data_per_groups', 1, 100),
        'verbosity': -1
    }
    model = lgbm.LGBMRegressor(**param)

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgbm.early_stopping(stopping_rounds=20)])
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds, squared=False)

    return mse

from optuna.integration.mlflow import MLflowCallback

class LigthTrainer:
    mlflow.start_run(nested=True)
    EXPERIMENT_NAME = f"IncomeModel_{datetime.datetime.now()}"
    """
        :param X: Data del Bureau de Crédito
        :param y: Ingreso comprobado
        :param kwargs:
        """
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.mlflow = kwargs.get("mlflow", True) 
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME) 
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.best_params = None
        self.model = None
        self.preds = None
        self.ids = None
        self.columns = None
        self.target = None
        self.log_machine_specs()
        self.log_kwargs_params()
        

    def fit(self, df,model_name="Income" ,to_drop=['researchable_id', 'estimate', 'declarativa'], target="ingreso_neto_comprobado"):
        self.ids = df[['researchable_id', 'estimate', 'declarativa']].copy()
        date = datetime.datetime.now()
        self.mlflow_log_tag(model_name, target,date)
        to_drop = to_drop + [target]
        columnas = [col for col in df.columns if col not in to_drop]
        self.columns = columnas
        self.target = target
        X = df[columnas]
        y = df[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    def optimize(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        func = lambda trial: objective(trial,self.X_train,self.X_test,self.y_train,self.y_test)
        study.optimize(func, n_trials=25)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        self.best_params = study.best_params
        print("  Value: {}".format(trial.value))
        print("  Params: ")

        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        lgbm_model = lgbm.LGBMRegressor(**self.best_params)
        self.model = lgbm_model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)],callbacks=[lgbm.early_stopping(stopping_rounds=50)])
        
    @simple_time_tracker
    def train(self):
        tic = time.time()
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)
        # mlflow logs
        self.mlflow_log_metric("train_time", int(time.time() - tic))

    def evaluate(self):
        rmse_train = self.compute_mape(self.X_train, self.y_train)
        self.mlflow_log_metric("mape_train", rmse_train)
        rmse_val = self.compute_mape(self.X_test, self.y_test, show=True)
        self.mlflow_log_metric("mape_val", rmse_val)
        print(colored("mape train: {} || mape val: {}".format(rmse_train, rmse_val), "blue"))


    def compute_mape(self, X, Y, show=True):
        preds = self.model.predict(X)
        mape = MAPE(Y, preds)
        mlflow.log_params(self.best_params)
        return round(mape, 3)

    def save_model(self,name):
        self.preds = self.model.predict(self.X_test)
        """Save the model into a .pickle format"""
        write_pickle( path_s3 + f"{name}.pkl", self.model)
        print(colored(f"{name}.pkl saved in {path_s3}", "green"))
        mlflow.end_run()

    ### MLFlow methods
    @memoized_property
    def mlflow_client(self):
        #mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
            
    def mlflow_optuna(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
    
    def mlflow_log_tag(self, model_name, target,date):
        if self.mlflow:
             self.mlflow_client.set_tag(self.mlflow_run.info.run_id,f"{model_name}_{target}", f"{date}")
           
            
             


    def log_estimator_params(self):
        reg = self.model
        self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        params = reg.get_params()
        for k, v in params.items():
            self.mlflow_log_param(k, v)

    def log_kwargs_params(self):
        if self.mlflow:
            for k, v in self.kwargs.items():
                self.mlflow_log_param(k, v)

    def log_machine_specs(self):
        cpus = multiprocessing.cpu_count()
        mem = virtual_memory()
        ram = int(mem.total / 1000000000)
        self.mlflow_log_param("ram", ram)
        self.mlflow_log_param("cpus", cpus)


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    experiment = "IncomeModel"
    print(colored("############  Loading data   ############", "yellow"))
    prospectos = pd.read_csv("/data/prospectos.csv", index_col=0)
    aprobados = pd.read_csv("/data/aprobados.csv", usecols=prospectos.columns)
    df = pd.concat([aprobados, prospectos], ignore_index=True, sort=False)
    del aprobados
    del prospectos
    df = df.drop_duplicates(subset="researchable_id", keep="last")
    df = df.query("ingreso_neto_comprobado > 8000 & ingreso_neto_comprobado < 300000")
    print("shape: {}".format(df.shape))
    print("size: {} Mb".format(df.memory_usage().sum() / 1e6))
    # Train and save model
    trainer = LigthTrainer()
    trainer.fit(df,to_drop=['researchable_id', 'estimate', 'declarativa'])
    print(colored("############  Training model   ############", "red"))
    trainer.optimize()
    print(colored("############  Evaluating model ############", "blue"))
    trainer.evaluate()
    print(colored("############   Saving model    ############", "green"))
    trainer.save_model()