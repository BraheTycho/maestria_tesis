#!pip install lightgbm
# !pip install optuna
import lightgbm as lgbm

import pandas as pd

import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error as mape
import dask.dataframe as dd
from s3_utils import read_pd_from_parquet, write_pickle, read_pickle
import logging
import sys
from s3_utils import write_parquet_from_pd

try:
    from awsglue.utils import getResolvedOptions

    args = getResolvedOptions(sys.argv, ["path"])
    path = args["path"]
except Exception as error:
    print("Running script locally")
    path = "glue_scripts/output"


logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logFormatter = logging.Formatter(
    "%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
)
handler.setFormatter(logFormatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

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


def load_train_sample():
    df = dd.read_parquet(path + "/income/05_model_input/train_test.parquet")
    base = df.compute()
    base.ingreso_neto_comprobado = pd.to_numeric(base.ingreso_neto_comprobado)
    base = base.query("ingreso_neto_comprobado > 4000 & ingreso_neto_comprobado < 100000")
    base = base[base["net_income_verified"] >= base["ingreso_neto_comprobado"]]
    return base


def trainer():
    base = load_train_sample()
    path_s3 = path + "/income/models/"
    resultados_test = {}
    to_drop = ['researchable_id', 'estimate', 'declarativa']
    columnas = [col for col in base.columns if col not in to_drop]
    declarado = [col for col in columnas if "ingreso_neto_comprobado" not in col]
    nodeclarado = [col for col in declarado if "net_income_verified" not in col]
    models = [declarado, nodeclarado]
    names = ["declarado", "nodeclarado"]

    for model, name in zip(models, names):
        X = base[model]
        y = base["ingreso_neto_comprobado"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="minimize")
        logger.info("TRIAL")
        func = lambda trial: objective(trial, X_train, X_test, y_train, y_test)
        study.optimize(func, n_trials=25, timeout=180)
        logger.info("TIMEOUT")

        logger.info("Number of finished trials: ", len(study.trials))
        logger.info("Best trial:")
        trial = study.best_trial

        hp = study.best_params
        logger.info("  Value: {}".format(trial.value))
        logger.info("  Params: ")

        for key, value in trial.params.items():
            logger.info("    {}: {}".format(key, value))

        lgbm_model = lgbm.LGBMRegressor(**hp)

        lgbm_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                       callbacks=[lgbm.early_stopping(stopping_rounds=25)])
        y_pred_lgbm = lgbm_model.predict(X_test)
        performance_test = mape(y_test, y_pred_lgbm)
        resultados_test[name] = performance_test

        write_pickle(path_s3 + f"{name}.pkl", lgbm_model)
        write_pickle(path_s3 + f"{name}_performance.pickle", performance_test)
        logger.info(" Save to",path_s3 + f"{name}.pkl")
        logger.info("Performance", performance_test)

        
def LocalTrainer(base=None, to_drop=['researchable_id', 'estimate', 'declarativa'], target="ingreso_neto_comprobado", path_s3=None, name='modelo_test'):
    to_drop = to_drop + [target]
    columnas = [col for col in base.columns if col not in to_drop]
    X = base[columnas]
    y = base[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="minimize")
    func = lambda trial: objective(trial, X_train, X_test, y_train, y_test)
    study.optimize(func, n_trials=25)


    logger.info("Number of finished trials: ", len(study.trials))
    logger.info("Best trial:")
    trial = study.best_trial

    hp = study.best_params
    logger.info("  Value: {}".format(trial.value))
    logger.info("  Params: ")

    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))

    lgbm_model = lgbm.LGBMRegressor(**hp)

    lgbm_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                   callbacks=[lgbm.early_stopping(stopping_rounds=25)])
    
    y_pred_lgbm = lgbm_model.predict(X_test)
    performance_test = mape(y_test, y_pred_lgbm)

    write_pickle(path_s3 + f"{name}.pkl", lgbm_model)
    write_pickle(path_s3 + f"{name}_performance.pkl", performance_test)
    logger.info(" Save to", path_s3 + f"{name}.pkl")
    logger.info("Performance", performance_test)
    

class LgbmTrainer