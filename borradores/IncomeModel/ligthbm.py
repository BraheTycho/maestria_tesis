import warnings
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna

warnings.filterwarnings('ignore')


class LightTrainer:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.best_params = None
        self.model = None

    def fit(self, df, to_drop=['researchable_id', 'estimate', 'declarativa'], target="ingreso_neto_comprobado"):
        to_drop = to_drop + [target]
        columnas = [col for col in df.columns if col not in to_drop]
        X = df[columnas]
        y = df[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(self,trial):
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
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], callbacks=[lgbm.early_stopping(stopping_rounds=20)])
        preds = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, preds, squared=False)

        return mse

    def optimize(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        func = lambda trial: self.objective(trial)
        study.optimize(func, n_trials=25)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        self.trial = study.best_trial
        self.best_params = study.best_params
        print("  Value: {}".format(self.trial.value))
        print("  Params: ")

        for key, value in self.trial.params.items():
            print("    {}: {}".format(key, value))
        lgbm_model = lgbm.LGBMRegressor(**self.best_params)
        self.model = lgbm_model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)],
                                    callbacks=[lgbm.early_stopping(stopping_rounds=50)])
