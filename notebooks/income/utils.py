import pandas as pd
import numpy as np

def normalize_core(core):
    core.columns = [c.replace(' ', '_') for c in core.columns]
    import re
    core["Ingreso_"] = core["Ingreso_"].apply(lambda x: re.sub("[^\d\,]", "", str(x)))
    core["Ing_Disp"] = core["Ing_Disp"].apply(lambda x: re.sub("[^\d\,]", "", str(x)))
    core["Ingreso_"] = [x.replace(',', '.') for x in core["Ingreso_"]]
    core["Ing_Disp"]= [x.replace(',', '.') for x in core["Ing_Disp"]]
    core["Ingreso_"] = [x.split(".")[0] for x in core["Ingreso_"]]
    core["Ing_Disp"] = [x.split(".")[0] for x in core["Ing_Disp"]]
    core["Ing_Disp"] = pd.to_numeric(core["Ing_Disp"])
    core["Ingreso_"]  = pd.to_numeric(core["Ingreso_"])
    core["Dependientes"] = core["Dependientes"].apply(lambda x: re.sub("[^\w]", "", str(x)))
    core["Dependientes"] = core["Dependientes"].replace("nan",0)
    core["Dependientes"] = core["Dependientes"].replace("None",0)
    core["Dependientes"] = pd.to_numeric(core["Dependientes"])
    core["Dependientes"]  = core["Dependientes"].fillna(0)
    core["descuentos"] =  core["Ingreso_"] - core["Ing_Disp"]
    core["Monto_de_la_mensualidad"] = core["Monto_de_la_mensualidad"].apply(lambda x: re.sub("[^\d\,]", "", str(x)))
    core["Monto_de_la_mensualidad"] = [x.replace(',', '.') for x in core["Monto_de_la_mensualidad"]]
    core["Monto_de_la_mensualidad"] = pd.to_numeric(core["Monto_de_la_mensualidad"])
    core['BC_Score_'] = core['BC_Score_'].apply(lambda x: re.sub("[^\d\,]", "", str(x)))
    core['BC_Score_'] = pd.to_numeric(core['BC_Score_'])
    core["rule"]= np.where((core["BC_Score_"]>=680), 1,0)
    core["perfil"]= np.where((core["rule"]==0), "otros","Perfil_X")
    columnas = ["perfil","BC_Score_","Monto_de_la_mensualidad","descuentos","Dependientes","Ingreso_","Ing_Disp"]
    return core[columnas]



selected = ['perfil',
            'CAP_ing_declarado',
     'CAP_validated_final_model',
     'CAP_preds_declarado',
     'CAP_preds_sin_declarado',
     'CAP_real','flag_ingreso_neto_comprobado',
            'CAP_min_pred',
     'CAP_pred_promedio',
     'flag_validated_final_model',
     'flag_preds_declarado',
     'flag_preds_sin_declarado',
     'flag_min_pred',
     'flag_pred_promedio',
     'BC_Score_',
     'Monto_de_la_mensualidad',
     'descuentos',
     'Dependientes',
     'Ingreso_',
     'Ing_Disp',
           'real',
     'ing_declarado',
     'pred_promedio',
     'min_pred']

def create_dinamic(df):
    from IPython.display import HTML
    from pivottablejs import pivot_ui
    HTML('pivottablejs.html')
    return pivot_ui(df[selected])



with mlflow.start_run(nested=True) as run:
    experiment_name = "Income Prediction"
    mlflow.set_experiment(experiment_name)
    mlflc = MLflowCallback(
    metric_name="MSE")
    @mlflc.track_in_mlflow()
    def objective(trial):
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

        to_drop = ["primary_key", "ingreso_real"]
        columnas = [col for col in raw_df.columns if col not in to_drop]
        X = raw_df[columnas]
        y = raw_df["ingreso_real"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgbm.early_stopping(stopping_rounds=20)])
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds, squared=False)

        return mse


    study = optuna.create_study(study_name="base")
    study.optimize(objective, n_trials=10, callbacks=[mlflc])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    best_params = study.best_params
    print("  Value: {}".format(trial.value))
    print("  Params: ")

    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    lgbm_model = lgbm.LGBMRegressor(**best_params)

    tic = time.time()
    model.fit(X_train, y_train)
    duration_training = time.time() - tic

      # Make the prediction
    tic1 = time.time()
    prediction = model.predict(X_test)
    duration_prediction = time.time() - tic1

    # Evaluate the model prediction
    metrics = {
        "rmse" : np.sqrt(mean_squared_error(y_test, prediction)),
        "MAPE" : MAPE(y_test, prediction),
        "r2" : r2_score(y_test, prediction),
        "duration_training" : duration_training,
        "duration_prediction" : duration_prediction }

      # Log in mlflow (parameter)
    mlflow.log_params(**best_params)

      # Log in mlflow (metrics)
    mlflow.log_metrics(metrics)

      # log in mlflow (model)
    mlflow.sklearn.log_model(model, "Optuna LightGBM")

      # Tag the model
    mlflow.set_tags("Income model ")