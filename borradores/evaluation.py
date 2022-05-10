from s3_utils import read_pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import RMSLE, MAE, MAPE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def predict_core(data=None, name_model=None, path_models=None):
    data.columns = [c.replace(' ', '_') for c in data.columns]
    predictor = read_pickle(path_models + name_model + ".pkl")
    boost = predictor.booster_
    columns = boost.feature_name()
    X = data[columns]
    return predictor.predict(X)

def predict_multiple_models(df, names, path_models):
    for name in names:
        df[f"{name}"]= predict_core(data=df, name_model=name, path_models=path_models)
        print(f"Creating columns {name}")
    return df


def evaluate_model(df=None, pred="validated_final_model",true="ingreso_neto_comprobado",metric="mape"):
    df["ratio_income_min_w"] = df[true]/4000
    
    df["mape"] = df.apply(lambda x: MAPE_m(x[true], x[pred]), axis=1)
    df["residuals"] = df.apply(lambda x: (x[pred]-x[true]), axis=1)

    lst = [df]
    for column in lst:
        column.loc[(column["ratio_income_min_w"] > 1) & (column["ratio_income_min_w"] <= 2), 'salarios_min'] = "1 hasta 2 salarios"
        column.loc[(column["ratio_income_min_w"] > 2) & (column["ratio_income_min_w"] <= 3), 'salarios_min'] = "2 hasta 3 salarios"
        column.loc[(column["ratio_income_min_w"] > 3) & (column["ratio_income_min_w"] <= 5), 'salarios_min'] = "3 hasta 5 salarios"
        column.loc[(column["ratio_income_min_w"] > 5) & (column["ratio_income_min_w"] <= 8), 'salarios_min'] = "5 hasta 8 salarios"
        column.loc[(column["ratio_income_min_w"] > 8) & (column["ratio_income_min_w"] <= 12), 'salarios_min'] = "8 hasta 12 salarios"
        column.loc[(column["ratio_income_min_w"] > 12) & (column["ratio_income_min_w"] <= 16), 'salarios_min'] = "12 hasta 16 salarios"
        column.loc[column["ratio_income_min_w"] >16, 'salarios_min'] = "+ de 16 salarios"

    reorderlist = [ "1 hasta 2 salarios","2 hasta 3 salarios","3 hasta 5 salarios","5 hasta 8 salarios" ,"8 hasta 12 salarios","12 hasta 16 salarios","+ de 16 salarios"]
    tabla_resultados = df.groupby(["salarios_min"]).agg(
        Mape_promedio =  ('mape', 'mean'),
        Diferencia_promedio = ('residuals', 'mean'),
        Cantidad_clientes =  ('mape', 'count')).reindex(reorderlist)
    print(tabla_resultados)
    df["ration_pred_min"] = df[pred]/4000


    newreorderlist = [ "1 hasta 2 salarios","2 hasta 3 salarios","3 hasta 5 salarios","5 hasta 8 salarios" ,"8 hasta 12 salarios","+ de 12 salarios"]

    lst = [df]
    for column in lst:
        column.loc[(column["ration_pred_min"] > 1) & (column["ration_pred_min"] <= 2), 'pred_min'] = "1 hasta 2 salarios"
        column.loc[(column["ration_pred_min"] > 2) & (column["ration_pred_min"] <= 3), 'pred_min'] = "2 hasta 3 salarios"
        column.loc[(column["ration_pred_min"] > 3) & (column["ration_pred_min"] <= 5), 'pred_min'] = "3 hasta 5 salarios"
        column.loc[(column["ration_pred_min"] > 5) & (column["ration_pred_min"] <= 8), 'pred_min'] = "5 hasta 8 salarios"
        column.loc[(column["ration_pred_min"] > 8) & (column["ration_pred_min"] <= 12), 'pred_min'] = "8 hasta 12 salarios"
        column.loc[(column["ration_pred_min"] > 12) & (column["ration_pred_min"] <= 16), 'pred_min'] = "12 hasta 16 salarios"
        column.loc[column["ration_pred_min"] >16, 'pred_min'] = "+ de 16 salarios"

    ingreso_diff_per = pd.crosstab(df['salarios_min'],df['pred_min']).reindex(reorderlist)

    columnas = [col for col in reorderlist if col in ingreso_diff_per.columns] 
    ingreso_diff_per = ingreso_diff_per[columnas]
    reorderlist2 = ['+ de 16 salarios',
                        '12 hasta 16 salarios',
                        '8 hasta 12 salarios',
                        '5 hasta 8 salarios',
                        '3 hasta 5 salarios',
                        '2 hasta 3 salarios',
                        '1 hasta 2 salarios']


    fig, ax = plt.subplots(figsize=(12, 12))
    label_size = 16
    plt.rcParams['xtick.labelsize'] = label_size
    plt.rcParams['ytick.labelsize'] = label_size
    # color map
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True).reversed()
    # plot heatmap
    sns.despine(left=True)
    sns.heatmap(ingreso_diff_per.T.reindex(reorderlist2), fmt=".2f",annot=True,annot_kws={"size": 16},
               linewidths=5, cmap=cmap,cbar=False, square=True)

    plt.rcParams["axes.labelsize"] = 12
    plt.xlabel('Salario Real \n ', fontsize = 20) # x-axis label with fontsize 15
    plt.ylabel('Predicho \n', fontsize = 20)

    plt.suptitle(f"Evaluación Modelo {pred}\n\n ", fontsize = 20)
    plt.title("Cantidad de Clientes por Ingreso Mínimo \n Ingreso declarado vs Verdadero\n", fontsize = 14)
    #plt.savefig("figures/evaluacion.png", bbox_inches='tight' )
    reorderlist = ["1 hasta 2 salarios","2 hasta 3 salarios","3 hasta 5 salarios","5 hasta 8 salarios" ,"8 hasta 12 salarios","12 hasta 16 salarios","+ de 16 salarios"]

    ingreso_diff_per = pd.crosstab(df['salarios_min'],df['pred_min'])
    ingreso_diff_per = ingreso_diff_per.apply(lambda x: x/x.sum()).reindex(reorderlist2)
    ingreso_diff_per = ingreso_diff_per.T.reindex(reorderlist2)
    columnas = [col for col in reorderlist if col in ingreso_diff_per.columns] 
    ingreso_diff_per = ingreso_diff_per[columnas]

    fig2, ax2 = plt.subplots(figsize=(12, 12))
    label_size = 16
    plt.rcParams['xtick.labelsize'] = label_size
    plt.rcParams['ytick.labelsize'] = label_size
    # color map
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True).reversed()
    # plot heatmap
    sns.despine(left=True)
    sns.heatmap(ingreso_diff_per, fmt=".2%",annot=True,annot_kws={"size": 16},
               linewidths=5, cmap=cmap,cbar=False, square=True)

    plt.rcParams["axes.labelsize"] = 12
    plt.xlabel('Salario Real \n ', fontsize = 20) # x-axis label with fontsize 15
    plt.ylabel('Predicho \n', fontsize = 20)

    plt.suptitle(f"Evaluación Modelo {pred}\n\n ", fontsize = 20)
    plt.title("Ingreso predicho vs Verdadero\n\n 100% predicho por bucket en comparación a real\n", fontsize = 14)
    return fig, fig2







class EvalRegression:
    def __init__(self):
        self.y_test = None
        self.y_pred = None
        self.results_df = None
        self.low_thresh = None
        self.high_thresh = None

    def fit(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred

    def create_results(self):
        self.results_df = pd.merge(pd.DataFrame(self.y_test), pd.DataFrame(self.y_pred, index=self.y_test.index), right_index=True,
                                   left_index=True)
        self.results_df.columns = ["true_income", "pred"]

    def calculate_whiskers(self):
        mean = self.results_df["pred"].mean()
        high = self.results_df["pred"].quantile(.75)
        low = self.results_df["pred"].quantile(.25)
        iqr = high - low
        self.low_thresh = mean - 1.5 * iqr
        self.high_thresh = mean + 1.5 * iqr

    def replace_irrational(self):
        self.results_df.loc[self.results_df["pred"] < self.low_thresh, "pred"] = self.results_df.pred.quantile(.25)
        self.results_df.loc[self.results_df["pred"] > self.high_thresh, "pred"] = self.high_thresh
        self.results_df["residuals"] = self.results_df.pred - self.results_df.true_income

    def print_metrics_results(self):
        y_test_series = self.results_df["true_income"]
        y_pred_series = self.results_df["pred"]
        print(f"el Error cuadratico medio es {RMSLE(y_test_series, y_pred_series)}")
        print(f"el MAE es {MAE(y_test_series, y_pred_series)}")
        print(f"el MAPE es {MAPE(y_test_series, y_pred_series)}")
        plt.figure(figsize=(10, 5))
        sns.kdeplot(y_test_series, label='true_income')
        sns.kdeplot(y_pred_series, label='pred')
        plt.legend()

    def min_declared_predicted(self, declared_df, threshold=0.05):
        self.results = pd.merge(declared_df, pd.DataFrame(self.y_pred), right_index=True, left_index=True)
        self.results.columns = ["declared", "true_income", "pred"]
        self.results["pred_cond"] = self.results[["pred"]].apply(lambda x: x * (1 + threshold), axis=1)
        self.results["pred_new"] = np.where(self.results["pred_cond"] < self.results["declared"],
                                            self.results["pred_cond"],
                                            self.results["declared"])

    def make_eval_pred_declared(self, declared_df, threshold=0.5):
        self.min_declared_predicted(declared_df, threshold=0.5)
        self.calculate_whiskers()
        self.replace_irrational()
        return self.results_df

    def make_eval(self):
        self.create_results()
        self.calculate_whiskers()
        self.replace_irrational()
        return self.results_df


def plot_income_results(df, column='income_binned', metric='MAPE'):
    import matplotlib.ticker as mtick
    fig, axs = plt.subplots(figsize=(12, 7))
    plt.xticks(rotation=90)
    sns.pointplot(data=df, x=column, y=metric)
    plt.ylabel(metric)
    plt.xlabel('Income Bins')
    plt.title(f'{metric} by Income Level')
    vals = axs.get_yticks()
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('error_by_income_level_classifier.png')
    plt.figure(figsize=(12, 7))
    plt.xticks(rotation=90)
    sns.countplot(data=df, x=f'{column}')
    plt.xlabel(f'{column}')
    plt.title(f'# of Observations by {column}')
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')
    plt.savefig('observations_by_income_level_classifier.png')


def add_metrics_results(model_results, y_true="true_income", y_pred="pred"):
    metricas = [RMSLE, MAE, MAPE]
    name_metricas = ["RMSLE", "MAE", "MAPE"]
    for name, metrica in zip(name_metricas, metricas):
        model_results[f'{name}'] = model_results.apply(lambda x: metrica(x[y_true], x[y_pred]), axis=1)
    model_results = model_results[model_results[y_true] <= 300000]
    intervals = np.concatenate((np.arange(7000, 63000, 5000), np.arange(100000, 350000, 50000)))
    model_results = model_results.assign(
        ingresos_bin=pd.cut(model_results[y_true], bins=intervals, labels=np.arange(1, len(intervals))),
        income_binned=pd.cut(model_results[y_true], bins=intervals)
    )
    return model_results


if __name__ == '__main__':
    main()
