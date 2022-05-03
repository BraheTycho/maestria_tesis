import pandas as pd
from IncomeModel.utils import simple_time_tracker

LOCAL_PATH = "/data/clients.csv"
AWS_BUCKET_PATH = None



@simple_time_tracker
def get_data(nrows=None, local=False, **kwargs):
    """method to get the training data (or a portion of it)"""
    # Add Client() here
    if local:
        path = LOCAL_PATH
    else:
        path = AWS_BUCKET_PATH
    df = pd.read_csv(path, nrows=nrows)
    return df


def clean_df(df, test=False):
    """ Limpiando el dataset en función de conocimiento del dominio
    - No puede haber clientes con ingreso menor a 4mil pesos del o contrario es una mala imputación
    - No puede haber ingresos mayores a 400mil pesos
    """
    df = df.query("net_income_verified > 4000 & net_income_verified < 400000")
    return df


if __name__ == "__main__":
    df = get_data()
