import dask.dataframe as dd
import logging
import pandas as pd
import sys
import os

from s3_utils import write_parquet_from_pd, read_pd_from_parquet, write_pickle, read_pickle

try:
    from awsglue.utils import getResolvedOptions

    args = getResolvedOptions(sys.argv, ["path"])
    path = args["path"]
except Exception as error:
    print("Running script locally")
    path = "glue_scripts/output"

import logging
# Logger SetUp
# -------------------------------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logFormatter = logging.Formatter(
    "%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
)
handler.setFormatter(logFormatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
# -------------------------------------------------------------------------------



def create_clients_with_income():
    logger.info("create clients with income")
    # lee las fuentes ya procesada y crea una lista (ids con income) y un dataset con las columnas
    netsuite_data = dd.read_parquet(path + "/income/01_raw/netsuitedata.parquet")
    data_folios_pq = dd.read_parquet(path + "/income/01_raw/data_folios.parquet")

    netsuite_data = netsuite_data.compute()
    data_folios_pq = data_folios_pq.compute()

    folios_income = pd.merge(netsuite_data, data_folios_pq, on="folio", how="inner")
    folios_income = folios_income[["researchable_id", "net_income_verified", "ingreso_neto_comprobado", "estimate"]]
    filter_list = list(folios_income.researchable_id)
    write_pickle(
        path + "/income/02_intermediate/kk_ids_filter_list.pickle",
        filter_list
    )

    write_parquet_from_pd(
        path + "/income/02_intermediate/ns_kk_income_data.parquet",
        folios_income
    )
    logger.info("PATH!!", {path})


def load_intl():
    logger.info("Load INTL")
    # carga datos de intl
    filter_list = read_pickle(path + "/income/02_intermediate/kk_ids_filter_list.pickle")
    logger.info(f'Read filter list OK! {len(filter_list)}')
    df_rs = dd.read_parquet(path + "/data/kikoya/03_processed/2022-04-26/rsSections.pq/")
    df_rs_filt = df_rs.loc[df_rs.researchable_id.isin(filter_list)].compute()
    df_rs_filt.drop_duplicates(subset="researchable_id", inplace=True, keep="last")
    df_sc = dd.read_parquet(path + "/data/kikoya/03_processed/2022-04-26/scSections.pq/")
    df_sc_filt = df_sc.loc[df_sc.researchable_id.isin(filter_list)].compute()
    df_sc_filt.drop_duplicates(subset="researchable_id", inplace=True, keep="last")
    return pd.merge(df_rs_filt, df_sc_filt, on="researchable_id")


def preprocess_intl():
    logger.info("Process INTL")
    base_df = load_intl()
    folios_income = dd.read_parquet(path + "/income/02_intermediate/ns_kk_income_data.parquet").compute()

    fechas = [col for col in base_df.columns if "fecha" in col]
    base_df[fechas] = base_df[fechas].replace("00000000", "01011900")
    for fecha in fechas:
        base_df[fecha] = pd.to_datetime(base_df[fecha], format="%d%m%Y", errors='coerce')
    for col in base_df.columns:
        if base_df[col].dtype == 'object':
            base_df[col] = base_df[col].str.extract('(\d+)').astype(float)
    base_modelo = pd.merge(base_df, folios_income, how="inner", on="researchable_id")
    write_parquet_from_pd(path + "/income/02_intermediate/income_data.parquet", base_modelo)
    logger.info("Write Income Data with len ", len(base_modelo))
    logger.info("With errors ",len(base_modelo)> 0)
