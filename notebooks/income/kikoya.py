import boto3
import dask.dataframe as dd
import os
import pandas as pd
from s3_utils.s3_utils import write_parquet_from_pd, start_logger


class Kikoya:
    def __init__(self):
        self.bucket = 'kavak-datalake-serving-layer'
        self.prefix = 'data_science/kikoya/ultima_actualizacion/'
        self.basepath = f's3://{self.bucket}/{self.prefix}'
        self.base_keys = ['folio', 'folio_id', 'researchable_id']
        self.connection = self.set_kikoya_client()

    def set_kikoya_client(self):
        sts = boto3.client('sts')
        role = sts.assume_role(
            RoleArn='arn:aws:iam::405458584171:role/dataScienceAccess',
            RoleSessionName="KavakCapital")
        self.creds = {
            "key": role['Credentials']['AccessKeyId'],
            "secret": role['Credentials']['SecretAccessKey'],
            "token": role['Credentials']['SessionToken']
        }

        return boto3.client('s3', aws_access_key_id=self.creds['key'], aws_secret_access_key=self.creds['secret'],
                            aws_session_token=self.creds['token'])


class KikoyaIncome(Kikoya):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tipos_de_objetos = ['product_applications', 
                                 'product_application_users',
                                 'product_application_financial_data']
        self.paths = self.connect_to_files()
        self.save_path = "s3://data-science-kavak-dev/projects/cerberus/v2/dev/income/01_raw/" #clase abstracta raw data

    # TODO class connetor o blue prints?
    def connect_to_files(self, start_date='2020-10'):
        s3_client = self.set_kikoya_client()
        all_bucket_files = s3_client.list_objects(Bucket=self.bucket, Prefix=self.prefix).get('Contents')
        files = [f"s3://{self.bucket}/{x.get('Key')}" for x in all_bucket_files]
        files = pd.Series(files, name='uris').reset_index()
        files['tipo'] = files['uris'].apply(lambda x: os.path.splitext(os.path.split(x)[1])[0])
        files = files.loc[files['tipo'].isin(self.tipos_de_objetos)]
        files['fechas'] = files['uris'].str.extract('(\d{4}\-\d{2})', expand=False)
        files['fechas'] = pd.to_datetime(files['fechas'])
        files = files.loc[files['fechas'] >= f'{start_date}']
        return files[["tipo", "uris"]].groupby('tipo')['uris'].agg(list).to_dict()

    def products_app(self):
        cols_to_product_applications = ['decorated_folio', 'id']
        cols_to_rename_product_applications = {'id': 'folio_id', 'decorated_folio': 'folio'}
        product_applications = dd.read_parquet(self.paths['product_applications'], storage_options=self.creds,
                                               columns=cols_to_product_applications)
        return product_applications.rename(columns=cols_to_rename_product_applications)

    def users_app(self):
        cols_to_product_applications_users = ['product_application_id', 'id']
        cols_to_rename_applications_users = {'product_application_id': 'folio_id', 'id': 'researchable_id'}
        product_applications_users = dd.read_parquet(
            self.paths['product_application_users'],
            storage_options=self.creds,
            columns=cols_to_product_applications_users)
        return product_applications_users.rename(columns=cols_to_rename_applications_users)

    def financial_app(self):
        cols_financial_data = ['researchable_id', 'net_income_verified']
        return dd.read_parquet(self.paths['product_application_financial_data'], storage_options=self.creds,
                               columns=cols_financial_data)

    def make_kk_table(self):
        data_folios = self.products_app().merge(self.users_app(), how='left', on='folio_id')
        data_folios = data_folios.merge(self.financial_app(), how='left', on='researchable_id')
        data_folios = data_folios.drop_duplicates()
        data_folios = data_folios.compute()
        subset_columns = ["net_income_verified", "researchable_id", "folio"]
        data_folios = data_folios[subset_columns]
        data_folios["folio"] = data_folios["folio"].str.replace("KV", "")
        write_parquet_from_pd(self.save_path + "data_folios.parquet", data_folios)


