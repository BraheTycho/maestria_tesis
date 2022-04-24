import pandas as pd


def quality_check(df):
    percent_m = [df[col].isna().sum() * 100 / len(df) for col in df.columns]
    percent_unique = [df[col].nunique() * 100 / len(df) for col in df.columns]
    missing_data = pd.DataFrame({"column_name": df.columns,
                                 "percent_missing": percent_m,
                                 "percent_unique": percent_unique,
                                 "sample_values": [df[col].unique()[0:10] for col in df.columns]} )
    print("Evaluation uniques an Nan by Column")
    missing_data = missing_data.sort_values(by="percent_missing", ascending=False)
    return missing_data


def missing_threshold(df, percentage):
    limitPer = len(df) * percentage
    print(f"Original shape {df.shape}")
    df.dropna(thresh=limitPer, axis=1, inplace=True)
    print(f"Final shape {df.shape}")
    

def df_optimized(df, verbose=True, **kwargs):
    """
    Reduces size of dataframe by downcasting numeircal columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return: df optimized
    """
    in_size = df.memory_usage(index=True).sum()
    # Optimized size here
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("optimized size by {} % | {} GB".format(ratio, GB))
    return df