import pandas as pd

def inspect_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:

    """Mostra os registros com valores outliers da coluna passada como argumento

    Parameters
    ----------
    df : pandas.DataFrame
        Base que contém a coluna especificada
    column : str
        Nome da coluna que deseja avaliar os outliers
    
    Returns
    -------
    pandas.DataFrame
        DataFrame que contém os registros com valores outliers na coluna especificada
    """

    iqr = df[column].quantile(0.75) - df[column].quantile(0.25)

    upper_thres = df[column].quantile(0.75) + 1.5 * iqr
    lower_thres = df[column].quantile(0.25) - 1.5 * iqr

    
    return df[
        (df[column] > upper_thres) | (df[column] < lower_thres)
        ]