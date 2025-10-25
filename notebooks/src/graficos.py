import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from typing import List

def pairplot(df: pd.DataFrame, columns: List[str], hue_column: str) -> None:

    """Plota em uma matriz de sistema de eixos gráficos de dispersão das colunas, e histogramas na diagonal principal

    Parameters
    ----------
    df: pd.DataFrame
        Fonte dos dados
    
    columns: List[str]
        Colunas a serem analisadas
    
    hue_column: str
        Coluna em columns que definirá categorias de cores associadas a cada valor

    """

    analysis_columns = columns + [hue_column]

    sns.pairplot(
        df[analysis_columns],
        hue=hue_column
    )

    plt.show()

def plot_bivariate_boxplot(df: pd.DataFrame, columns: List[str], x_cat_column: str, hue_cat_column: str=None) -> None:

    fig, axs = plt.subplots(5, 1, figsize=(8, 16), sharex=True)

    for ax, column in zip(axs, columns):

        sns.boxplot(
            df,
            x=x_cat_column,
            y=column,
            hue=hue_cat_column,
            showmeans=True,
            ax=ax,
        )

    plt.show()