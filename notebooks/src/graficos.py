import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, CenteredNorm

import pandas as pd
import seaborn as sns


from typing import List, Literal


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

def plot_corr_barplot(df: pd.DataFrame, column: str, corr_method: Literal["pearson", "spearman"]="pearson") -> None:

    """Plota um gráfico de barras da correlação entre 'column' e as demais colunas de 'df'

    Parameters
    ----------
    df: pd.DataFrame
        Fonte dos dados
    
    column: str
        Coluna a ser analisada a correlação
    
    corr_method: Literal["pearson", "spearman"]
        Tipo de correlação. Pode ser de pearson (relação linear) ou spearman (relação monocotômica).

    """

    series_corr = df.corr(method=corr_method)[column].drop(column).sort_values()

    cmap = "coolwarm_r"
    cnorm = CenteredNorm(vcenter=0, halfrange=max([abs(series_corr.min()), abs(series_corr.max())]))
    smap = ScalarMappable(norm=cnorm, cmap=cmap)

    listed_colors = ListedColormap([smap.to_rgba(value) for value in series_corr.values]).colors


    _, ax = plt.subplots(figsize=(16 ,6))

    sns.barplot(
        x=series_corr.index,
        y=series_corr.values,
        hue=series_corr.values,
        legend=False,
        palette=listed_colors,
        ax=ax
    )

    ax.set_xlabel("Atributos")
    ax.set_ylabel("Correlação")
    ax.set_title("Correlação de Pearson entre Response e outros atributos")

    ax.tick_params(axis='x', rotation=90)

    plt.show()
