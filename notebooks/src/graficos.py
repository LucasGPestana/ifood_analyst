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