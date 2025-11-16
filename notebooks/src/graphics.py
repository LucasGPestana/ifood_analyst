import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, CenteredNorm
from matplotlib.ticker import PercentFormatter

import numpy as np
import pandas as pd
import seaborn as sns


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


from typing import List, Literal, Tuple, Dict, Any, Sequence


def pairplot(df: pd.DataFrame, columns: List[str], hue_column: str, palette: str="tab10") -> None:

    """Plota em uma matriz de sistema de eixos gráficos de dispersão das colunas, e histogramas na diagonal principal

    Parameters
    ----------
    df: pd.DataFrame
        Fonte dos dados
    
    columns: List[str]
        Colunas a serem analisadas
    
    hue_column: str
        Coluna em columns que definirá categorias de cores associadas a cada valor
    
    palette : str
        Paleta do gráfico

    """

    analysis_columns = columns + [hue_column]

    sns.pairplot(
        df[analysis_columns],
        hue=hue_column,
        palette=palette,
        corner=True,
    )

    plt.show()

def plot_bivariate_boxplot(
        df: pd.DataFrame, 
        columns: List[str], 
        x_cat_column: str, 
        hue_cat_column: str=None
) -> None:
    
    """Plota boxplots de colunas numéricas para cada categoria de "x_cat_column"

    Parameters
    ----------
    df: pandas.DataFrame
        Base de dados
    columns: List[str]
        Colunas numéricas para avaliação dos quartis e outliers
    x_cat_column: str
        Coluna categórica com as categorias de cada boxplot
    hue_cat_column: str
        Coluna categórica para definição das cores de cada boxplot
    
    """

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

def plot_n_cluster_estimator_methods(X: pd.DataFrame, random_state: int, k_range: Tuple[int]=(2, 11)) -> None:

    """Plota um gráfico dos métodos do cotovelo e silhueta para estimar o número de clusters ideal

    Parameters
    ----------
    X : pandas.DataFrame
        Fonte dos dados

    random_state : int
        Seed de replicação de aleatoriedade

    k_range : Tuple[int]
        Intervalo do número de clusters que serão avaliados
    """

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    elbow = {}
    silhouettes = []

    k_range = range(*k_range)

    for k in k_range:

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

        kmeans.fit(X)

        elbow[k] = kmeans.inertia_

        labels = kmeans.labels_

        silhouettes.append(
            silhouette_score(
                X, labels
            )
        )

    sns.lineplot(
        x=list(elbow.keys()),
        y=list(elbow.values()),
        ax=axs[0],
    )

    axs[0].set_xlabel("Número de clusters")
    axs[0].set_ylabel("Inércia")

    sns.lineplot(
        x=list(k_range),
        y=silhouettes,
        ax=axs[1],
    )

    axs[1].set_xlabel("Número de clusters")
    axs[1].set_ylabel("Índice de Silhueta")

    plt.tight_layout()
    plt.show()

def plot_values_pct_per_cluster(
        df: pd.DataFrame,
        columns: List[str],
        fig_kwargs: Dict[str, Any]=dict(figsize=(10, 12), sharey=True),
        cluster_column: str="cluster",
):
    
    """Plota um gráfico de barras com a porcentagem de cada categoria de colunas categóricas, para cada cluster definido em uma segmentação.

    Parameters
    ----------
    df: pandas.DataFrame
        Base de dados
    columns: List[str]
        Colunas categóricas
    fig_kwargs: Dict[str, Any]
        Parâmetros da figura
    cluster_column: str
        Coluna com os rótulos dos clusters
    """
    
    if not fig_kwargs.get("nrows") and not fig_kwargs.get("ncols"):

        fig_kwargs["nrows"] = 1
        fig_kwargs["ncols"] = len(columns)
    
    fig, axs = plt.subplots(**fig_kwargs)

    for ax, column in zip(axs.flatten() if isinstance(axs, np.ndarray) else [axs], columns):

        sns.histplot(
            df,
            x=cluster_column,
            hue=column,
            multiple="fill",
            stat="percent",
            discrete=True,
            shrink=0.8,
            ax=ax,
        )

        clusters = sorted(df[cluster_column].unique())

        ax.set_xticks(clusters)

        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel("")

        ax.tick_params(
            axis="both", 
            which="both", 
            length=0
        )

        for container in ax.containers:

            labels = []

            for artist in container:

                labels.append(f"{artist.get_height():.1%}")
                artist.set_linewidth(0)

            ax.bar_label(
                container, 
                label_type="center",
                labels=labels,
                color="white",
            )

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()

def plot_clusters_pct_per_value(
        df: pd.DataFrame,
        columns: List[str],
        fig_kwargs: Dict[str, Any]=dict(figsize=(10, 12), sharey=True),
        cluster_column: str="cluster",
):
    """Plota um gráfico de barras com a porcentagem de cada cluster definido em uma segmentação, para cada categoria de colunas categóricas.

    Parameters
    ----------
    df: pandas.DataFrame
        Base de dados
    columns: List[str]
        Colunas categóricas
    fig_kwargs: Dict[str, Any]
        Parâmetros da figura
    cluster_column: str
        Coluna com os rótulos dos clusters
    """
    
    if not fig_kwargs.get("nrows") and not fig_kwargs.get("ncols"):

        fig_kwargs["nrows"] = 1
        fig_kwargs["ncols"] = len(columns)
    
    fig, axs = plt.subplots(**fig_kwargs)

    last_legend = None

    clusters = sorted(df[cluster_column].unique())

    for ax, column in zip(axs.flatten() if isinstance(axs, np.ndarray) else [axs], columns):

        sns.histplot(
            df,
            x=column,
            hue=cluster_column,
            multiple="fill",
            stat="percent",
            discrete=True,
            shrink=0.8,
            palette="tab10",
            ax=ax,
        )

        values = sorted(df[column].unique())

        ax.set_xticks(range(len(values)), labels=values)

        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel("")

        ax.tick_params(
            axis="both", 
            which="both", 
            length=0
        )

        for container in ax.containers:

            labels = []

            for artist in container:

                labels.append(f"{artist.get_height():.1%}")
                artist.set_linewidth(0)

            ax.bar_label(
                container, 
                label_type="center",
                labels=labels,
                color="white",
            )
        
        legend = ax.get_legend()
        last_legend = legend

        legend.remove()
        
    fig.legend(
        handles=last_legend.legend_handles,
        labels=clusters,
        loc="upper center",
        ncol=len(clusters)
    )
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()

def plot_clusters_2D(
        data: pd.DataFrame,
        columns: List[str],
        centroids: Sequence[Sequence[float]],
        show_centroids: bool=True,
        show_data_points: bool=False,
        cluster_labels: Sequence[int]=None,
) -> None:
    
    """Plota gráfico em 2 dimensões de visualização de clusters, com seus respectivos centróides

    Parameters
    ----------
    data : pandas.DataFrame
        Dados que compoem os clusters
    
    columns : Sequence[str]
        Nomes das colunas de 'data'
    
    centroids: Sequence[Sequence[int]]
        Coordenadas de todos os centroides
    
    show_centroids : bool
        Exibe os centroides no gráfico
    
    show_data_points : bool
        Exibe os pontos de dados no gráfico
    
    cluster_labels : Sequence[int]
        Rótulos de cada ponto de dados. Necessário apenas se show_data_points for True
    """
    
    fig, ax = plt.subplots()

    colors = plt.cm.tab10.colors[:len(centroids)]
    colors = ListedColormap(colors)

    x_label, y_label = columns

    x = data[x_label]
    y = data[y_label]

    for i, centroid in enumerate(centroids):

        if show_centroids:

            ax.scatter(*centroid, s=500, alpha=0.5, color=colors.colors[i])
            ax.text(
                *centroid, 
                f"{i}", 
                fontsize=20, 
                ha="center", 
                va="center"
            )
    
    if show_data_points:

        s = ax.scatter(x, y, c=cluster_labels, cmap=colors)

        ax.legend(*s.legend_elements(), bbox_to_anchor=(1.3, 1))
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Clusters")


    plt.show()

def compare_metrics(df_results: pd.DataFrame):

    """Plota boxplots que comparam as métricas de algoritmos de classificação após uma validação cruzada

    Parameters
    ----------
    df_results: pandas.DataFrame
        Objeto DataFrame que possui uma coluna model, que representa um dado algoritmo de classificação, e outras colunas com as métricas para cada "dobra" da validação cruzada
    
    """

    metrics = ["accuracy", "recall", "precision", "roc_auc", "average_precision"]

    metrics_name = ["time"] + metrics

    scoring = ["total_time"] + [
        "test_" + metric for metric in metrics
    ]

    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

    for ax, metric_name, metric_scoring in zip(axs.flatten(), metrics_name, scoring):

        sns.boxplot(
            df_results,
            x="model",
            y=metric_scoring,
            hue="model",
            palette="tab10",
            ax=ax,
        )

        ax.set_xlabel("model")
        ax.set_ylabel(metric_name)

        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()