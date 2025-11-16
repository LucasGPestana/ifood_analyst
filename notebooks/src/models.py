import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline as SKLPipeline
from imblearn.pipeline import Pipeline as IMBLPipeline

from sklearn.model_selection import cross_validate


from typing import Dict

def construct_pipeline(classificator, preprocessing=None, feat_selection=None, pca=None, resampler=None):

    """Constrói um pipeline com um classificador como etapa final. Além disso, também pode conter etapas de:

        - Pré-processamento (preprocessing): Transformação das variáveis da base
        - Seleção de features (feat_selection): Seletor de variáveis com base em algum método
        - PCA (pca): Análise de componentes principais para redução de dimensionalidade
        - Reamostragem (sampler): Algoritmo que balancea os valores da variável alvo
    
    Parameters
    ----------
    classificator
        Classificador
    preprocessing: sklearn.compose.ColumnTransformer
        Transformadores das variáveis preditoras
    feat_selection
        Seletor de variáveis que reduz a dimensionalidade da base a partir de um método, como o estatístico
    pca: sklearn.decomposition.PCA
        Análise de componentes principais para redução de dimensionalidade da base
    resampler
        Algoritmo de reamostragem que irá balancear as ocorrências da variável alvo
    
    Returns
    -------
    Union[sklearn.pipeline.Pipeline, imblearn.pipeline.Pipeline]
        Pipeline construído com as etapas passadas como argumento. A ordem e nome de cada etapa:
        
        1. Pré-processamento - preprocessing
        2. feat_selection - selector
        3. PCA - pca
        4. Reamostragem - sampler
        5. Classificador - clf

    """

    steps = []
    use_imblearn_pipeline = False

    if preprocessing is not None:
        
        steps.append(
            ("preprocessing", preprocessing)
        )
    
    if feat_selection is not None:

        steps.append(
            ("selector", feat_selection)
        )
    
    if pca is not None:

        steps.append(
            ("pca", pca)
        )
    
    if resampler is not None:

        steps.append(
            ("sampler", resampler)
        )
        use_imblearn_pipeline = True
    
    steps.append(
        ("clf", classificator)
    )

    if use_imblearn_pipeline:

        pipeline = IMBLPipeline(steps)
    
    else:

        pipeline = SKLPipeline(steps)

    return pipeline

def get_cv_results(
        X, y, kf, classificator, preprocessing=None, feat_selection=None, pca=None, resampler=None
):
    
    """Aplica uma validação cruzada, com o número de iterações definida em 'n_splits' de kf, em um dado classificador, para avaliar a sua generalização.

    Parameters
    ----------
    X: pandas.DataFrame
        Variáveis preditoras
    y: Union[pandas.Series, numpy.ndarray]
        Variável alvo
    classificador
        Classificador
    preprocessing: sklearn.compose.ColumnTransformer
        Transformadores das variáveis preditoras
    feat_selection
        Seletor de variáveis que reduz a dimensionalidade da base a partir de um método, como o estatístico
    pca: sklearn.decomposition.PCA
        Análise de componentes principais para redução de dimensionalidade da base
    resampler
        Algoritmo de reamostragem que irá balancear as ocorrências da variável alvo
    
    Returns
    -------
    Dict[str, numpy.ndarray]
        Resultado da validação cruzada, com as métricas de avaliação para cada iteração.
    """
    
    pipeline = construct_pipeline(
        classificator,
        preprocessing,
        feat_selection,
        pca,
        resampler,
    )

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=kf,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ]
    )

    return scores

def organize_results(results: Dict[str, Dict[str, np.ndarray]]):

    """Organiza os resultados da validação cruzada de classificadores em um objeto DataFrame, com uma coluna model (algoritmo) e colunas de métricas.

    Parameters
    ----------
    results: Dict[str, Dict[str, np.ndarray]]
        Resultados da validação cruzada de classificadores, cuja chave é o nome do classificador
    
    Returns
    -------
    pandas.DataFrame
        Resultados organizados em um DataFrame, com uma coluna correspondente ao nome do classificador (model).
    
    """

    for name in results.keys():

        results[name]["total_time"] = results[name]["fit_time"] + results[name]["score_time"]
    
    df_results = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})

    df_results = df_results.explode(df_results.columns.drop("model").to_list())

    return df_results