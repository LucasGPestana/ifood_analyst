from pathlib import Path


PROJECT_DIRPATH = Path(__file__).resolve().parents[2]

DATA_DIRPATH = PROJECT_DIRPATH / "data"

# coloque abaixo o caminho para os arquivos de dados de seu projeto
DADOS_ORIGINAIS = DATA_DIRPATH / "ml_project1_data.csv"
DADOS_DUMMIES = DATA_DIRPATH / "ml_project1_data_dummies.csv"
DADOS_TRATADOS = DATA_DIRPATH / "ml_project1_data_tratado.parquet"
DADOS_CLUSTERIZADOS = DATA_DIRPATH / "clustered_data.parquet"

# coloque abaixo o caminho para os arquivos de modelos de seu projeto
PASTA_MODELOS = PROJECT_DIRPATH / "modelos"

# coloque abaixo outros caminhos que você julgar necessário
REPORTS_DIRPATH = PROJECT_DIRPATH / "relatorios"
IMAGES_DIRPATH = REPORTS_DIRPATH / "imagens"
