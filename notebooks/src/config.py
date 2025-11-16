from pathlib import Path


PROJECT_DIRPATH = Path(__file__).resolve().parents[2]

DATA_DIRPATH = PROJECT_DIRPATH / "data"

ORIGINAL_DATA = DATA_DIRPATH / "ml_project1_data.csv"
DUMMIES_DATA = DATA_DIRPATH / "ml_project1_data_dummies.csv"
CLEANED_DATA = DATA_DIRPATH / "ml_project1_data_tratado.parquet"
CLUSTERED_DATA = DATA_DIRPATH / "clustered_data.parquet"

MODELS_DIRPATH = PROJECT_DIRPATH / "models"

FINAL_MODEL_FILEPATH = MODELS_DIRPATH / "logreg_preprocessing_undersampling.joblib"

REPORTS_DIRPATH = PROJECT_DIRPATH / "relatorios"
IMAGES_DIRPATH = REPORTS_DIRPATH / "imagens"
