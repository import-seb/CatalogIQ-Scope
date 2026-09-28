from pathlib import Path
import csv
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 80)

ROOT_PATH = Path.cwd().parents[0]
# print(ROOT)

DATA_PATH = ROOT_PATH/"data"
PROVIDED_DATA_PATH = DATA_PATH/"provided"
TRAIN_PATH = PROVIDED_DATA_PATH/"Selfcare_Training_data (1).csv"
TARGET_PATH = PROVIDED_DATA_PATH/"Selfcare_Target_data (1).csv"

DOCS_PATH = ROOT_PATH/"docs"

NOTEBOOKS_PATH = ROOT_PATH/"notebooks"

