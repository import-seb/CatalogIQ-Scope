from pathlib import Path
import csv
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 80)

ROOT_PATH = Path.cwd().parents[0]
# print(ROOT)

TRAIN_PATH = Path(ROOT_PATH/"data/provided/Selfcare_Training_data (1).csv")
TARGET_PATH = Path(ROOT_PATH/"data/provided/Selfcare_Target_data (1).csv")