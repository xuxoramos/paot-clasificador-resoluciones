import pandas as pd
import pyarrow.csv as pv
import pyarrow.parquet as pq

sourcedir = "data/processed/"
filename = "denuncias_paot_2001_2019.csv"
outputdir = "data/clean/"

table = pd.read_csv("data/processed/" + filename)

for column in table.columns:
    if column.startswith("fecha"):
        table[column] = pd.to_datetime(table[column], format="%d/%M/%Y")
    
table.to_parquet(outputdir + filename.replace("csv", "parquet"), index=False)