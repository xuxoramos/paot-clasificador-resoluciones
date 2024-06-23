import pandas as pd
from unidecode import unidecode
from pandas.api.types import is_object_dtype

mapping = {'á': 'a',
           'é': 'e',
           'í': 'i',
           'ó': 'o',
           'ú': 'u',
           'ñ': 'n',
           '0000-00-00': None}

data_raw = pd.read_csv("data/raw/rptDatosAbiertos.csv", encoding="iso-8859-1")

new_column_names = []
for column in data_raw.columns:
    new_column_name = unidecode(column).lower().replace(".","").replace(" ", "_").replace(",","").replace("/","_")
    new_column_names.append(new_column_name)
    if is_object_dtype(data_raw[column]):
        data_raw[column] = data_raw[column].replace(mapping, regex=True)

data_raw.columns = new_column_names

data_raw.to_csv("data/processed/denuncias_paot_2001_2019.csv", index=False)






