import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_recall_curve, precision_score


if __name__ == "__main__":
    denuncias = pd.read_parquet("data/clean/denuncias_paot_2001_2019.parquet")

    denuncias["dias_entre_recepcion_ratificacion"] = (denuncias["fecha_de_ratificacion"] - denuncias["fecha_de_recepcion"]).dt.days
    denuncias["dias_entre_ratificacion_turno"] = (denuncias["fecha_de_turno"] - denuncias["fecha_de_ratificacion"]).dt.days
    denuncias["dias_entre_turno_admision"] = (denuncias["fecha_de_admision_radicacion"] - denuncias["fecha_de_turno"]).dt.days
    denuncias["dias entre_admision_conclusion"] = (denuncias["fecha_de_conclusion"] - denuncias["fecha_de_admision_radicacion"]).dt.days
    denuncias["codigo_postal"] = denuncias["codigo_postal"].astype("category")
    denuncias["colonia_alcaldia"] = (denuncias["colonia"] + "-" + denuncias["alcaldia"]).str.strip()
    
    denuncias["admitida"] = (denuncias["estatus"].apply(lambda x: 0 if x == "No admitida" else 1)).astype("category")

    denuncias_ivs = denuncias.drop(["id_denuncia",
                    "expediente",
                    "fecha_de_admision_radicacion",
                    "fecha_de_recepcion",
                    "fecha_de_ratificacion",
                    "fecha_de_conclusion",
                    "fecha_de_turno",
                    "entre_calle_1",
                    "entre_calle_2",
                    "referencias",
                    "actos_hechos_y_omisiones",
                    "coord_x",
                    "coord_y",
                    "domicilio",
                    "colonia",
                    "alcaldia",
                    "estatus"],
                    axis=1)

    mlflow.xgboost.autolog()

    X = denuncias_ivs.drop(["admitida"], axis=1)
    y = denuncias_ivs.loc[:,denuncias_ivs.columns != "admitida"]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

    cat_feats = [
        'tipo_de_denuncia',
        'materia', 
        'medio_de_recepcion', 
        'codigo_postal',
        'regimen_propiedad',
        'area_responsable',
        'colonia_alcaldia',
        'admitida'
    ]

    # Train an initial LightGBM model
    model = XGBClassifier(tree_method="hist", enable_categorical=True, use_label_encoder=False, eval_metric="logloss")
    
    run = mlflow.start_run()

    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=True)

    y_pred = model.predict(X_valid)

    precision_recall_curve = precision_recall_curve
    recall_score = recall_score(y_valid, y_pred)
    precision_score = precision_score(y_valid, y_pred)
