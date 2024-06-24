import pandas as pd
import mlflow.xgboost
import time
from xgboost import XGBClassifier
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score


if __name__ == "__main__":

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    denuncias = pd.read_parquet("data/clean/denuncias_paot_2001_2019.parquet")

    denuncias["dias_entre_recepcion_ratificacion"] = (denuncias["fecha_de_ratificacion"] - denuncias["fecha_de_recepcion"]).dt.days
    denuncias["dias_entre_ratificacion_turno"] = (denuncias["fecha_de_turno"] - denuncias["fecha_de_ratificacion"]).dt.days
    denuncias["dias_entre_turno_admision"] = (denuncias["fecha_de_admision_radicacion"] - denuncias["fecha_de_turno"]).dt.days
    denuncias["dias entre_admision_conclusion"] = (denuncias["fecha_de_conclusion"] - denuncias["fecha_de_admision_radicacion"]).dt.days
    denuncias["colonia_alcaldia"] = ((denuncias["colonia"] + "-" + denuncias["alcaldia"]).str.strip())
    
    denuncias["admitida"] = (denuncias["estatus"].apply(lambda x: 0 if x == "No admitida" else 1)).astype("int")

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

    cat_feats = [
        'tipo_de_denuncia',
        'materia', 
        'medio_de_recepcion', 
        'codigo_postal',
        'regimen_propiedad',
        'area_responsable',
        'colonia_alcaldia',
    ]

    denuncias_ivs.reset_index()

    denuncias_clean = denuncias_ivs.dropna()

    X = denuncias_clean.drop(["admitida"], axis=1)
    y = denuncias_clean["admitida"]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

    for cat in cat_feats:
        X_train[cat] = X_train[cat].astype("category")
        X_valid[cat] = X_valid[cat].astype("category")

    params = {
        "max_depth":10,
        "subsample":0.33,
        "objective":'binary:logistic',
        "n_estimators":300,
        "learning_rate":0.001,
        "tree_method":"hist",
        "enable_categorical":True,
        "eval_metric":"aucpr",
        "early_stopping_rounds":8
    }

    mlflow.set_experiment("Abandoned Report Classifier for the Environmental and Urban Development Agency")
    with mlflow.start_run(run_name=str(time.time())):

        mlflow.xgboost.autolog()

        model = XGBClassifier(**params)
        
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

        y_pred = model.predict(X_valid)

        recall = recall_score(y_valid, y_pred)
        precision = precision_score(y_valid, y_pred)

        print("Recall: %.2f%%" % (recall * 100.0))
        print("Precision: %.2f%%" % (precision * 100.0))

        # Log the hyperparameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Reports to the environmental agency from 2002 to 2019")

        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))

        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="paot_classifier",
            signature=signature,
            #input_example=X_valid,
            registered_model_name="paot_classifier",
        )