import mlflow
import pickle


mlflow.set_tracking_uri('http://mlflow:5000')
mlflow.set_experiment('homework_3')


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    dv, lr = data

    with mlflow.start_run():
        with open('vectorizer.bin', 'wb') as out:
            pickle.dump(dv, out)
        mlflow.log_artifact('vectorizer.bin')
        mlflow.sklearn.log_model(lr, 'model')
