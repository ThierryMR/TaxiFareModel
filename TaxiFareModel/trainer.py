# imports
from sklearn.pipeline import Pipeline
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
import numpy as np
from TaxiFareModel.data import clean_data, get_data
from sklearn.model_selection import train_test_split
from memoized_property import memoized_property

import mlflow
from  mlflow.tracking import MlflowClient

import joblib

class Trainer():
    # MLFLOW_URI = "https://mlflow.lewagon.co/" Tive problemas com essa variavel global
    def __init__(self, X, y, model, model_name):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.model = model
        self.experiment_name = f"[55] [BH] [Thierry] {model_name} TaxiFare + 3.8.12"
        self.mlflow_uri = "https://mlflow.lewagon.co/"

    def set_pipeline(self):
        '''returns a pipelined model'''
        
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', self.model)
        ])
        
        return pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self.pipeline 

    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = np.sqrt(((y_pred - y_test)**2).mean())
        print(rmse)
        return rmse
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.mlflow_uri )
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{self.mlflow_experiment_id}")
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
        
        
    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')
        print("Modelo Treinado Salvo lindao")
        return "Saved Model" 


if __name__ == "__main__":
    # get data
    n_rows = int(input("Digite o numero de linhas que voce deseja trabalhar "))
    df = get_data(n_rows)
    print("Datados obtidos numero de registros = ", df.shape )
    # clean data
    df = clean_data(df)
    print("Dados limpos numero de registros = ", df.shape )
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    print(f"X_shape  {X.shape}   y.shape {y.shape} ")
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.15)    
    print("Hold out finalizado")
    # train
    trainer = Trainer(X_train, y_train, LinearRegression(), "Linear Regression")
    trainer.run()
    print("Pipeline treinado")
    # evaluate
    evaluate = trainer.evaluate(X_test, y_test)
    print(f"Evaluacao final RMSE: {evaluate}")
    
    #Enviando Informacoes para nuvem Wagon
    
    #Bloco desnecessario
    # trainer.mlflow_client()
    # print("mlflow_client Rodou")
    # trainer.mlflow_experiment_id()
    # print("mlflow_experiment_id Rodou")
    # trainer.mlflow_run()
    # print("mlflow_run Rodou")
    
    
    #Bloco de codigos necessarios
    trainer.mlflow_log_metric("rmse", evaluate)
    print("mlflow_log_metric Rodou")
    trainer.mlflow_log_param("model", "Linear Regression")
    print("mlflow_log_param Rodou")
    print("Modelo Enviado para cloud wagon")
    
    #Salvando Modelo Enviado
    
    trainer.save_model()
    