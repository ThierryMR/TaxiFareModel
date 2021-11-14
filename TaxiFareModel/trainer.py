# imports
from sklearn.pipeline import Pipeline
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
import numpy as np
from TaxiFareModel.data import clean_data, get_data
from sklearn.model_selection import train_test_split

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

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
            ('linear_model', LinearRegression())
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
    trainer = Trainer(X_train, y_train)
    trainer.run()
    print("Pipeline treinado")
    # evaluate
    evaluate = trainer.evaluate(X_test, y_test)
    print(f"Evaluacao final RMSE: {evaluate}")