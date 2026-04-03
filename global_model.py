from sklearn_extra.cluster import KMedoids
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
import pyarrow as py
import pickle
import time
import warnings

warnings.filterwarnings("ignore")

early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True, verbose=0
)


def load_data(path, var, shape=365, freq="1D"):
    df = pd.read_parquet(path, engine="pyarrow")
    df.set_index(pd.to_datetime(df["date"]), inplace=True)
    df.drop("date", axis=1, inplace=True)
    estation = []  # list of names from each station
    tensor = []  # list of lat an long from each station
    pos = {}

    for station in df["station"].unique():
        frame = df.loc[df["station"] == station].loc[:, var]
        frame = frame.sort_index()
        lat_long = tuple(
            frame[["latitude", "longitude"]].drop_duplicates().values.flatten()
        )
        data = frame.loc[:, var[2:-1]]
        target = frame.loc[:, var[-1]]
        frame = data.resample(rule=freq).mean().values
        target = target.resample(rule=freq).sum().values
        if (frame.shape[0] == shape) and (np.isnan(frame).sum() == 0):
            estation.append(station)
            tensor.append(np.column_stack((frame, target)))
            pos[station] = lat_long
    return np.array(tensor), estation


def medir_tiempo(funcion):
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = funcion(*args, **kwargs)
        fin = time.time()
        print(f"Tiempo de ejecución de {funcion.__name__}: {fin - inicio:.4f} segundos")
        return resultado

    return wrapper


def create_sequences(X, Y, w, p=7):
    xs, ys = [], []
    for i in range(len(X) - w):
        if i + w + p <= len(X):
            x = X[i : i + w, :]
            y = Y[i + w : i + w + p]
            xs.append(x)
            ys.append(y)
        else:
            break
    return np.array(xs), np.array(ys)


def partition(data, percentage=0.8):
    series_train, series_test = [], []
    for series in data:
        split_idx = int(len(series) * percentage)
        series_train.append(series[:split_idx])
        series_test.append(series[split_idx:])
    return np.vstack(series_train), np.vstack(series_test)


def prepare_data(data, w, p):
    scaler = StandardScaler().fit(data)
    X, y = data[:, :-1], data[:, -1]
    X = scaler.fit_transform(X)
    X, y = create_sequences(X, y, w, p)
    idx = int(0.8 * len(X))
    x_train, x_test = X[:idx, :], X[idx:, :]
    y_train, y_test = y[:idx], y[idx:]
    return x_train, y_train, x_test, y_test, scaler


def lstm_model(window, cols, p=7):
    model = Sequential()
    model.add(LSTM(32, input_shape=(window, cols), return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(p))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def fit_model(X, y, w, p):
    global_model = lstm_model(w, X.shape[1], p)
    global_model.fit(
        X,
        y,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )
    return global_model


if __name__ == "__main__":

    path = r"/data/samuelrt/data/data_17_18.parquet"
    var = [
        "latitude",
        "longitude",
        "humidity",
        "temperature",
        "wind",
        "pressure",
        "precipitation",
    ]
    w, p = 7, 1
    resultados = {}

    tensor, station = load_data(path, var, 730, "1D")

    with open("tensor.pkl", "rb") as archivo:
        tensor = pickle.load(archivo)

    print(tensor.shape, "\n")

    i = 0
    for t in tensor:
        print(t.shape)
        x_train, y_train, x_test, y_test, scaler = prepare_data(t, w, p)
        # ---------------Training-----------------------#
        start_time = time.time()
        model = fit_model(x_train, y_train, w, p)
        end_time = time.time()
        training_time = round(end_time - start_time, 3)
        # ----------------------Testing----------------#
        pred = model.predict(x_test)
        resultados[i] = [y_test, pred, training_time]
        i = i + 1

    with open("chuva_1D_one.pkl", "wb") as archivo:
        pickle.dump(resultados, archivo)

    df = pd.DataFrame(columns=["time", "station"])
    df.loc[:, "time"] = training_time
    df.loc[:, "station"] = station
    df.to_csv("data.csv", index_label=False)
