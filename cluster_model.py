import pandas as pd
import numpy as np
import pyarrow as py
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pickle
import warnings

warnings.filterwarnings("ignore")


def medir_tiempo(funcion):
    def wrapper(*args, **kwargs):
        inicio = time.time()  # Marca el inicio
        resultado = funcion(*args, **kwargs)  # Llama a la función original
        fin = time.time()  # Marca el fin
        print(f"Tiempo de ejecución de {funcion.__name__}: {fin - inicio:.4f} segundos")
        return resultado  # Devuelve el resultado de la función original

    return wrapper


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True, verbose=0
)


def create_sequences(X, Y, w, p=7):
    xs, ys = [], []
    for i in range(len(X) - w):
        # Verificar si quedan al menos `p` elementos después de `i + w`
        if i + w + p <= len(X):
            # x = data[i:i + w, :-1]
            x = X[i : i + w, :]
            # y = data[i + w:i + w + p, -1]
            y = Y[i + w : i + w + p]
            xs.append(x)
            ys.append(y)
        else:
            # Salir del bucle si no se puede generar una secuencia válida
            break
    return np.array(xs), np.array(ys)


def inverse_scaled(y_test, p, scaler):
    y_test_inverse = []
    for y_seq in y_test:
        # Construir una fila ficticia para invertir solo el último valor
        padded = np.zeros((p, scaler.n_features_in_))
        padded[:, -1] = y_seq  # Colocar valores de y_seq en la última columna
        y_inverse = scaler.inverse_transform(padded)  # Inversión
        y_test_inverse.append(y_inverse[:, -1])  # Extraer valores restaurados
    y_test = np.array(y_test_inverse)
    return y_test


def partition(data, percentage=0.8):
    series_train, series_test = [], []
    for series in data:
        split_idx = int(len(series) * percentage)
        series_train.append(series[:split_idx])
        series_test.append(series[split_idx:])
    return np.vstack(series_train), np.vstack(series_test)


def prepare_data(data, w, p):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train, X_test = partition(data)

    x_train, y_train = X_train[:, :-1], X_train[:, -1]
    x_test, y_test = X_test[:, :-1], X_test[:, -1]

    scaler_x.fit(np.vstack((x_train, x_test)))
    scaler_y.fit(np.vstack((y_train.reshape(-1, 1), y_test.reshape(-1, 1))))

    x_train_scaled = scaler_x.transform(x_train)
    x_test_scaled = scaler_x.transform(x_test)
    y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1))
    # y_test_scaled = scaler_y.transform(y_test.reshape(-1,1))

    x_train, y_train = create_sequences(x_train_scaled, y_train_scaled, w, p)
    x_test, y_test = create_sequences(x_test_scaled, y_test, w, p)

    return x_train, y_train, x_test, y_test, scaler_y


@medir_tiempo
def fit_global_model(X, y, w, p):
    global_model = lstm_model(w, X.shape[2], p)
    global_model.fit(
        X,
        y,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )
    return global_model


@medir_tiempo
def fit_cluster_models(cluster_values, w, p):
    cluster_models = {}
    cluster_test = {}
    for cluster_label, tensor in cluster_values.items():
        x_train, y_train, x_test, y_test, scaler = prepare_data(tensor, w, p)
        model = lstm_model(w, x_train.shape[2], p)
        model.fit(
            x_train,
            y_train,
            epochs=100,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0,
        )
        cluster_models[cluster_label] = [model, scaler]
        cluster_test[cluster_label] = [x_test, y_test]

    return cluster_models, cluster_test


def lstm_model(window, cols, p=7):
    print(window, cols, p)
    model = Sequential()
    model.add(LSTM(32, input_shape=(window, cols), return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(p))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def prediction(model, x_test, p, scaler):
    pred = model.predict(x_test)
    pred = inverse_scaled(pred, p, scaler)
    return pred


def model_cluster_predict(cluster_models, cluster_test, p):
    clusters_predictions = {}
    for label in cluster_models.keys():
        pred = prediction(
            cluster_models[label][0],
            cluster_test[label][0],
            p,
            cluster_models[label][1],
        )
        clusters_predictions[label] = [cluster_test[label][1], pred]
    return clusters_predictions


if __name__ == "__main__":

    start_time = time.time()

    w = 14
    p = 2
    # global
    with open("/data/samuelrt/kmedoids/K_chuva_12H.pkl", "rb") as archivo:
        tensor = pickle.load(archivo)
    # clusters
    with open("/data/samuelrt/kmedoids/cluster_K_chuva_12H.pkl", "rb") as archivo:
        cluster_values = pickle.load(archivo)

    x_train, y_train, x_test, y_test, global_scaler = prepare_data(tensor, w, p)

    global_model = fit_global_model(x_train, y_train, w, p)

    cluster_models, cluster_test = fit_cluster_models(cluster_values, w, p)

    global_prediction = prediction(global_model, x_test, p, global_scaler)

    global_predictions = {"Global": [y_test, global_prediction]}

    cluster_predictions = model_cluster_predict(cluster_models, cluster_test, p)

    results = global_predictions | cluster_predictions

    with open("R_chuva_12H.pkl", "wb") as archivo:
        pickle.dump(results, archivo)

    end_time = time.time()
    print("time", end_time - start_time)
