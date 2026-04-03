import pandas as pd
import numpy as np
import pyarrow as py
from sklearn_extra.cluster import KMedoids
from tslearn.metrics import cdist_dtw
from tslearn.clustering import silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import pickle
import time


def load_data(path, var, shape=365, freq="1D"):
    df = pd.read_parquet(path, engine="pyarrow")
    df.set_index(pd.to_datetime(df["date"]), inplace=True)
    df.drop("date", axis=1, inplace=True)
    estation = []  # store names from each station
    tensor = []  # store lat an long from each station
    # values = {} # store lat long and phtw
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


def get_clusters(tensor, chuva=True):
    if chuva:
        nuevo_tensor = tensor[:, :, :]
    else:
        nuevo_tensor = tensor[:, :, :-1]

    scaler = TimeSeriesScalerMeanVariance()
    X = scaler.fit_transform(nuevo_tensor)
    dtw_distance_matrix = cdist_dtw(X, n_jobs=-1)
    scores_dtw = []

    for k in range(3, 15):
        kdoids_dtw = KMedoids(n_clusters=k, metric="precomputed", random_state=0)
        kdoids_dtw.fit(dtw_distance_matrix)
        labels = kdoids_dtw.labels_
        scores_dtw.append(
            (silhouette_score(X, labels, metric="dtw", n_jobs=24), labels, k)
        )

    max_dtw = max(scores_dtw, key=lambda x: x[0])
    return max_dtw, tensor


def cluster_data(max_dtw, tensor):  # max_dtw = (sc,labels,k)
    # n,t,c = tensor.shape
    cluster_values = {}
    for k in set(max_dtw[1]):
        cluster_values[k] = list()
    for idx, label in enumerate(max_dtw[1]):
        cluster_values[label].append(tensor[idx])
    cluster_values = {key: np.array(value) for key, value in cluster_values.items()}
    for k in cluster_values.keys():
        print(len(cluster_values[k]))
    return cluster_values, tensor  # tensor.reshape(n*t,c)


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

    start_time = time.time()

    tensor, estation = load_data(path, var, 2920, "6h")

    max_dtw, _ = get_clusters(tensor, chuva=True)

    cluster_values, tensor = cluster_data(max_dtw, tensor)

    with open("K_chuva_6H.pkl", "wb") as archivo:
        pickle.dump(tensor, archivo)

    with open("cluster_K_chuva_6H.pkl", "wb") as archivo:
        pickle.dump(cluster_values, archivo)

    data = pd.DataFrame()
    data.loc[:, "station"] = estation
    data.loc[:, f"D{max_dtw[2]}"] = max_dtw[1]
    data.to_csv("dtw_chuva_6H.csv", index=False)

    end_time = time.time()

    print("time", end_time - start_time)
