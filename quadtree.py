from tslearn.metrics import cdist_dtw
import numpy as np
import pandas as pd
from scipy.stats import entropy
import time
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
import pickle


def dibujar_limite(boundary, ax, color="r"):
    x1, y1, x2, y2 = boundary
    ax.plot([x1, x2], [y1, y1], color)
    ax.plot([x1, x1], [y1, y2], color)
    ax.plot([x1, x2], [y2, y2], color)
    ax.plot([x2, x2], [y1, y2], color)


def get_entropy(node, entropies):
    if node is not None:
        entropies.append(node.entropy)
        for child in node.children:
            get_entropy(child, entropies)
    return entropies


def divide_boundary(boundary):
    x_min, y_min, x_max, y_max = boundary
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2

    boundaries = [
        (x_min, y_min, mid_x, mid_y),
        (mid_x, y_min, x_max, mid_y),
        (x_min, mid_y, mid_x, y_max),
        (mid_x, mid_y, x_max, y_max),
    ]

    return boundaries


def get_points_in_boundary(data, boundary):
    x_min, y_min, x_max, y_max = boundary
    return [
        point
        for point in data
        if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
    ]


def similarity(data, n):
    info_children = []
    for lg in data:
        info_children.append(values[lg])
    info_children = np.array(info_children)
    distances = cdist_dtw(info_children, n_jobs=48)
    vector_no_cero = distances[np.tril_indices(len(info_children), -1)]
    N = len(vector_no_cero)
    if N > 0:
        H_max = np.log2(n)
        H = entropy(vector_no_cero, base=2)
        H_normalizada = H / H_max
        return H_normalizada
    else:
        return False


class QuadTreeNode:
    def __init__(self, data, boundary, cluster_id, n=0):
        self.data = data
        self.boundary = boundary
        self.children = []
        # self.entropy = similarity(data,n)
        self.cluster_id = cluster_id


def build_quadtree(
    data, boundary, epsilon, min_points, clusters, cluster_id=0, node_id=[0], n=0
):
    """
    Construye un quadtree, asignando IDs solo a nodos que tienen elementos.

    Args:
        data: Lista de puntos en el nodo actual.
        boundary: Límite del nodo actual (x_min, y_min, x_max, y_max).
        epsilon: Umbral de entropía para dividir nodos.
        min_points: Número mínimo de puntos para dividir nodos.
        cluster_id: ID inicial del nodo actual.
        node_id: Lista que actúa como contador global para IDs únicos.

    Returns:
        QuadTreeNode: Nodo raíz del quadtree construido.
    """
    # Verifica si hay datos en este nodo
    if not data:
        return None

    # Crea el nodo actual
    node = QuadTreeNode(data, boundary, cluster_id=node_id[0], n=n)
    node_id[0] += 1  # Incrementa el ID solo para nodos creados con datos

    # Verifica si el nodo debe subdividirse
    if (len(data) > min_points) and (similarity(data, n) > epsilon):
        sub_boundaries = divide_boundary(boundary)
        for idx, sub_boundary in enumerate(sub_boundaries):
            sub_data = get_points_in_boundary(data, sub_boundary)
            # Asegura que haya suficientes puntos para crear subnodos
            dissimilatiry = similarity(sub_data, n)
            if (dissimilatiry > epsilon) and (len(sub_data) > min_points):
                node.children.append(
                    build_quadtree(
                        sub_data,
                        sub_boundary,
                        epsilon,
                        min_points,
                        clusters,
                        cluster_id=4 * cluster_id + idx + 1,
                        node_id=node_id,
                        n=n,
                    )
                )
            else:
                if len(sub_data) != 0:
                    clusters[4 * cluster_id + idx] = sub_data
                    print(dissimilatiry)
    return node


def get_data(path, var, shape, freq, chuva=True):
    df = pd.read_parquet(path, engine="pyarrow")
    df.set_index(pd.to_datetime(df["date"]), inplace=True)
    df.drop("date", axis=1, inplace=True)
    geo = []  # list of lat an long from each station
    values = {}  # dict of lat long and phtw
    values_complete = {}
    pos = {}
    r = lambda x: (round(x[0], 4), round(x[1], 4))

    for station in df["station"].unique():
        frame = df.loc[df["station"] == station].loc[:, var]
        frame = frame.sort_index()
        lat_long = r(
            tuple(frame[["longitude", "latitude"]].drop_duplicates().values.flatten())
        )
        data = frame.loc[:, var[2:-1]]
        target = frame.loc[:, var[-1]]
        frame = data.resample(rule=freq).mean().values
        target = target.resample(rule=freq).sum().values
        if (frame.shape[0] == shape) and (np.isnan(frame).sum() == 0):
            geo.append(lat_long)
            pos[lat_long] = station
            if chuva:
                frame = np.column_stack((frame, target))
            values[lat_long] = StandardScaler().fit_transform(frame)
            values_complete[lat_long] = np.column_stack(
                (frame, target)
            )  # here is the complete information with no tranformations

    return geo, values, pos, values_complete


if __name__ == "__main__":
    start_time = time.time()
    path = "/data/samuelrt/data/data_17_18.parquet"

    var = [
        "latitude",
        "longitude",
        "humidity",
        "temperature",
        "wind",
        "pressure",
        "precipitation",
    ]

    data, values, pos, values_c = get_data(path, var, shape=2920, freq="6h", chuva=True)

    x = [i[0] for i in data]
    y = [i[1] for i in data]
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    boundary = (x_min, y_min, x_max, y_max)
    epsilon_values = [0.5]
    min_points_values = [60]
    N = (len(data) - 1) * len(data) * 0.5
    quads = []
    clusters = {}

    for epsilon, min_points in itertools.product(epsilon_values, min_points_values):
        quads.append(
            (
                epsilon,
                min_points,
                build_quadtree(
                    data=data,
                    boundary=boundary,
                    epsilon=epsilon,
                    min_points=min_points,
                    clusters=clusters,
                    n=N,
                ),
            )
        )

    data_cluster_Q = {}
    for k, v in clusters.items():
        data_cluster_Q[k] = []
        for latlong in v:
            data_cluster_Q[k].append(values_c[latlong])

    data_cluster_Q = {key: np.array(value) for key, value in data_cluster_Q.items()}

    # with open('clustQ_chuva_6H.pkl', 'wb') as archivo:
    #         pickle.dump(data_cluster_Q, archivo) # complete information
    with open("clustQ_chuva_latlong_6H.pkl", "wb") as archivo:
        pickle.dump(clusters, archivo)  # lat long for clustering

    end_time = time.time()
    print("time", end_time - start_time)
