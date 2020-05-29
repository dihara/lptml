import numpy as np
import pandas as pd
from string import ascii_uppercase
from struct import unpack
from tqdm import tqdm
from sklearn import datasets
from scipy.io import loadmat
import IPython
import os
RUN_ALL_EXPERIMENTS = True

def readInt32(num):
    return unpack(">I", num)[0]


def readInt8(num):
    return unpack(">B", num)[0]


def read_x_mnist(x_path, normalize=False):
    with open(x_path, 'rb') as f:
        # skip the first 4 bytes, from the beginning of the file
        f.seek(4, 0)
        n_samples = readInt32(f.read(4))
        n_rows = readInt32(f.read(4))
        n_cols = readInt32(f.read(4))
        x = []

        for i in tqdm(range(n_samples), desc="[MNIST] Read values", leave=False):
            image = []
            for j in range(n_rows * n_cols):
                image.append(readInt8(f.read(1)))
            x.append(np.array(image).reshape(784, 1))

        ret = np.array(x)

        if normalize:
            return ret / np.max(ret) - 0.5
        else:
            return ret


def read_y_mnist(y_path):
    with open(y_path, 'rb') as f:
        # skip the first 4 bytes, from the beginning of the file
        f.seek(4, 0)
        n_samples = readInt32(f.read(4))

        y = []
        for i in tqdm(range(n_samples), desc="[MNIST] Read labels", leave=False):
            idx = readInt8(f.read(1))
            y.append(idx)
        return np.array(y)


def read_german_credit(path):
    with open(path, "r") as f:
        x = []
        y = []
        lines = [line.strip() for line in f.readlines()]

        for line in lines:
            elements = [int(el) for el in line.split(" ") if el]
            x.append(elements[:-1])
            y.append(elements[-1])

    return np.array(x), np.array(y)


def read_image_segment(path):
    df = pd.read_csv(path)
    y = df["LABEL"]
    x = df[["REGION-CENTROID-COL", "REGION-CENTROID-ROW", "REGION-PIXEL-COUNT", "SHORT-LINE-DENSITY-5",
            "SHORT-LINE-DENSITY-2", "VEDGE-MEAN", "VEDGE-SD", "HEDGE-MEAN", "HEDGE-SD", "INTENSITY-MEAN", "RAWRED-MEAN",
            "RAWBLUE-MEAN", "RAWGREEN-MEAN", "EXRED-MEAN", "EXBLUE-MEAN", "EXGREEN-MEAN", "VALUE-MEAN",
            "SATURATION-MEAN", "HUE-MEAN"]]
    return x.to_numpy(), y.to_numpy()


def read_iris(poisoned=False):
    if poisoned:
        df = pd.read_csv("./datasets/poisoned/iris_poisoned.csv", header=None)
        return df.loc[:, df.columns != 4].to_numpy(), df[4].to_numpy()
    else:
        iris = datasets.load_iris()
        return iris.data, iris.target


def read_isolet(path):
    df = pd.read_csv(path)
    attributes = [f"f{i}" for i in range(1, 618)]
    x = df[attributes].to_numpy()
    y = df["class"].apply(lambda x: x.replace("'", "")).apply(lambda x: int(x)).to_numpy()
    return x, y


def read_letters(path):
    df = pd.read_csv(path)
    y = df["Letter"]
    x = df.loc[:, df.columns != 'Letter']
    return x.to_numpy(), y.to_numpy()


def read_mnist(x_path, y_path):
    x = read_x_mnist(x_path).reshape((10000, 784))
    y = read_y_mnist(y_path)
    return x[:4000, :], y[:4000]


def read_breast_cancer(path):
    df = pd.read_csv(path)
    df = df.loc[:, df.columns != 'id']
    x  = df.loc[:, df.columns != 'class']
    y  = df["class"]

    return x.to_numpy(), y.to_numpy()


def read_vehicle(path):
    df = pd.read_csv(path, header=None)
    x  = df.loc[:, df.columns != 18]
    y  = df[18]

    return x.to_numpy(), y.to_numpy()


def read_wine():
    wine = datasets.load_wine()
    return wine.data, wine.target


def read_soybean(path):
    df = pd.read_csv(path, header=None).interpolate(method="linear")
    x = df.loc[:, df.columns != 0]
    y = df[0]
    labels_map = dict([(l,i) for i,l in enumerate(y.unique())])
    y = y.apply(lambda x: labels_map[x])
    return x.to_numpy(), y.to_numpy()


def read_ionosphere(path):
    df = pd.read_csv(path, header=None)
    x = df.loc[:, df.columns != 34].fillna(0)
    y = df[34]
    return x.to_numpy(), y.to_numpy()


def read_poisoned_synth(x_path, y_path):
    x = loadmat(x_path)
    y = loadmat(y_path)
    return x["XXX"].T, y["YYY"].T.reshape((106,))


def read_synth(x_path, y_path):
    # Remove the last 5 points (the poisoned points)
    x, y = read_poisoned_synth(x_path, y_path)
    return x[:100, :], y[:100]


def load_poisoned():
    # #yield (*read_iris(poisoned=True), "poisoned_iris")
    # yield (*read_soybean('./datasets/poisoned/soybean_poisoned.csv'), "poisoned_soybean")
    # # yield (*read_poisoned_synth("./datasets/poisoned_synthetic/xt_lm.mat","./datasets/poisoned_synthetic/y_lm.mat"), "poisoned_synth")
    # # yield (*read_ionosphere('./datasets/poisoned/ionosphere_poisoned.csv'), "poisoned_ionosphere")

    for dataset_name in os.listdir("./datasets/fully_random_poisoned"):

        if dataset_name in ["wine_poisoned.csv", "ionosphere_poisoned.csv", "iris_poisoned.csv", "synthetic_poisoned.csv"]:
            continue

        path = f"./datasets/fully_random_poisoned/{dataset_name}"

        df = pd.read_csv(path, header=None)
        class_idx = len(df.columns) - 1

        x = df.loc[:, df.columns != class_idx]
        y = df[class_idx]
        yield x.to_numpy(), y.to_numpy(), dataset_name.split(".")[0]



def load_datasets():
    #
    # #Synthetic dataset
    # x_synth, y_synth = read_synth("./datasets/poisoned_synthetic/xt_lm.mat",
    #                                        "./datasets/poisoned_synthetic/y_lm.mat")
    # yield x_synth, y_synth, "synthetic"
    #
    # # Soybean dataset
    # x_soybean, y_soybean = read_soybean("datasets/soybean/soybean-large.csv")
    # yield x_soybean, y_soybean, "soybean_real"
    #
    # # Iris dataset
    # x_iris, y_iris = read_iris()
    # yield x_iris, y_iris, "iris"
    #
    # # Wine dataset
    # x_wine, y_wine = read_wine()
    # yield x_wine, y_wine, "wine"
    #
    # # Ionosphere dataset
    # x_ionosphere, y_ionosphere = read_ionosphere("./datasets/ionosphere/ionosphere.csv")
    # yield x_ionosphere, y_ionosphere, "ionosphere"

    if RUN_ALL_EXPERIMENTS:
        # Image segment dataset
        x_is, y_is = read_image_segment(
            "./datasets/image_segment/segmentation.test")  # pd.read_csv("./datasets/german_credit/german_credit.tsv", sep="\t")
        yield x_is, y_is, "poisoned_image_segment"

        # Breast cancer dataset
        x_bc, y_bc = read_breast_cancer("./datasets/breast_cancer/breast-cancer-wisconsin.data")
        yield x_bc, y_bc, "breast_cancer"

        # Vehicle dataset
        x_vehicle, y_vehicle = read_vehicle("./datasets/vehicle/xa.csv")
        yield x_vehicle, y_vehicle, "vehicle"

        # German Credit dataset
        x_gc, y_gc = read_german_credit(
            "./datasets/german_credit/german_credit.tsv")  # pd.read_csv("./datasets/german_credit/german_credit.tsv", sep="\t")
        yield x_gc, y_gc, "german_credit"

        # Isolet dataset
        x_isolet, y_isolet = read_isolet("./datasets/isolet/isolet_csv.csv")
        yield x_isolet, y_isolet, "isolet"

        # Letters dataset
        x_letters, y_letters = read_letters("./datasets/letters/letters.csv")
        yield x_letters, y_letters, "letters"

        # MNIST dataset
        x_mnist, y_mnist = read_mnist("./datasets/mnist/t10k-images-idx3-ubyte",
                                      "./datasets/mnist/t10k-labels-idx1-ubyte")
        yield x_mnist, y_mnist, "mnist"

