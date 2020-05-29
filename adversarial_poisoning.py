import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from read_dataset import read_synth, read_iris, read_mnist, read_breast_cancer, read_wine, read_image_segment, \
    read_soybean, read_ionosphere
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import sys
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd
from itertools import combinations
import seaborn as sns

def clear_figures():
    plt.clf()
    plt.cla()
    plt.close()


def normalize(vector, remove_negative_probabilities=False):
    ret = vector.copy()

    if remove_negative_probabilities:
        ret[ret <= 0] = 0

    return ret / np.linalg.norm(ret, ord=1)


def plot_new_dataset(x, y, idxs_to_change, run=0):

    if x.shape[1] > 2:
        pca = PCA(n_components=2)
        new_x = pca.fit_transform(x)
    else:
        new_x = x.copy()

    new_y = y.copy()
    new_y[idxs_to_change] = "poisoned"

    df = pd.DataFrame({"c1": new_x[:, 0], "c2": new_x[:, 1], "label": new_y})

    sns.scatterplot(x="c1", y="c2", hue="label", data=df)

    plt.savefig(f"./results/poisoned_image_segment/{run}.png")
    clear_figures()


def run_tests(x, y):
    accuracies = []
    # Repeat N_TESTS times the N_FOLD-Fold crossvalidation
    for _ in tqdm(range(N_TESTS), desc="Tests", total=N_TESTS,leave=False):
        for train_index, test_index in kf.split(x):
            knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            predictions = knn.fit(x_train, y_train).predict(x_test)
            accuracies.append(accuracy_score(y_test, predictions))

    return sum(accuracies) / len(accuracies)


def rename(indices, labels, y, rename_map=None):
    def _random_rename(indices, labels, y):
        new_y = y.copy()
        labels_to_change = new_y[indices]
        for idx, label_to_change in enumerate(labels_to_change):
            to_remove = {label_to_change}
            possible_labels = labels - to_remove
            new_label = np.random.choice(list(possible_labels))
            new_y[indices[idx]] = new_label

        return new_y

    def _map_rename(indices, labels, y):
        new_y = y.copy()
        labels_to_change = new_y[indices]
        for idx, label_to_change in enumerate(labels_to_change):
            new_y[indices[idx]] = rename_map[label_to_change]

        return new_y

    if rename_map is None:
        return _random_rename(indices, labels, y)
    else:
        return _map_rename(indices, labels, y)


def build_rename_map(x, y):
    unique_labels = np.unique(y)
    assert len(unique_labels) >= 2
    if len(unique_labels) == 2:
        return {
            unique_labels[0]: unique_labels[1],
            unique_labels[1]: unique_labels[0]
        }

    centroids = []
    for label in unique_labels:
        centroids.append(x[y == label].sum(axis=0) / x.shape[0])

    correlations = np.corrcoef(centroids)
    rename_map = dict([(val, unique_labels[np.argmin(correlations[:,idx])]) for idx, val in enumerate(unique_labels)])
    return rename_map


TEST_SIZE = 0.8
N_FOLDS = 10
N_TESTS = 10
EPOCHS = 10
POISONED_FRACTION = 0.05
ETA = 0.01
N_RUNS = 5


if __name__ == "__main__":
    clear_figures()
    dataset_name = "ionosphere"
    x, y = read_ionosphere("./datasets/ionosphere/ionosphere.csv")
    # x, y = read_soybean("./datasets/soybean/soybean-large.csv")
    # x, y = read_iris()
    # x, y = read_synth("./datasets/poisoned_synthetic/xt_lm.mat", "./datasets/poisoned_synthetic/y_lm.mat")
    # x, y = read_breast_cancer("./datasets/breast_cancer/breast-cancer-wisconsin.data")
    # x, y = read_wine()
    # x, y = read_image_segment("./datasets/image_segment/segmentation.test")
    worst_y = y.copy()
    worst_accuracy = 1
    for rename_map in [None, build_rename_map(x, y)]:
        indices_to_poison = []
        for run in range(N_RUNS):
            sys.stdout.write(f"\rRun: {run}\n")
            sys.stdout.flush()
            # Use multiplicative weights to adversarially poison the synthetic dataset

            labels = set(y)
            probability_distribution = np.array([1 / len(x) for _ in range(len(x))])
            kf = KFold(n_splits=N_FOLDS, shuffle=True)
            baseline_accuracy = run_tests(x, y)
            prev_accuracy = baseline_accuracy
            for i in tqdm(range(EPOCHS), desc="Epochs", total=EPOCHS):

                # Take POISONED_FRACTION elements based off initial_probability_distribution
                indices = np.random.choice(len(x), int(len(x) * POISONED_FRACTION), p=probability_distribution)
                new_y = rename(indices, labels, y, rename_map=rename_map)

                new_accuracy = run_tests(x, new_y)
                delta = prev_accuracy - new_accuracy

                if delta > 0:
                    prev_accuracy = new_accuracy

                    worst_accuracy = new_accuracy
                    worst_y = new_y.copy()

                    for idx in indices:
                        probability_distribution[idx] = probability_distribution[idx] + ETA * np.sign(delta) * np.exp(
                            delta)

                probability_distribution = normalize(probability_distribution, remove_negative_probabilities=True)

            top_k = sorted([(idx, val) for idx, val in enumerate(list(probability_distribution))], key=lambda xx: xx[1],
                           reverse=True)[:int(POISONED_FRACTION * len(x))]

            idxs_to_change = [idx for idx, _ in top_k]
            new_y = rename(idxs_to_change, labels, y, rename_map=rename_map)

            acc = run_tests(x, new_y)

            # plot_new_dataset(x, y, idxs_to_change, run=run)

            print(
                f"Original accuracy was -> {baseline_accuracy * 100}%\nCurrent -> {acc * 100}%\nDelta -> {(baseline_accuracy - acc) * 100}%")
            indices_to_poison.extend(idxs_to_change)


        # # Remove duplicates
        # indices_to_poison = list(set(indices_to_poison))
        # poisoned_points = x[indices_to_poison]
        # #print(y[indices_to_poison])
        # correlation = np.corrcoef(poisoned_points)
        # #print(correlation)
        # np.savetxt("correlations.txt", correlation, fmt="%.5f")
        # #print(len(indices_to_poison))
        # sns.heatmap(correlation)
        # plt.savefig("./results/poisoned_image_segment/correlation.png")
        # clear_figures()
    df = pd.DataFrame(x)
    df["class"] = pd.Series(worst_y)
    df.to_csv(f"./datasets/poisoned/{dataset_name}_poisoned.csv", index=False, header=False)
