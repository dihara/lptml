import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from read_dataset import read_synth, read_iris, read_mnist, read_breast_cancer, read_wine, read_image_segment, \
    read_soybean, read_ionosphere, load_datasets
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import sys
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd
from itertools import combinations
import seaborn as sns
from itertools import product

def clear_figures():
    plt.clf()
    plt.cla()
    plt.close()


def normalize(vector, remove_negative_probabilities=False):
    ret = vector.copy()

    if remove_negative_probabilities:
        ret[ret <= 0] = 0

    return ret / np.linalg.norm(ret, ord=1)


def plot_new_dataset(x, y, name="none"):
    if x.shape[1] > 2:
        pca = PCA(n_components=2)
        new_x = pca.fit_transform(x)
    else:
        new_x = x.copy()

    new_y = y.copy()
    limit = int(new_x.shape[0] * (1 - POISONED_FRACTION))
    df_real = pd.DataFrame({"c1": new_x[:limit, 0], "c2": new_x[:limit, 1], "label": new_y[:limit]})
    df_poisoned = pd.DataFrame({"c1": new_x[limit:, 0], "c2": new_x[limit:, 1], "poisoned label": new_y[limit:]})
    sns.scatterplot(x="c1", y="c2", hue="label", data=df_real)
    sns.scatterplot(x="c1", y="c2", hue="poisoned label", data=df_poisoned, palette=sns.color_palette("cubehelix", len(df_poisoned["poisoned label"].unique())))

    plt.savefig(f"./results/images/random_poisoning/{name}.png")
    clear_figures()


def run_tests(x, y):
    accuracies = []
    # Repeat N_TESTS times the N_FOLD-Fold crossvalidation
    for _ in tqdm(range(N_TESTS), desc="Tests", total=N_TESTS, leave=False):
        for train_index, test_index in kf.split(x):
            knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            predictions = knn.fit(x_train, y_train).predict(x_test)
            accuracies.append(accuracy_score(y_test, predictions))

    return sum(accuracies) / len(accuracies)


def rename(indices, labels, y, rename_map=None, random=False):
    def _selective_random_rename(indices, labels, y):
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

    def _random_rename(indices, labels, y):
        new_y = y.copy()
        labels_to_change = new_y[indices]
        for idx, _ in enumerate(labels_to_change):
            new_y[idx] = np.random.choice(list(labels))
        return new_y


    if random:
        return _random_rename(indices, labels, y)
    else:
        if rename_map is None:
            return _selective_random_rename(indices, labels, y)
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
    rename_map = dict([(val, unique_labels[np.argmin(correlations[:, idx])]) for idx, val in enumerate(unique_labels)])
    return rename_map


TEST_SIZE = 0.8
N_FOLDS = 10
N_TESTS = 10
EPOCHS = 10
POISONED_FRACTION = 0.1
ETA = 0.01
N_RUNS = 5
SCALE_FACTOR = 3


def get_poisoned_points(x):
    bounds = []
    for column_idx in range(x.shape[1]):
        bounds.append((np.min(x[:, column_idx]), np.max(x[:, column_idx])))
    poisoning_bounds = []
    for l, h in bounds:
        #                        low,           high,            range
        poisoning_bounds.append((h, h + (h - l) * SCALE_FACTOR, (h - l)))
    hypercube_centers_for_dimension = []
    for dimension in range(len(poisoning_bounds)):
        centers_for_d = []
        l, h, r = poisoning_bounds[dimension]
        base_value = l
        while base_value < h:
            center = base_value + r
            base_value += r
            centers_for_d.append(center)

        hypercube_centers_for_dimension.append(centers_for_d)
    hypercube_centers = []
    for indices in product(range(SCALE_FACTOR), repeat=len(hypercube_centers_for_dimension)):
        hypercube_centers.append([0 for _ in range(len(hypercube_centers_for_dimension))])
        for dimension_idx, center_idx in enumerate(indices):
            hypercube_centers[-1][dimension_idx] = hypercube_centers_for_dimension[dimension_idx][center_idx]
    random_points_for_center = [[0 for _ in range(len(hypercube_centers[0]))] for _ in range(len(hypercube_centers))]
    ranges = [x[2] for x in poisoning_bounds]
    for i in tqdm(range(len(hypercube_centers)), desc="Creating points"):
        for j in range(len(hypercube_centers[i])):
            diameter = ranges[j]
            l_range = hypercube_centers[i][j] - diameter / np.sqrt(len(ranges)) / 2
            h_range = hypercube_centers[i][j] + diameter / np.sqrt(len(ranges)) / 2
            random_points_for_center[i][j] = np.random.uniform(hypercube_centers[i][j], h_range - l_range)

    return np.array(random_points_for_center)


if __name__ == "__main__":
    clear_figures()

    for x, y, dataset_name in tqdm(load_datasets(), desc="Datasets"):
        print(f"doing -> {dataset_name}")
        if x.shape[1] > 10:
            pca = PCA(n_components=10)
            x_pca = pca.fit_transform(x)
        else:
            x_pca = x.copy()
        worst_x = x_pca.copy()
        worst_y = y.copy()
        worst_accuracy = 1

        try:
            poisoned_points = get_poisoned_points(x_pca)
        except Exception as e:
            print(e)
            print(f"Skipping {dataset_name}")
            continue

        # Initial run: Classify the adversarial points!
        predicted_y = KNeighborsClassifier(n_neighbors=5).fit(x_pca, y).predict(poisoned_points)
        poisoned_labels = rename([i for i in range(len(predicted_y) - 1)], {*np.unique(y)}, predicted_y, random=True)

        print("Let the battle begin!")

        for rename_map in [None]:
            indices_to_poison = []
            for run in range(N_RUNS):
                sys.stdout.write(f"\rRun: {run}\n")
                sys.stdout.flush()
                # Use multiplicative weights to adversarially poison the synthetic dataset
                labels = set(y)
                probability_distribution = np.array([1 / len(poisoned_points) for _ in range(len(poisoned_points))])
                kf = KFold(n_splits=N_FOLDS, shuffle=True)
                baseline_accuracy = run_tests(x, y)
                prev_accuracy = baseline_accuracy
                for i in tqdm(range(EPOCHS), desc="Epochs", total=EPOCHS):

                    # Take POISONED_FRACTION elements based off initial_probability_distribution
                    indices = np.random.choice(len(poisoned_points), int(len(x) * POISONED_FRACTION), p=probability_distribution)
                    x_to_add = poisoned_points[indices]
                    y_to_add = poisoned_labels[indices]

                    new_x = np.append(x_pca, x_to_add, axis=0)
                    new_y = np.append(y, y_to_add, axis=0)

                    new_accuracy = run_tests(new_x, new_y)
                    delta = prev_accuracy - new_accuracy

                    if delta > 0:
                        prev_accuracy = new_accuracy

                        worst_accuracy = new_accuracy
                        worst_y = new_y.copy()
                        worst_x = new_x.copy()

                        for idx in indices:
                            probability_distribution[idx] = probability_distribution[idx] + ETA * np.sign(delta) * np.exp(
                                delta)

                    probability_distribution = normalize(probability_distribution, remove_negative_probabilities=True)

                top_k = sorted([(idx, val) for idx, val in enumerate(list(probability_distribution))], key=lambda xx: xx[1],
                               reverse=True)[:int(POISONED_FRACTION * len(x))]

                idxs_to_include = [idx for idx, _ in top_k]
                x_to_add = poisoned_points[indices]
                y_to_add = poisoned_labels[indices]

                new_x = np.append(x_pca, x_to_add, axis=0)
                new_y = np.append(y, y_to_add, axis=0)

                acc = run_tests(new_x, new_y)

                print(
                    f"Original accuracy was -> {baseline_accuracy * 100}%\nCurrent -> {acc * 100}%\nDelta -> {(baseline_accuracy - acc) * 100}%")
                indices_to_poison.extend(idxs_to_include)

        df = pd.DataFrame(worst_x)
        df["class"] = pd.Series(worst_y)
        try:
            plot_new_dataset(worst_x, worst_y, name=f"{dataset_name}_poisoned")
        except Exception as e:
            print(e)
            continue
        df.to_csv(f"./datasets/fully_random_poisoned/{dataset_name}_poisoned.csv", index=False, header=False)
