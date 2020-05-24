from read_dataset import load_datasets
from tqdm import tqdm
from metric_learn import LMNN, ITML_Supervised, LFDA, MLKR, NCA, RCA_Supervised, MMC_Supervised, LSML_Supervised
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA


if __name__ == "__main__":

    # with open("results-other.csv", "w+") as f:
    #     f.write(
    #         "algorithm,dataset_name,dataset_dimensions(elements|features|classes),PCA,adversarial_noise,accuracy,precision,recall,f1\n")

    for x, y, dataset_name in tqdm(load_datasets(), desc="Datasets", total=7):
        if dataset_name not in ["iris", "wine", "soybean"]:
            continue
        for MLConstructor in tqdm([RCA_Supervised, MMC_Supervised, LSML_Supervised, ITML_Supervised, NCA, MLKR, LFDA, LMNN], desc=f"Metric Learning Constructor", leave=False):

            for pca_dim in tqdm([2, 4], desc="PCA", leave=False):

                for adversarial_noise in [0, 0.1, 0.2, 0.3]:
                    results = [0 for _ in range(4)]

                    try:

                        if pca_dim:
                            x_pca = PCA(n_components=pca_dim).fit_transform(x)
                        else:
                            x_pca = x.copy()

                        x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.5)

                        try:
                            ml_algo = MLConstructor(max_iter=100)

                        except TypeError:
                            ml_algo = MLConstructor()

                        # From main branch, same generation of label noise
                        if adversarial_noise > 0:
                            all_labels = np.unique(y_train)
                            nss = ShuffleSplit(test_size=adversarial_noise / 100, n_splits=1)
                            for no_noise, yes_noise in nss.split(y_train):
                                for i in yes_noise:
                                    y_train[i] = np.random.choice(np.setdiff1d(all_labels, y_train[i]), 1)

                        G = ml_algo.fit(x_train, y_train).get_mahalanobis_matrix()
                        x_ml_train = np.matmul(G, np.transpose(x_train)).T
                        x_ml_test = np.matmul(G, np.transpose(x_test)).T
                        knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
                        knn.fit(x_ml_train, y_train)

                        y_ml_prediction = knn.predict(x_ml_test)
                        results[0] = accuracy_score(y_test, y_ml_prediction)
                        results[1] = precision_score(y_test, y_ml_prediction, average="macro")
                        results[2] = recall_score(y_test, y_ml_prediction, average="macro")
                        results[3] = f1_score(y_test, y_ml_prediction, average="macro")

                    except ValueError as e:
                        continue

                    with open("results-other.csv", "a+") as f:
                        results_string = f"{MLConstructor.__name__},{dataset_name},({'|'.join([str(el) for el in (*x.shape, len(np.unique(y)))])})," + f"{pca_dim}," + f"{adversarial_noise}," + ','.join(
                            [str(el) for el in results]) + "\n"
                        f.write(results_string)
