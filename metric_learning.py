from read_dataset import load_datasets
from tqdm import tqdm
from metric_learn import ITML_Supervised, LMNN
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    for x, y, dataset_name in tqdm(load_datasets(), desc="Datasets"):
        itml = ITML_Supervised(max_iter=100)
        x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.5)


        G = itml.fit(x_train, y_train).get_mahalanobis_matrix()
        print(G)
        break
