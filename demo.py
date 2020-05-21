from time import time
import numpy as np
import lptml
import itertools
import csv
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from read_dataset import read_german_credit, read_image_segment, read_isolet, read_letters, read_mnist
import scipy.io as sio

run_mlwga = False
previous_solution = []

def split_pca_learn_metric(x, y, PCA_dim_by, repetitions, t_size, lptml_iterations, S, D, ut, lt, run_hadoop=False, num_machines=10, label_noise=0, rand_state=-1):
    experiment_results = {}
    global previous_solution

    d = len(x[0])

    if rand_state < 0:
        ss = ShuffleSplit(test_size=1-t_size, n_splits=repetitions)
    else:
        ss = ShuffleSplit(test_size=1 - t_size, n_splits=repetitions, random_state=rand_state)

    for train_index, test_index in ss.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # add label noise by re-sampling a fraction of the training labels
        if label_noise > 0:
            all_labels = np.unique(y_train)
            np.random.seed(rand_state)
            nss = ShuffleSplit(test_size=label_noise/100, n_splits=1, random_state=rand_state)
            for no_noise, yes_noise in nss.split(y_train):
                for i in yes_noise:
                    y_train[i] = np.random.choice(np.setdiff1d(all_labels, y_train[i]), 1);

        np.random.seed(None)

        for reduce_dim_by in PCA_dim_by:
            print("Reducing dimension by", reduce_dim_by)
            dimensions = d - reduce_dim_by
            if reduce_dim_by > 0:
                pca = PCA(n_components=dimensions)
                x_pca_train = pca.fit(x_train).transform(x_train)
                x_pca_test = pca.fit(x_test).transform(x_test)
            else:
                x_pca_train = x_train
                x_pca_test = x_test

            if (ut == 0) and (lt == 0):
                distances = []
                all_pairs = []
                for pair_of_indexes in itertools.combinations(range(0, min(1000, len(x_pca_train))), 2):
                    all_pairs.append(pair_of_indexes)
                    distances.append(np.linalg.norm(x_pca_train[pair_of_indexes[0]] - x_pca_train[pair_of_indexes[1]]))

                u = np.percentile(distances, 10)
                l = np.percentile(distances, 90)
            else:
                u = ut
                l = lt

            previous_solution = []
            # replace use of d by dim from here
            previous_t = 0
            for target_iteration in lptml_iterations:
                t = target_iteration - previous_t
                previous_t = t

                print("Algorithm t=", lptml_iterations)
                if str(reduce_dim_by) not in experiment_results.keys():
                    experiment_results[str(reduce_dim_by)] = {str(target_iteration): []}
                else:
                    if str(target_iteration) not in experiment_results[str(reduce_dim_by)].keys():
                        experiment_results[str(reduce_dim_by)][str(target_iteration)] = []

                iteration_results = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                print('u', u, 'l', l)
                start_time = time()
                if run_mlwga:
                    if len(S) > 0:
                        sim = []
                        dis = []
                        # select those pairs in S & D that are in the training set
                        for j in range(len(S)):
                            if ((S[j][0] - 1) in train_index) and ((S[j][1] - 1) in train_index):
                                # print("here", np.where(train_index == S[j][0]))
                                sim.append([np.where(train_index == (S[j][0] - 1))[0][0], np.where(train_index == (S[j][1] - 1))[0][0]])
                                # print(S[j])

                        for j in range(len(D)):
                            if ((D[j][0] - 1) in train_index) and ((D[j][1] - 1) in train_index):
                                # print(train_index)
                                dis.append([np.where(train_index == (D[j][0] - 1))[0][0], np.where(train_index == (D[j][1] - 1))[0][0]])
                                # print(D[j])

                        G = lptml.fit(x_pca_train, y_train, u, l, t, sim, dis, run_hadoop=run_hadoop, num_machines=num_machines, initial_solution=previous_solution)
                    else:
                        G = lptml.fit(x_pca_train, y_train, u, l, t, run_hadoop=run_hadoop, num_machines=num_machines, initial_solution=previous_solution, random_seed=rand_state)
                        previous_solution = np.dot(np.transpose(G), G)
                else:
                    G = np.identity(len(x_pca_train[1]))

                elapsed_time = time() - start_time
                print("elapsed time to get G", elapsed_time)
                # x_lptml = np.matmul(G, x.T).T
                print("what I got back was of type", type(G))
                # x_lptml_train, x_lptml_test = x_lptml[train_index], x_lptml[test_index]
                try:
                    x_lptml_train = np.matmul(G, np.transpose(x_pca_train)).T
                    x_lptml_test = np.matmul(G, np.transpose(x_pca_test)).T
                except:
                    print("continue")
                    raise

                neigh_lptml = KNeighborsClassifier(n_neighbors=4, metric="euclidean")
                neigh_lptml.fit(x_lptml_train, np.ravel(y_train))

                neigh = KNeighborsClassifier(n_neighbors=4, metric="euclidean")
                neigh.fit(x_pca_train, np.ravel(y_train))

                y_prediction = neigh.predict(x_pca_test)
                y_lptml_prediction = neigh_lptml.predict(x_lptml_test)

                iteration_results[0] = accuracy_score(y_test, y_prediction)
                iteration_results[1] = accuracy_score(y_test, y_lptml_prediction)

                iteration_results[4] = precision_score(y_test, y_prediction, average="macro")
                iteration_results[5] = precision_score(y_test, y_lptml_prediction, average="macro")

                iteration_results[8] = recall_score(y_test, y_prediction, average="macro")
                iteration_results[9] = recall_score(y_test, y_lptml_prediction, average="macro")

                iteration_results[12] = f1_score(y_test, y_prediction, average="macro")
                iteration_results[13] = f1_score(y_test, y_lptml_prediction, average="macro")

                iteration_results[16] = lptml.initial_violation_count
                iteration_results[17] = lptml.max_best_solution_d + lptml.max_best_solution_s #violated constraints

                d_viol, s_viol = lptml.count_violated_constraints(x_pca_test, y_test, lptml.transformer(np.identity(dimensions)), u, l)
                iteration_results[18] = d_viol + s_viol

                d_viol, s_viol = lptml.count_violated_constraints(x_pca_test, y_test, G, u, l)
                iteration_results[19] = d_viol + s_viol

                iteration_results[20] = elapsed_time

                print(iteration_results)
                experiment_results[str(reduce_dim_by)][str(target_iteration)].append(iteration_results)

    return experiment_results

def perform_experiment(x, y, number_of_folds, feat_count, PCA_dim_by, repeat_experiment, result_header, filename, lptml_iterations, S, D, ut, lt, run_hadoop=False, num_machines=10, label_noise=0, rand_state=-1):

    results_dict = split_pca_learn_metric(x, y, PCA_dim_by, repeat_experiment, number_of_folds, lptml_iterations, S, D, ut, lt, run_hadoop=run_hadoop, num_machines=num_machines, label_noise=label_noise, rand_state=rand_state)

    for pca in PCA_dim_by:
        for ite in lptml_iterations:
            final_results = ["", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            results = np.array(results_dict[str(pca)][str(ite)])

            if pca == 0:
                final_results[0] = result_header + " NOPCA"
            else:
                final_results[0] = result_header + " to " + str(feat_count - pca)

            final_results[0] += " t=" + str(ite)

            # Averages accuracy for Euclidean, lptml, LMNN, ITML
            final_results[1] = np.round(np.average(results[:, 0]), 2)
            final_results[2] = np.round(np.average(results[:, 1]), 2)

            # Std accuracy for Euclidean, lptml, LMNN, ITML
            final_results[3] = np.round(np.std(results[:, 0]), 2)
            final_results[4] = np.round(np.std(results[:, 1]), 2)

            # Averages precision for Euclidean, lptml, LMNN, ITML
            final_results[5] = np.round(np.average(results[:, 4]), 2)
            final_results[6] = np.round(np.average(results[:, 5]), 2)

            # Std precision for Euclidean, lptml, LMNN, ITML
            final_results[7] = np.round(np.std(results[:, 4]), 2)
            final_results[8] = np.round(np.std(results[:, 5]), 2)

            # Averages recall for Euclidean, lptml, LMNN, ITML
            final_results[9] = np.round(np.average(results[:, 8]), 2)
            final_results[10] = np.round(np.average(results[:, 9]), 2)

            # Std recall for Euclidean, lptml, LMNN, ITML
            final_results[11] = np.round(np.std(results[:, 8]), 2)
            final_results[12] = np.round(np.std(results[:, 9]), 2)

            # Averages F1 score for Euclidean, lptml, LMNN, ITML
            final_results[13] = np.round(np.average(results[:, 12]), 2)
            final_results[14] = np.round(np.average(results[:, 13]), 2)

            # Std F1 score for Euclidean, lptml, LMNN, ITML
            final_results[15] = np.round(np.std(results[:, 12]), 2)
            final_results[16] = np.round(np.std(results[:, 13]), 2)

            # Train initial  # violated
            final_results[17] = np.round(np.average(results[:, 16]), 2)
            final_results[18] = np.round(np.std(results[:, 16]), 2)

            # Train final  # violated
            final_results[19] = np.round(np.average(results[:, 17]), 2)
            final_results[20] = np.round(np.std(results[:, 17]), 2)

            # Test initial  # violated
            final_results[21] = np.round(np.average(results[:, 18]), 2)
            final_results[22] = np.round(np.std(results[:, 18]), 2)

            # Test final  # violated
            final_results[23] = np.round(np.average(results[:, 19]), 2)
            final_results[24] = np.round(np.std(results[:, 19]), 2)

            # Training time
            final_results[25] = np.round(np.average(results[:, 20]), 2)
            final_results[26] = np.round(np.std(results[:, 20]), 2)

            with open(filename, 'a', newline='') as resultsfile:
                wr = csv.writer(resultsfile, quoting=csv.QUOTE_ALL)
                wr.writerow(final_results)

if __name__ == "__main__":
    loaded_datasets = []
    # BEGIN EXPERIMENT LIST
    # Parameters for all experiments
    run_mlwga = True
    filename = 'demo-results.csv'
    train_size = 0.5

    # Breast cancer dataset
    bc = datasets.load_breast_cancer()
    x_bc = bc.data
    y_bc = bc.target
    loaded_datasets.append((x_bc, y_bc, "breast_cancer"))
    # Vehicle dataset
    # MISSING FOR NOW

    # German Credit dataset
    x_gc, y_gc = read_german_credit("./datasets/german_credit/german_credit.tsv") #pd.read_csv("./datasets/german_credit/german_credit.tsv", sep="\t")
    loaded_datasets.append((x_gc, y_gc, "german_credit"))

    # Image segment dataset
    x_is, y_is = read_image_segment("./datasets/image_segment/segmentation.data") #pd.read_csv("./datasets/german_credit/german_credit.tsv", sep="\t")
    loaded_datasets.append((x_is, y_is, "image_segment"))

    # Isolet dataset
    x_isolet, y_isolet = read_isolet("./datasets/isolet/isolet_csv.csv")
    loaded_datasets.append((x_isolet, y_isolet, "isolet"))

    # Letters dataset
    x_letters, y_letters = read_letters("./datasets/letters/letters.csv")
    loaded_datasets.append((x_letters, y_letters, "letters"))

    # MNIST dataset
    x_mnist, y_mnist = read_mnist("./datasets/mnist/t10k-images-idx3-ubyte", "./datasets/mnist/t10k-labels-idx1-ubyte")
    loaded_datasets.append((x_mnist, y_mnist, "mnist"))

    # Results presented in Figure 1
    # Average time as dimensionality increases

    PCA_dim_by = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    lptml_iterations = [100, 500, 1000, 1500, 2000]
    repeat_experiment = 10

    for x, y, dataset_name in loaded_datasets:
        print(f"Running test for -> {dataset_name}")
        for noise_fraction in [0]:
            random_seed = np.random.random_integers(1000)

            result_header = str(noise_fraction) + " noise WINE"
            feat_count = len(pd.Series(y).unique())
            perform_experiment(x, y, train_size, feat_count, PCA_dim_by, repeat_experiment, result_header,
                               filename, lptml_iterations, [],
                               [], 0, 0, label_noise=noise_fraction, rand_state=random_seed)

        # Results presented in Figure 2
        # Fraction of constraints violated and Accuracy as the number of iterations increases
        PCA_dim_by = [9, 5, 1]

        lptml_iterations = [10, 20, 30, 40, 50]
        repeat_experiment = 50

        for noise_fraction in [0]:
            random_seed = np.random.random_integers(1000)

            result_header = str(noise_fraction) + f" noise {dataset_name}"
            feat_count = len(pd.Series(y).unique())
            perform_experiment(x, y, train_size, feat_count, PCA_dim_by, repeat_experiment, result_header,
                               filename, lptml_iterations, [],
                               [], 0, 0, label_noise=noise_fraction, rand_state=random_seed)

        # Figure 3
        # Average accuracy as the fraction of label perturbation increases

        PCA_dim_by = [0]
        lptml_iterations = [2000]
        repeat_experiment = 10

        for noise_fraction in [0, 0.1, 0.2, 0.3]:
            random_seed = np.random.random_integers(1000)

            result_header = str(noise_fraction) + f"% noise {dataset_name}"
            feat_count = len(pd.Series(y).unique())
            perform_experiment(x, y, train_size, feat_count, PCA_dim_by, repeat_experiment, result_header,
                               filename, lptml_iterations, [],
                               [], 0, 0, label_noise=noise_fraction, rand_state=random_seed)
