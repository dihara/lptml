"""LP-type metric learning

Requires CVXPY and numpy to be installed.

This file can be imported as a module and contains the following
functions to learn a metric:

    * fit() - returns the matrix for the Mahalanobis distance
    * transformer() -  returns the cholesky decomposition of a matrix

Change the value of limit_constraints in order to restrict the number
of constraints to be considered when counting violations.

This work was supported by the National Science Foundation under award CAREER 1453472, and grant CCF 1815145.
"""

import numpy as np
from cvxpy import *
import mosek
import math
import random
import itertools
import time
import json
from mr_lptml import MRLPTML

# this serves to limit the number of constraints to check when counting violations
limit_constraints = 2000

# Global variables
max_best_solution_s = 0
max_best_solution_d = 0
initial_violation_count = 0

rand_vector_r = []
identity_d = []


def learn_metric(S, D, u, l, iterations, fixed_p=None, initial_solution=[]):
    """Learn a Mahalanobis metric

    Parameters
    ----------
    S : list
        List of pairs of points that are similar i.e. have the same label
    D : list
        List of pairs of points that are disimilar
    u : float
        upper threshold for distance between similar points
    l : float
        lower threshold for distance between disimilar points
    iterations : int
        Number of times the input is sampled and a metric is learned
    fixed_p : float, optional
        Sample constraints with some fixed probability
    initial_solution : list, optional
        Initialize answer to this value. This serves to 'continue' the search
        for a better solution from this point.

    Returns
    -------
    list
        Mahalanobis distance matrix
    """

    global identity_d
    global max_best_solution_s
    global max_best_solution_d
    global initial_violation_count

    d = get_dimension(D, S)

    n = len(S) + len(D)

    identity_d = np.identity(d)
    if initial_solution == []:
        best_A = identity_d
    else:
        # print("using initial solution:", initial_solution)
        best_A = initial_solution

    viol_d, viol_s = count_violated_constraints_SD(S, D, transformer(best_A), u, l)
    print("##Initial violated constraints: d ", viol_d, "s", viol_s)
    print("t =", iterations)
    max_best_solution_d = viol_d
    max_best_solution_s = viol_s
    initial_violation_count = max_best_solution_d + max_best_solution_s #saving this value for reference

    for t in range(iterations):
        # 3 is chosen so we do not end up with a sample that is too small
        if fixed_p is None:
            p = 2 ** (-1 * np.random.randint(3, math.log2(n)))
        else:
            p = fixed_p

        M = math.floor(np.log2(n))

        min_constraints = d**2
        R, r_count = subsample(S, D, p, min_constraints)

        #calculate initial basis
        B0 = calculate_initial_basis(R)

        #solve the problem as LPType
        choose_vector_r_randomly(d)

        R = initial_sort(R, best_A, u, l) #not sure

        Bn, A = pivot_LPType(B0, R, u, l, d, 0, False, [])

        if A==[]:
            continue

        #Count the number of constraints of L that the solution violates
        violated_constraints_d, violated_constraints_s = count_violated_constraints_SD(S, D, transformer(A), u, l)

        if (max_best_solution_s + max_best_solution_d) > (violated_constraints_d + violated_constraints_s):
            best_A = A
            max_best_solution_d = violated_constraints_d
            max_best_solution_s = violated_constraints_s

        #if no constraints violated return immediately
        if ((violated_constraints_d + violated_constraints_s) == 0):
            break

    return best_A


def transformer(A):
    """Return the cholesky decomposition of A to be used to perform a
    linear transformation

    Parameters
    ----------
    A : np.array
        Mahalanobis distance matrix

    Returns
    -------
    np.array
        The cholesky decomposition of the provided matrix
    """

    global identity_d
    try:
        G = np.linalg.cholesky(A).T
    except:
        return identity_d

    return G


def pivot_LPType(B, C, u, l, d, last_cost, use_last_cost, basis_A):
    """LP-Type metric learning implementing the move-to-front and pivot heuristics

    """

    calculate_basis_cost = not use_last_cost
    current_basis_cost = last_cost

    C_perm = get_permutation(B, C)

    i = 0

    A = np.identity(d)
    if basis_A != []:
        A = basis_A

    Buc = B.copy()

    while True:
        c, e, t = maximal_violation(C_perm, i, A, u, l)
        if e <= 0:
            break

        C_perm.insert(0, C_perm.pop(t))

        i = i + 1

        Bh = B.copy()
        Bh.append(c)

        Buc.append(c)

        #Calculate the cost of the new candidate basis and the cost of the current basis
        try:
            ABh = semidefsolver(Bh, u, l)
            if ABh == []:
                return [], []
        except:
            print("Solver error")
            raise

        Bh_cost = w(ABh)

        if calculate_basis_cost:
            A = semidefsolver(B, u, l)
            if A == []:
                return [], []

            B_cost = w(A)
        else:
            B_cost = current_basis_cost

        if Bh_cost > B_cost:
            T, z = compBasis(Bh, Bh_cost, u, l, d)

            B, A = pivot_LPType(T, Buc, u, l, d, Bh_cost, True, ABh)
            if A == []:
                return [], []
                break

            calculate_basis_cost = False
            current_basis_cost = w(A)
        else:
            calculate_basis_cost = False
            current_basis_cost = B_cost

    return B, A


def compBasis(B, cost, u, l, d):
    """compute the basis from a set of constraints
    """

    # Try to remove one constraint at a time
    if len(B) > d**2:
        for t in itertools.combinations(B, len(B) - 1):
            A = semidefsolver(t, u, l)
            cost_1 = w(A)
            if cost_1 >= cost:
                # If I can remove one constraint, try removing another one
                t2, w2 = compBasis(list(t), cost_1, u, l, d)
                A2 = semidefsolver(t, u, l)
                cost_2 = w(A2)
                # Return Basis
                if cost_2 >= cost_1:
                    return(t2, cost_2)
                else:
                    return(list(t), cost_1)

    return B, cost


def get_permutation(B0, C):
    C_prime = []

    # C has been randomly permuted
    for c in C:
        exists_in_both = False
        for b in B0:
            if np.array_equal(c[0], b[0]) and np.array_equal(c[1], b[1]) and (c[2] == b[2]):
                exists_in_both = True

        if not exists_in_both:
            C_prime.append(c)

    return C_prime


def get_mahalanobis_distance(a, b, G):
    """Compute the distance between points a and b
    """

    point_i = np.matrix(a)
    point_j = np.matrix(b)

    point_i = np.transpose(np.matmul(G, np.transpose(point_i)))
    point_j = np.transpose(np.matmul(G, np.transpose(point_j)))

    d = np.linalg.norm(point_i - point_j)

    return d


def count_violated_constraints(x, y, G, u, l):
    """Count the number of violated constraints
    """

    if G == []:
        return math.inf, math.inf

    s_count = 0
    d_count = 0

    if len(x) < limit_constraints:
        all_pairs = []
        for pair_of_indexes in itertools.combinations(range(0, len(x)), 2):
            all_pairs.append(pair_of_indexes)

        xg = np.matmul(G, np.transpose(x)).T

        for i in range(len(all_pairs)):
            point_i = np.matrix(xg[all_pairs[i][0]])
            point_j = np.matrix(xg[all_pairs[i][1]])

            d = np.linalg.norm(np.subtract(point_i, point_j))
            if y[all_pairs[i][0]] != y[all_pairs[i][1]]:
                if d < l:
                    d_count = d_count + 1
            else:
                if d > u:
                    s_count = s_count + 1
    else:
        for i in range(limit_constraints):
            i = np.random.randint(0, len(x))
            j = np.random.randint(0, len(x))
            point_i = np.transpose(np.matmul(G, np.transpose(np.matrix(x[i]))))
            point_j = np.transpose(np.matmul(G, np.transpose(np.matrix(x[j]))))

            distance = np.linalg.norm(point_i - point_j)

            if y[i] != y[j]:
                if distance < l:
                    d_count = d_count + 1
            else:
                if distance > u:
                    s_count = s_count + 1

    return d_count, s_count


def count_violated_constraints_file(file, G, u, l):
    """when running the parallel version, the constraints to
       check are in a file
    """

    if G == []:
        return math.inf

    s_count = 0
    d_count = 0

    with open(file, 'r') as f:
        for line in f:
            try:
                constraint = json.loads(line)
            except:
                continue

            distance = get_mahalanobis_distance(constraint[0], constraint[1], G)
            if constraint[2] == 'S':
                if distance > u:
                    s_count = s_count + 1
            else:
                if distance < l:
                    d_count = d_count + 1

    return d_count, s_count


def count_violated_constraints_SD(S, D, G, u, l):
    """count violated constraints when given the sets
       S and D as input
    """

    if G == []:
        return math.inf

    s_count = 0
    d_count = 0

    for constraint in S:
        distance = get_mahalanobis_distance(constraint[0], constraint[1], G)

        if distance > u:
            s_count = s_count + 1

    for constraint in D:
        distance = get_mahalanobis_distance(constraint[0], constraint[1], G)

        if distance < l:
            d_count = d_count + 1

    return d_count, s_count


def semidefsolver(H, u, l):
    """Solve a small semi-definite program.
       Other SDP solvers can be used. See the cvxpy documentation.
    """

    if len(H) == 0:
        return []

    d = len(H[0][0])

    #A = Semidef(d)
    #changed after upgrading to cvxpy 1.0
    A = Variable(shape=(d,d), PSD=True)

    #build set of constraints
    constraints = []
    obj = Minimize(0)

    try:
        for h in H:
            point_i = np.array(h[0])
            point_j = np.array(h[1])

            if h[2] == 'S':
                constraints.append(((point_i - point_j).T * A * (point_i - point_j)) <= (u ** 2))
            else:
                constraints.append(((point_i - point_j).T * A * (point_i - point_j)) >= (l ** 2))
    except:
        raise

    problem = Problem(obj, constraints)
    try:
        # There are other solvers available. see the
        # problem.solve(solver=MOSEK, mosek_params={mosek.iparam.num_threads: 1})
        # problem.solve(solver=MOSEK, verbose=True)
        # problem.solve(solver=SCS, use_indirect=True)
        # problem.solve(solver=SCS)
        problem.solve(solver=MOSEK, mosek_params={mosek.iparam.num_threads: 8})
    except Exception as e:
        return []

    if problem.status != "optimal":
        if problem.status != "optimal_inaccurate":
            return []

    return A.value


def w(A):
    """cost function of the LP-type problem
    """

    if A is None:
        return float("-inf")
    if len(A) == 0:
        return float("-inf")

    d = len(A)
    ei = get_vector_r(d)

    try:
        W = np.transpose(ei) * A * ei
    except:
        i = 0

    w = np.sum(W)

    return w

def choose_vector_r_randomly(d):
    global rand_vector_r

    base = np.random.randn(d, 1)

    base /= np.linalg.norm(base, axis=0)

    rand_vector_r = base

    return None


def get_vector_r(d):
    global rand_vector_r

    base = rand_vector_r

    return base


def preprocess(D, S):
    H = []
    n = 0

    for s in S:
        s.append('S')
        H.append(s)
        n = n + 1

    for d in D:
        d.append('D')
        H.append(d)
        n = n + 1

    return H, n


def get_dimension(D, S):
    if len(D) > 0:
        d = len(D[0][0])
    else:
        d = len(S[0][0])

    return d


def subsample(S, D, p, min_sample):
    """subsample from S and D
    """
    H_prime = []
    s_count = 0
    d_count = 0

    while s_count == 0:
        for c in S:
            if random.random() <= p:
                c = np.array(c).tolist()
                c.append('S')
                H_prime.append(c)
                s_count += 1

    while d_count == 0:
        for c in D:
            if random.random() <= p:
                c = np.array(c).tolist()
                c.append('D')
                H_prime.append(c)
                d_count += 1

    return H_prime, s_count + d_count


def calculate_initial_basis(H):
    """The initial basis consists of one arbitrary similarity constraint
    """

    B0 = []
    for h in H:
        if h[2] == 'S':
           B0.append(h)
           break

    return B0


def fit(x, y, u, l, t, S=[], D=[], run_hadoop=False, num_machines=2, initial_solution=[], random_seed=-1):
    """Learn a Mahalanobis metric

    Parameters
    ----------
    x : list
        List of points
    y : list
        Labels for each point
    u : float
        Upper threshold
    l : float
        Lower threshold
    t : int
        Number of iterations of lptml to perform
    S : list, optional
        List of pairs of points to be used as the set of similarity constraints
    D : list, optional
        List of pairs of points to be used as the set of disimilarity constraints
    run_hadoop : bool, optional
        Flag to indicate if mrjob is to be used to run lptml on a Hadoop
        cluster (default False)
    num_machines : int, optional
        Number of machines to request for the Hadoop cluster
    initial_solution : ndarray
        Start with some initial solution (for example: A matrix from a previous
        execution)
    random_seed : int
        Set random seed for permutations and sampling.

    Returns
    -------
    ndarray
        Distance matrix
    """

    # how many data points are there?
    n = len(x)
    # get the number of features from the first data point
    d = len(x[0])

    if random_seed > 0:
        np.random.seed(random_seed)

    similar_pairs_S = []
    dissimilar_pairs_D = []

    print(n, " # of points")
    if (len(S) + len(D)) > 0:
        for i in range(len(S)):
            similar_pairs_S.append([x[S[i][0]], x[S[i][1]]])

        for i in range(len(D)):
            dissimilar_pairs_D.append([x[D[i][0]], x[D[i][1]]])
    else:
        all_pairs = []
        for pair_of_indexes in itertools.combinations(range(0, len(x)), 2):
            all_pairs.append(pair_of_indexes)

        #get the number of features from the first data point
        d = len(x[0])

        randomized_indexes = range(0, len(all_pairs))
        #print("I have", len(randomized_indexes), "constraints")

        if run_hadoop:
            data = []
            #print("I have", len(randomized_indexes), "lines of data")
            for i in randomized_indexes:
                if y[all_pairs[i][0]] == y[all_pairs[i][1]]:
                    data.append([x[all_pairs[i][0]].tolist(), x[all_pairs[i][1]].tolist(), 'S'])
                else:
                    data.append([x[all_pairs[i][0]].tolist(), x[all_pairs[i][1]].tolist(), 'D'])
        else:
            for i in randomized_indexes:
                if y[all_pairs[i][0]] == y[all_pairs[i][1]]:
                    similar_pairs_S.append([x[all_pairs[i][0]], x[all_pairs[i][1]]])
                else:
                    dissimilar_pairs_D.append([x[all_pairs[i][0]], x[all_pairs[i][1]]])

                if (len(similar_pairs_S) > limit_constraints) and (len(dissimilar_pairs_D) > limit_constraints):
                    break

    print("number of constraints: d = ", len(dissimilar_pairs_D),  " s = ", len(similar_pairs_S))

    np.random.seed(None)
    if run_hadoop:
        A = mp_learn_metric(data, u, l, t, num_machines)
    else:
        A = learn_metric(similar_pairs_S, dissimilar_pairs_D, u, l, t, initial_solution=initial_solution);

    try:
        #Maybe I found no solution
        if A == []:
            return np.identity(d)

        G = transformer(A)
    except:
        #sometimes there is some error involving the solver
        G = np.identity(d)

    return G


def initial_sort(constraints, A, u, l):
    worst_index = 0
    worst_value = 0

    G = transformer(A)

    idx = 0
    for c in constraints:
        i = np.matmul(G, c[0])
        j = np.matmul(G, c[1])

        if c[2] == 'S':
            violation = np.linalg.norm(i - j) - u
        else:
            violation = l - np.linalg.norm(i - j)

        if violation > worst_value:
            worst_value = violation
            worst_index = idx

        idx = idx + 1

    constraints.insert(0, constraints.pop(worst_index))

    return constraints


def maximal_violation(constraints, skip, A, u, l):
    worst_index = 0
    worst_value = math.inf * -1

    if len(constraints)==0:
        return [], worst_value, worst_index

    try:
        G = transformer(A)
    except:
        G = identity_d

    idx = 0
    for c in constraints:
        i = np.matmul(G, c[0])
        j = np.matmul(G, c[1])

        if c[2] == 'S':
            violation = np.linalg.norm(i - j) - u
        else:
            violation = l - np.linalg.norm(i - j)

        if (idx >= skip) and (violation > worst_value):
            worst_value = violation
            worst_index = idx

        idx = idx + 1

    return constraints[worst_index], worst_value, worst_index


def mp_learn_metric(data, u, l, t, num_machines):
    """Learn the metric using mrjob


    Parameters
    ----------
    data : list
        List of triples where the first and second elements are points and the third
        is either "S" or "D" depending on whether the pair is a similar or disimilar
        constraint
    u: float
        Upper threshold
    l: float
        Lower threshold
    t: int
        Number of iterations
    num_machines: int
        Number of machines to launch for the cluster

    Returns
    -------
    ndarray
        Best distance matrix
    """

    part_c = 10 #number of copies of each partition
    part_f = 0.1 #fraction of training constraints included in each partition

    validate_fraction = 0.1 #fraction of data to use as validation set

    total_data_size = len(data)
    validate_size = round(total_data_size * validate_fraction)
    training_size = total_data_size - validate_size

    f_train = open('train.data', 'w')
    f_train.truncate(0)
    f_val = open('validate.data', 'w')
    f_val.truncate(0)
    for constraint in data:
        line = json.dumps(constraint)

        if training_size > 0:
            f_train.writelines("%s\n" % line)
            training_size -= 1

        if (training_size == 0) & (validate_size > 0):
            f_val.writelines("%s\n" % line)
            validate_size -= 1

        if (training_size + validate_size) == 0:
            break
    f_train.close()
    f_val.close()

    print("launching mrjob")
    best_A = []
    parameters = ['-c', 'mrjob.conf', '-r', 'emr', 'train.data',
                  '--cmdenv', 'lptmlUpperb=' + str(u), '--cmdenv', 'lptmlLowerb=' + str(l), '--cmdenv', 'lptmlCopies=' + str(part_c), '--cmdenv', 'lptmlFraction=' + str(part_f),
                  '--num-core-instances=' + str(num_machines)
                  ]

    mr_job = MRLPTML(args=parameters)
    print("args", parameters)
    with mr_job.make_runner() as runner:
        print("running")

        start_time = time.time()
        runner.run()
        print("finished after", time.time() - start_time, "seconds")

        print("Here be the counters:")
        print(runner.counters())

        best = math.inf
        try:
            z_count = 0
            for key, value in mr_job.parse_output(runner.cat_output()):
                # print("key is", key)
                # print("value is", value)

                if key < best:
                    z_count += 1
                    best = key
                    best_A = json.loads(value)
        except Exception as e:
            print("error", e, "but continue")

    print("best value is", best_A, "best in", z_count)

    return best_A
