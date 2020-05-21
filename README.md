# LP-type Metric Learning (LPTML)

An implementation of an LP-type algorithm for metric learning introduced in "[Learning Mahalanobis Metric Spaces via Geometric Approximation Algorithms](https://arxiv.org/pdf/1905.09989.pdf)". 

## Table of Contents

* [Getting Started](#getting-started)
* [Usage](#usage)
  * [Learning a metric from labels](#learning-a-metric-from-labels)
  * [Learning a metric from similarity and disimilarity constraints](#learning-a-metric-from-similarity-and-disimilarity-constraints)  
  * [Parallel version](parallel-version)
* [Authors](#authors)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Getting started

This code requires Python 3.6+

* Dependencies
  * Numpy
  * [CVXPY 1.0](https://www.cvxpy.org/) 
* For the parallel version only:
  * [mrjob](https://github.com/Yelp/mrjob)
  * Due to some compilation issues numpy=1.15.4 and cvxpy=1.0.14 have been set as required for installation in the Hadoop runner machines. These issues may be solved in future releases of these libraries.

#### SDP solver
CVXPY supports several solvers but we are restricted to choosing one that supports SDP. The code has been tested with [SCS](http://github.com/cvxgrp/scs) (open source). and MOSEK, which is a commercial product but provides free Academic Licences. To use MOSEK, it is sufficient to install the python library.

## Usage
### Learning a metric from labels
The following example learns a transformation matrix from points in the Iris dataset such that most points that belong to the same class are at distance at most *u* and points that belong to different classes are at distance at least *l*.

```python
import numpy as np
import lptml
from sklearn.datasets import load_iris

iris_data = load_iris()
X = iris_data['data']
Y = iris_data['target']

# upper threshold
u = 1
# lower threshold
l = 5
# t is the number of iterations
t = 1000

G = lptml.fit(x_train, y_train, u, l, t)
```
### Learning a metric from similarity and disimilarity constraints

The sets of similar and disimilar constraints can be passed to the algorithm.
```python
G = lptml.fit(x_train, y_train, u, l, t, sim, dis)
```

### Parallel version

A parallelized \LPTML was implemented using the [mrjob](https://github.com/Yelp/mrjob) package. As provided, this implementation uses AWS instances of type \emph{m4.xlarge}, AMI 5.20.0 and configured with Hadoop. OS and Python library dependencies are installed during the bootstrap stage. Because a new cluster of servers is provisioned for each experiment there is some time overhead before learning begins. Booting up the servers and installing all necessary dependencies requires around 16 minutes. This overhead can be avoided by changing some configurations in order to reuse an already running cluster (more details on the mrjob documentation).

The MapReduce job consists of one map and one reduce function. Each line of the input consists of a pair of points and a value that indicates if the points are similar or dissimilar.The input data is divided into a training and a validation subset. The map function divides the training into many pieces. The reduce function applies LPTML to learn a transformation matrix and return a count of violated constraints in the validation subset. The best result in terms of number of violations is selected as the final result.

To run the provided code, the following information has to be provided:

```
aws_access_key_id:
aws_secret_access_key:
```

The following are optional, only needed if you want mrjob to fetch log files via ssh (allegedly sooner):
```
ec2_key_pair:
ec2_key_pair_file:
ssh_tunnel: true
```

When running the code there is the option of reusing an already configured cluster. If the cluster is created on demand, the necessary libraries and dependencies will be installed first. This is configured with the bootstrap option. The first line is used to provide MOSEK with a valid license. The last one installs the library. Both lines can be removed if SCS is to be used.
```
    bootstrap:
      - echo -e "[replace with contents of mosek.lic]" >/tmp/mosek.lic
      - sudo yum install -y cmake
      - sudo yum install -y lapack-devel blas-devel
      - sudo pip-3.6 install numpy==1.15.4
      - sudo pip-3.6 install cvxpy==1.0.14
      - sudo pip-3.6 install -f https://download.mosek.com/stable/wheel/index.html Mosek
```

The number of pieces in which to map the input is controlled by lptmlCopies. Each piece contains lptmlFraction of all constraints. The number of machines that will run the tasks is controlled by num_core_instances. These parameters are overwritten when the mrjob task is launched in the mp_learn_metric() function in lptml.py.

```
    cmdenv:
      MOSEKLM_LICENSE_FILE: /tmp/mosek.lic
      lptmlUpperb: '2.82'
      lptmlLowerb: '7.26'
      lptmlCopies: '20'
      lptmlFraction: '1'
    num_core_instances: 2
    jobconf:
      mapreduce.task.timeout: 3600000
```

## Authors
* [Diego Ihara](https://dihara2.people.uic.edu/)
* Neshat Mohammadi
* [Francesco Sgherzi](https://fsgher2.people.uic.edu/)
* [Anastasios Sidiropoulos](http://sidiropoulos.org)

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments
This work was supported by the National Science Foundation under award CAREER 1453472, and grant CCF 1815145.
