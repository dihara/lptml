# LP-type Metric Learning (LPTML)

An implementation of the LP-type algorithm for metric learning introduced in the paper "[Learning Mahalanobis Metric Spaces via Geometric Approximation Algorithms](http://arxiv.org/)". 

## Table of Contents

* [Getting Started](#getting-started)
  * [Dependencies](#dependencies)
  * [SDP solver](#sdp-solver)
* [Usage](#usage)
  * [Learning a Metric](#learning-a-metric)
    * [Code example](#code-example)
    * [Parameters](#parameters)
  * [Parallel version](parallel-version)
* [Authors](#authors)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Getting started

### Dependencies
* Python 3.6+
* Numpy, [CVXPY](https://www.cvxpy.org/)
#### For the parallel version only:
* [mrjob](https://github.com/Yelp/mrjob)

### SDP solver
CVXPY supports several solvers but we are restricted to choosing one that supports SDP. The code has been tested with [SCS](http://github.com/cvxgrp/scs) (open source). We have also used MOSEK, which is a commercial product but provides free Academic Licences. To use MOSEK, it is sufficient to install the python library.

## Usage
### Learning a metric
#### Code example:
A simple example that learns a transformation matrix from points in the Iris dataset s.t. points that belong to the same class are at distance at most *u* and points that belong to different classes are at distance at least *l*.
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

#### Parameters

### Parallel version

The MapReduce job consists of one map and one reduce function. Each line of the input consists of a pair of points and a value that indicates if the points are similar or dissimilar.The input data $X = S \cup D$ is divided into $X_{train}$ and $X_{validate}$ sub sets. 

First, $X_{train}$ is partitioned into pieces. Due to the potentially large size of the input, the map function allows some flexibility in the way the data is partitioned according to the values of some parameters $p$ and $k$. Specifically, $p$ corresponds to the expected fraction of constraints in $X_{train}$ that is assigned to any of $k$ partitions. If $p=1$ then this corresponds to making $k$ copies of the input.

All pieces are processed by the reduce function in parallel using any of the available machines. The reduce function applies \LPTML to the current piece of input in order to learn a matrix $G$. All reduce functions have access to the same $X_{validate}$, which is used to evaluate $G$. The evaluation consists of counting the number of violated constraints in $X_{validate}$ when applying $G$. Finally, the reducer function returns the number of violated constraints and $G$ as a (key, value) pair and the calling program simply keeps the transformation matrix with the lowest number of violations.

The parallel version of \LPTML was implemented using the mrjob \cite{mrjob-docs} package. Amazon provides several types of machines with different processor and memory configurations. For these experiments, we used instances of type \emph{m4.xlarge}, AMI 5.20.0 and configured with Hadoop. OS and Python library dependencies are installed during bootstrap. 

Because we provision a new cluster of servers for each experiment there is some time overhead before learning begins. Booting up the servers and installing all necessary dependencies requires on average 16 minutes. This overhead can be avoided by changing some configurations in order to reuse an already running cluster.

For our experiments, we used [Amazon EMR](https://aws.amazon.com/emr/) but there is no reason why this shouldn't work in other environments. In general, it should run in any environment supported by mrjob with some modifications (see mrjob's documentation for more details).

To run the code, the following information has to be provided:

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

When running the code there is the option of reusing an already configured cluster. If the cluster is created on demand, first we must set it up with the necessary libraries and dependencies. This is done with the bootstrap option:
```
    bootstrap:
      - echo -e "[replace with contents of mosek.lic]" >/tmp/mosek.lic
      - sudo yum install -y cmake
      - sudo yum install -y lapack-devel blas-devel
      - sudo pip-3.6 install numpy==1.15.4
      - sudo pip-3.6 install cvxpy==1.0.14
      - sudo pip-3.6 install -f https://download.mosek.com/stable/wheel/index.html Mosek
```
The first line is used to provide MOSEK with a valid license. The last one installs the library. Both lines can be removed if SCS is to be used.

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
  local:
    upload_files:
      - ./mlwga.py
      - ./validate.data
      - ./mosek.lic
      - ./train.data
    cmdenv:
      UPPERB: '2.82'
      LOWERB: '7.38'
      LPTML_COPIES: '10.0'
      LPTML_FRACTION: '0.1'
```

## Authors
* [Diego Ihara](https://dihara2.people.uic.edu/)
* [Neshat Mohammadi]()
* [Anastasios Sidiropoulos](http://sidiropoulos.org)

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments
Supported by NSF under award CAREER 1453472, and grants CCF 1815145 and CCF1423230.
