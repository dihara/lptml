from mrjob.job import MRJob
from mrjob.step import MRStep
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from lptml import *

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class MRLPTML(MRJob):

    def steps(self):
        return [MRStep(mapper=self.mapper_partition_data, reducer=self.reducer_learn_metric)]

    def mapper_partition_data(self, _, line):
        a = round(float(os.environ['lptmlCopies']))
        f = float(os.environ['lptmlFraction'])

        number_pieces = round(1/f)
        pad = 10^len(str(number_pieces))
        for i in range(a):
            bucket = random.randrange(0, number_pieces)
            index = (pad * i) + bucket
            yield index, line

    def reducer_learn_metric(self, key, constraints):
        u = round(float(os.environ['lptmlUpperb']), 1)
        l = round(float(os.environ['lptmlLowerb']), 1)

        S = []
        D = []
        print("key is", key)
        for c in constraints:
            constraint = json.loads(c)
            if constraint[2] == 'S':
                S.append(constraint[:-1])
            else:
                D.append(constraint[:-1])

        # run only one iteration per reducer
        k = 100
        blockPrint()
        A = learn_metric(S, D, u, l, k)
        enablePrint()

        s_viol, d_viol = count_violated_constraints_file('validate.data', A, u, l)

        yield s_viol + d_viol, json.dumps(A.tolist())

if __name__ == '__main__':
    MRLPTML.run()