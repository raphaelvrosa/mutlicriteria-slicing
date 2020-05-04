import numpy as np


class AHP:
    def __init__(self):
        self.weights = None
        self.data = None
        self.raw_data = []
        self.costs = []
        self.preferences = []

    def set(self, data, preferences, costs):
        self.raw_data = data
        self.costs = costs
        self.preferences = preferences

    def norm_col(self, a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1.0
        return a / np.expand_dims(l2, axis)

    def lin_col(self, data, cost=True):
        min_max = [10., 1.] if cost else [1., 10.]
        r = [np.min(data), np.max(data)]
        m = (min_max[1] - min_max[0]) / (r[1] - r[0])
        b = min_max[0] - m*r[0]

        norm_data = np.dot(data,m) + b
        # print 'norm_data', norm_data
        norm_col = self.norm_col(norm_data, axis=-1, order=1)
        # print 'norm_col', norm_col.T[:,0]
        return norm_col.T[:,0]

    def norm(self, a, costs):
        # a = np.array(data)
        cols = []
        i = 0
        for c in a.T:
            cols.append( self.lin_col(c, cost=costs[i]) )
            i += 1

        norm_cols = np.array(cols)
        return norm_cols.T

    def weight(self, A):
        # A = np.array(preferences)

        "Define vector of weights based on eigenvector and eigenvalues"
        eigenvalues, eigenvector = np.linalg.eig(A)
        maxindex = np.argmax(eigenvalues)
        # eigenvalues = np.float32(eigenvalues)  # float32
        eigenvector = np.float32(eigenvector.real)  # float32
        weights = eigenvector[:, maxindex]

        self.weights = np.array([w / np.sum(weights) for w in weights])

    def ahp(self, data=None, preferences=None, costs=None):
        if preferences is not None:
            self.preferences = preferences
        if costs is not None:
            self.costs = costs
        if data is not None:
            self.raw_data = data

        self.weight(self.preferences)
        self.data = self.norm(self.raw_data, self.costs)
        # print self.weights
        # print self.data
        rank = np.dot(self.data, self.weights)
        return rank


if __name__ == "__main__":
    data = [[110,448,12.6,25100], [104, 896, 13.2,26360], [0,100, 3.1, 39530]]
    prefs = [[1,1,5,5], [1,1,5,5], [1./5,1./5,1,3], [1./5, 1./5, 1./3, 1]]
    d = np.array(data)
    p = np.array(prefs)

    # print p
    ahp = AHP()

    costs = [True, True, False, True]
    # print ahp.ahp(data=d, preferences=p, costs=costs)
