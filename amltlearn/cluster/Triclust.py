from tqdm import tqdm
import numpy as np
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

"""
.. module:: Triclust

Triclust
*************

:Description: Effective clustering and boundary detection algorithm based on Delaunay triangulation
    Inspired in work from Dongquan Liu, Gleb V. Nosovskiy, Olga Sourina.

    Uses Delaunay traingulation for finding neighboring data and then applies basic statistics to separate
    between inner and boundary data in an incremental way.

:Authors: mora

:Version: 1.0

:Created on: 09/12/2016 18:29

"""

__author__ = 'mora'


class Triclust(object):

    """ Triclust algorithm """

    def __init__(self,
                 s_min=200,
                 s_max=10000,
                 rc_param=5000,
                 inner_ratio=1.0,
                 num_bins=15,
                 verbose=True):
        """ Initializes algorithm and sets parameters. Default values are recommended though they can be tuned
        for data-specific needs. The original paper shows little variance using different values anyway.
        Args:
            s_min: Number of data points at which human vision starts missing individual data points
                in the specific domain. Based on human vision studies. Defaults to 200.
            s_max: Number of data points at which global features become relevant. Defaults to 10000.
            rc_param: Minimum number of instances of a dataset to be considered big. Default is 5000
            inner_ratio: Fraction of the inner threshold value to use instead of the original one. Used for
                reducing the number of inner points and, therefore, helping reducing noise
            num_bins: Number of bins to use in the frequency histogram that determines rc. Default is 10
            verbose: whether to output process information
        """
        self.s_min = s_min
        self.s_max = s_max
        self.rc_param = rc_param
        self.thresh_ratio = inner_ratio
        self._num_bins = num_bins
        self.clusters = None
        self.complete_clustering = None
        self.clus_ptr = 0
        self.fcs = None
        self.stats = None
        self.inner_threshold = None
        self.verbose = verbose


    def fit(self, data):
        self.fit_predict(data)
        return self


    def fit_predict(self, data):
        """ Starts to iterate clustering the data """

        if self.verbose:
            print('Computing Delaunay and stats ...')

        # Step 1-2: Compute statistics of the data
        self.stats = StatsManager(data, self.s_min, self.s_max)
        self.stats.compute()

        # Step 3.1 : Compute function criteria for all data points
        self.fcs = np.array([self.stats.compute_fc(i) for i in range(data.shape[0])])

        # Step 3.2: Get rc threshold
        rc = self._get_rc(self.fcs, data.shape[0])

        # Step 4.1: Get data less than rc to discard outliers
        fcs_inds = np.where(self.fcs < rc)[0]
        fcs_subset = np.expand_dims(self.fcs[fcs_inds], axis=1)

        # Step 4.2: Use k-means to obtain boundary
        # 2 clusters (inner and boundary). Centers -> minima and mean
        centers = np.array([[np.min(fcs_subset)], [np.mean(fcs_subset)]])
        kmeans = KMeans(n_clusters=2, init=centers).fit(fcs_subset)

        # Step 4.3: Compute threshold as mid point between centers
        self.inner_threshold = np.mean(kmeans.cluster_centers_) * self.thresh_ratio

        # Step 4.4: Select inner data points using computed threshold
        inner_indexes = np.where(self.fcs < self.inner_threshold)[0]
        boundary_indexes = np.where(self.fcs >= self.inner_threshold)[0]

        # Step 5-6: Iterative cluster construction
        # Initialize clusters to -1 so boundary points are marked as -1
        self.clusters = np.zeros(data.shape[0]) - 1
        inner_set = tqdm(inner_indexes, desc="Clustering inner points ...") if self.verbose else inner_indexes
        for i in inner_set:
            self.cluster_step(i)

        # Step 7: Convert boundary points into clusters using majority of their neighbors
        self.complete_clustering = np.copy(self.clusters)
        bound_set = tqdm(boundary_indexes, desc="Clustering boundary points ...") if self.verbose else boundary_indexes
        for i in bound_set:
            self.complete_clustering[i] = self._cluster_majority(i)

        return self.complete_clustering


    def predict(self):
        raise NotImplementedError("Function not implemented for Triclust clustering")


    def _cluster_majority(self, i):
        """ Returns the most popular cluster in the neighborhood of the input point"""
        # Get clusters of neighbors
        neighs = self.clusters[self.stats.get_neighbors(i)]
        # Discard boundary points
        neighs = neighs[np.where(neighs != -1)[0]]
        if len(neighs) == 0:
            # Point is an outlier
            return -1
        else:
            # Get majority cluster
            unique, counts = np.unique(neighs, return_counts=True)
            return unique[np.argmax(counts)]


    def cluster_step(self, pi):
        """ Performs a single cluster step for input point """
        if self.clusters[pi] == -1:
            # Create new cluster and assign pi to it
            self._initialize_new_cluster(pi)
        else:
            # Cluster already computed
            return


    def _initialize_new_cluster(self, pi):
        """ Starts new cluster with one instance """
        self.clus_ptr += 1
        self.build_cluster([pi], self.clus_ptr)


    def build_cluster(self, current_inst, cluster_id):
        """ Recursively build cluster by adding those neighboring instances to the given
        ones that are inner points. If an instance is added, its neighborhood is searched
        for inner points as well. Already visited points are ignored """
        # Iterate through potential list
        for i in current_inst:

            # Check whether current instance is inner cluster point
            if self.clusters[i] != cluster_id and self._is_inner_cluster(i):
                # Add current instance
                self.clusters[i] = cluster_id
                # Get inner instances in the neighbors of the added instance
                self.build_cluster(self.stats.get_neighbors(i), cluster_id)


    def _is_inner_cluster(self, pi):
        """ Returns whether the given point is an inner cluster point """
        return self.fcs[pi] < self.inner_threshold


    def _assign_to_cluster(self, pi, clus):
        """ Assigns point to given cluster """
        self.clusters[pi] = clus


    def _get_rc(self, fcs, data_size):
        """ Returns the Rc value of the list of input feature criteria """
        # Compute histogram and its centers
        counts, cuts = np.histogram(fcs, bins=self._num_bins, density=False)

        # Compute R1
        zero_freq = np.where(counts == 0)[0]
        if len(zero_freq) == 0:
            r = np.inf
        else:
            centers = [cuts[i] + ((cuts[i + 1] - cuts[i]) / 2.0) for i in range(len(cuts) - 1)]
            r = centers[zero_freq[0]]

        # Compute R2 for large datasets
        if data_size > self.rc_param:
            r = min(r, np.percentile(fcs, 97))

        return r


class StatsManager(object):

    """ Wrapper for computing the Delaunay triangulation, edge length and statistics of the data """

    SEPARATOR = '-'

    def __init__(self, X, s_min, s_max):
        self.X = X
        self.num_points = X.shape[0]
        self.s_min = s_min
        self.s_max = s_max
        self.a, self.b, self.c = self._a(), self._b(), self._c()
        self.edges = {}
        self.mean = {}
        self.dm = {}
        self.pdm = {}

    def _a(self):
        """ Compute coefficient a """
        if self.num_points < self.s_min:
            return self.num_points / 2000.0
        elif self.s_min <= self.num_points <= self.s_max:
            num = 1.9 * self.num_points + (self.s_max / 10.0) - (2.0 * self.s_min)
            den = float(self.s_max) - self.s_min
            return num / den
        else:
            return 2.0

    def _b(self):
        """ Compute coefficient b """
        if self.num_points < self.s_min:
            return 1.0
        elif self.s_min <= self.num_points <= self.s_max:
            return 1.0 - ((self.num_points - self.s_min) / (2*(self.s_max - self.s_min)))
        else:
            return 0.5

    def _c(self):
        """ Compute coefficient c """
        if self.num_points < self.s_min:
            return 0.5
        elif self.s_min <= self.num_points <= self.s_max:
            num = 0.5 * self.num_points + 0.5 * self.s_max - self.s_min
            den = float(self.s_max) - self.s_min
            return num / den
        else:
            return 1.0

    def compute(self):
        """ Computes Delaunay info and edge length """
        self._compute_delaunay()
        self._compute_edge_length()
        self._compute_statistics()

    def _compute_delaunay(self):
        """ Computes and stores Delaunay tessalation """
        self.delaunay = Delaunay(self.X, qhull_options="QJ")
        for i in range(self.num_points):
            # Check whether there is any point without neighbors
            if len(self.get_neighbors(i)) == 0:
                raise ValueError('Found point "%d" without neighbors' % i)

    def get_neighbors(self, point):
        """ Returns the indices of the neighbors of input vertex """
        neighbors_inds, neighbors = self.delaunay.vertex_neighbor_vertices
        return neighbors[neighbors_inds[point]:neighbors_inds[point+1]]

    def _get_point(self, p):
        """ Returns point(vertex) with given index """
        return self.X[p]

    def _compute_edge_length(self):
        """ Computes the length of the edges in the graph. It has a complexity around O(N) """
        # Map edge length into dictionary so it can be accessed in O(1) later
        for i in range(self.num_points):
            neigh_indices = self.get_neighbors(i)
            for ni in neigh_indices:
                if not self.exists_edge(i, ni):
                    self._compute_length(i, ni)

    def _get_key(self, v1, v2):
        """ Returns the string formatted key for two vertices to be used in the dictionary """
        return self.SEPARATOR.join([str(v1), str(v2)])

    def exists_edge(self, v1, v2):
        """ Whether exists data for edge between input vertices """
        return self._get_key(v1, v2) in self.edges

    def _get_length(self, v1, v2):
        """ Returns length of the edge between input nodes"""
        return self.edges[self._get_key(v1, v2)]

    def get_edge_length(self, v1, v2):
        """ Returns length of edge between input nodes """
        if self.exists_edge(v1, v2):
            return self._get_length(v1, v2)
        elif self.exists_edge(v2, v1):
            return self._get_length(v2, v1)
        else:
            raise ValueError("Edge does not exist between %d and %d" % (v1, v2))

    def _compute_length(self, v1, v2):
        """" Computes the length between two vertices """
        self.edges[self._get_key(v1, v2)] = euclidean(self._get_point(v1),
                                                      self._get_point(v2))

    def _compute_statistics(self):
        """ Computes statistics for each point """
        self._compute_means()
        for i in range(self.num_points):
            self.dm[i] = self._dm(i)
            self.pdm[i] = self._pdm(i)

    def _compute_means(self):
        for i in range(self.num_points):
            self.mean[i] = self._mean(i)

    def _get_neighbors_length(self, v):
        """ Returns length of the edges in the neighborhood of the point (incident edges and
        edges between points in the neighborhood """
        points = self.get_neighbors(v)
        points = np.append(points, [v])
        edges = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if i != j and self.exists_edge(points[i], points[j]):
                    edges.append(self.get_edge_length(points[i], points[j]))
        return edges

    def _mean(self, v):
        """ Computes the mean of a point as the mean length in the point neighborhood """
        return np.mean([self.get_edge_length(v, vi) for vi in self.get_neighbors(v)])

    def _std(self, v):
        """ Standard deviation of edge lengths in the neighborhood """
        mean = self.mean[v]
        neighbors_length = self._get_neighbors_length(v)
        num_neighs = len(neighbors_length)
        return np.sqrt(np.sum(np.square(mean - neighbors_length)) / (num_neighs - 1))

    def _dm(self, v):
        """ Computes the DM as the quotient of the STD divided by the mean of the point """
        return self._std(v) / self.mean[v] if self.mean[v] != 0 else 0.0

    def _pdm(self, pi):
        """ Computes the PDM of a data point as the mean of the positive parts of the derivative
        of the mean along all edges connected to the point. In other words, it is the mean of the
        lengths of edges connecting the point and those points in the neighborhood which have smaller
        mean statistic """
        pdms = [self._pd(pi, pj) for pj in self.get_neighbors(pi) if self.mean[pi] > self.mean[pj]]
        return np.mean(pdms) if len(pdms) > 0 else 0.0

    def _pd(self, pi, pj):
        """ Returns the positive part of the derivative of mea along the edge
        connecting Pi and Pj """
        if self.mean[pi] <= self.mean[pj]:
            return 0
        else:
            if self.get_edge_length(pi, pj) > 0:
                return (self.mean[pi] - self.mean[pj]) / self.get_edge_length(pi, pj)
            else:
                return 0.0


    def compute_fc(self, pi):
        """ Computes the function criteria for the input point """
        return self.a * self.mean[pi] + self.b * self.dm[pi] + self.c * self.pdm[pi]
