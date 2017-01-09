from joblib import Parallel, delayed
import numpy as np
from scipy.spatial.distance import cdist, pdist

from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import calinski_harabaz_score

"""
.. module:: MembraneClustering

MembraneClustering
*************

:Description: Automatic clustering algorithm inspired in tissue-like membrane systems.
    Inspired in work from Hong Peng, Jun Wang, Peng Shi, Agustin Riscos-Nunez and Mario J.Perez-Jimenez.
    This clustering method is only intended for numerical data but can be extended to support heterogeneous
    data.

:Authors: bejar, mora

:Version: 1.0

:Created on: 09/12/2016 18:29

"""

__author__ = 'mora'


class Metric(object):

    CLUSTER_SEPARATION = "cs"
    CALINSKI = "calinski"


class ObjectStr(object):

    """ Represents an object, a clustering solution. Can be read from existing position, velocity
    and score (if provided) or can be initialized randomly (must provided feature ranges """

    def __init__(self, data, k_max, metric, pref_dist,
                 lower_b=None, upper_b=None, position=None, velocity=None, cs=None):
        self.k_max = k_max
        self.data = data
        self.metric = metric
        self.pref_dist = pref_dist

        init = False

        # Initialize using lower and upper bounds on the centers
        if lower_b is not None and upper_b is not None:
            init = True
            self._initialize_object(upper_b, lower_b)

        # Initialize using initial vector, velocity and cluster separation metric
        if position is not None and velocity is not None and cs is not None:
            if init:
                raise ValueError('Cannot provide both feature ranges or initial context. '
                    + 'Please choose either of them')
            else:
                init = True
                self.position = position
                self.velocity = velocity
                self.cs = cs

        # Check it has been initialize in any way
        if not init:
            raise ValueError("Could not initialize Object. Provide either the initial feature " +
                             " ranges or the initial context of the object (position, velocity and cs)")


    def _initialize_object(self, upper, lower):
        """ Creates random object"""
        # Empty array
        dims = len(upper)
        position = np.zeros((self.k_max + self.k_max * dims))

        # Need to set seed for different initializations due to Parallel processing
        np.random.seed(None)

        # First z positions are initialized as real numbers between
        position[0:self.k_max] = np.random.rand(self.k_max)

        # Update random coordinates for each column in each cluster
        for i in range(self.k_max):
            for j in range(dims):
                random = self._to_range(np.random.rand(), 0, 1, lower[j], upper[j])
                index = self.k_max + (i * dims) + j  # jth position of cluster i
                position[index] = random
        self.position = position

        # Compute random velocity
        self.velocity = np.random.rand(len(self.position))

        # Compute cluster separation
        self.update_cs()


    def valid_clusters(self):
        """ Returns the clusters which have enough support """
        return [i for i in range(self.k_max) if self.position[i] >= 0.5]


    def assign(self, data, pref_dist):
        """ Returns the cluster closest to each input row """
        clusters = self.position[self.k_max:].reshape([self.k_max, -1])
        clusters = clusters[self.valid_clusters()]
        return np.argmin(cdist(data, clusters, pref_dist), axis=1)


    def update_cs(self):
        """ Updates the CS score """
        if self.metric == Metric.CLUSTER_SEPARATION:
            func = self.compute_cs
        elif self.metric == Metric.CALINSKI:
            func = self.compute_calinski
        else:
            raise ValueError("Unvalid metric %s" % self.metric)

        self.cs = func(data=self.data, pref_dist=self.pref_dist)


    def compute_cs(self, data, pref_dist='euclidean'):
        """ Computes the Cluster separation as described in:

        [H. Peng, J. Wang, P. Shi, A. Riscos-Nunez, M.J. Perez-Ramirez]
        An automatic clusterin
        g algorithm inspired by membrane computing

        and using fixes for empty clusters or < 2 active clusters from:

        [S. Das, A. Abraham, A. Konar]
        Automatic kernel clustering with a Multi-Elitist Particle
        Swarm Optimization Algorithm

        Complexity: O(N^2)

        Args:
            data: Source data
            pref_dist: Distance to use
        Returns:
            coeff: Cluster separation coefficient
        """
        # Check at least 2 clusters are valid. Otherwise, fix it
        active = self.valid_clusters()
        if len(active) < 2:
            active = self.update_activations()

        # Get cluster for each instance
        assigned = self.assign(data, pref_dist)

        # Get rows for each active cluster
        indexes = {k: np.where(assigned == i)[0] for i,k in enumerate(active)}

        # Check clusters have at least 2 points. Otherwise, restructure
        indexes = self.check_small_clusters(indexes)

        # Compute numerator as averaged sum of maximum distances within cluster
        num = sum([self.max_sum(v, data, pref_dist) for k,v in indexes.iteritems()])

        # Compute distance between clusters
        clus_dists = self.cluster_dists(indexes, data, pref_dist)

        # Compute sum of minimum distances between clusters
        den = sum([self.min_sum(clus_dists, i, active) for i, k in enumerate(active)])

        result = float(num)/den

        # We know denominator is never 0 at this point
        return result


    def compute_calinski(self, data, pref_dist):
        """ Computes the Calinkski Harabaz metric
        Complexity: O(N) [Harabaz] + O(N*k) [Cluster distances] -> O(Nk)

        Args:
            data: Source data
            pref_dist: Distance to use
        Returns:
            calinski: Calinski-Harabaz score
        """
        # Check at least 2 clusters are valid. Otherwise, fix it
        active = self.valid_clusters()
        if len(active) < 2:
            active = self.update_activations()

        # Get cluster for each instance
        assigned = self.assign(data, pref_dist)
        indexes = {k: np.where(assigned == i)[0] for i,k in enumerate(active)}

        # Check clusters have at least 2 points. Otherwise, restructure
        self.check_small_clusters(indexes)
        assigned = self.assign(data, pref_dist)
        return calinski_harabaz_score(data, assigned)


    def clip_activations(self):
        """ Clip cluster activations to 0 and 1 in case they are out of range [0,1] """
        for i in range(self.k_max):
            if self.position[i] < 0:
                self.position[i] = 0
            elif self.position[i] > 1:
                self.position[i] = 1


    def update_activations(self):
        """ If less than 2 activated clusters, we activate up to 2 of them with
        random values between 0.5 and 1 """
        active = self.valid_clusters()
        unactive = [i for i in range(self.k_max) if i not in active]
        indexes = np.random.permutation(unactive)[:(2 - len(active))]
        for i in indexes:
            self.position[i] = self._to_range(np.random.rand(), 0, 1, 0.5, 1)
        return indexes.tolist() + active


    def check_small_clusters(self, c_indexes):
        """ If any clusters is found with less than 2 points, recompute cluster
        points so n/k points goes into each cluster """
        less = np.any([len(v) < 2 for k, v in list(c_indexes.iteritems())])
        if less:
            return self.restart_clusters()
        else:
            # Otherwise remain unchanched
            return c_indexes


    def restart_clusters(self):
        """ Recomputes new clusters centers for the current active ones
        balancing the data in the clusters. Returns dictionary pf instance indexes
        in each cluster """

        # Distribute balanced data
        active = self.valid_clusters()
        shards = np.linspace(0, self.data.shape[0], len(active) + 1)
        data_permutation = np.random.permutation(self.data.shape[0])

        # Redistribute points in an equitative and random way
        inds = {k: data_permutation[int(shards[int(i)]):int(shards[i + 1] - 1)]
                for i, k in enumerate(active)}

        # Recompute centers
        for (k, v) in inds.iteritems():
            cluster_center = np.mean(self.data[v], axis=0)
            init_ptr = self.k_max + (k * self.data.shape[1])
            end_ptr = self.k_max + ((k + 1) * self.data.shape[1])
            self.position[init_ptr:end_ptr] = cluster_center

        return inds


    @staticmethod
    def max_sum(indices_k, data, dist):
        """ Computes the sum of maximum distances withing the same cluster:

            1/N_i * sum_{i=1}^{N_i} max_{X_q \in C_i} d(x_i, x_q)

            N_i: Number of instances in cluster i
            C_i: Instances in cluster i
         """
        data_cluster = data[indices_k, :]
        dists = pdist(data_cluster, dist)
        # Get maximum distance for each row
        n_i = len(indices_k)
        sum_maxs = 0.0
        for x in range(n_i):
            dist_to_others = [dists[square_to_condensed(x, i, n_i)] for i in range(n_i) if i != x]
            sum_maxs += max(dist_to_others)
        return float(sum_maxs) / n_i


    @staticmethod
    def min_sum(dists, k, active):
        """ Returns the minimum distance from cluster k to another cluster center

            min_{j \in k, i != j} (d(m_i, m_j))

            k: Current cluster
            m_i: First cluster center
            m_j: Second cluster center

         """
        num_clusters = len(active)
        distances = [dists[square_to_condensed(k, i, num_clusters)]
                     for i in range(num_clusters) if i != k]
        return min(distances)


    @staticmethod
    def cluster_dists(cluster_indexes, data, dist):
        """ Computes the condensed distance matrix for clusters """
        rows = np.zeros((len(cluster_indexes.keys()), data.shape[1]))
        for i,k in enumerate(cluster_indexes.keys()):
            rows[i, ...] = np.mean(data[cluster_indexes[k]], axis=0)
        return pdist(rows, dist)


    @staticmethod
    def _to_range(value, old_min, old_max, new_min, new_max):
        """ Converts value into a new range """
        return (((value - old_min) * (new_max - new_min)) / (old_max - old_min)) \
               + new_min


    def __str__(self):
        return 'Object {}: {}'.format(self.position, self.cs, self)


class Object(object):

    """ Represents the state of an object contained in a cell """

    def __init__(self, data, k_max, lower_b, upper_b, metric, pref_dist):
        """ Builds an object
        Args:
            data: Numpy N x M data matrix where N are rows and M columns
            k_max: Maximum number of clusters to consider
            lower_b: Array where ith position is the considered lower bound
                for feasible values of the ith data feature
            upper_b: Array where ith position is the considered upper bound
                for feasible values of the ith data feature
            metric: Metric to use to guide evolution
            pref_dist: Preferred distance to use in computations.
        """
        if len(lower_b) != len(upper_b):
            raise ValueError('Lower, upper and integer list must have same length')
        self.obj = None
        self.best = None
        self._initialize_state(data, k_max, lower_b, upper_b, metric, pref_dist)
        self.best_list = []


    def _initialize_state(self, data, k_max, lower_b, upper_b, metric, pref_dist):
        """ Initializes the current and best object randomly """
        self.obj = ObjectStr(data=data, k_max=k_max, lower_b=lower_b,
                             upper_b=upper_b, metric=metric, pref_dist=pref_dist)
        self.best = object_copy(self.obj, data, k_max, metric, pref_dist)


    def evolve(self, c1, c2, c3, w, best_cell, best_ext):
        """ Evolves the current object using the velocity-position model
        Args:
            c1: first learning factor
            c2: second learning factor
            c3: third learning factor
            w: inertia weight
            best_cell: Best object from same cell. If None, do not contribute
            best_ext: Best object from other cells. If None, do not contribute
        """

        if best_cell is None or best_ext is None:
            return self

        # Compute and add velocity
        vel_w = np.random.rand() * c1 * (self.best.position - self.obj.position)
        vel_cell = np.random.rand() * c2 * (best_cell.position - self.obj.position)
        vel_ext = np.random.rand() * c3 * (best_ext.position - self.obj.position)
        velocity = (w * self.obj.velocity) + vel_w + vel_cell + vel_ext

        # Update position and velocity
        self.obj.position += velocity
        self.obj.velocity = velocity

        self.obj.clip_activations() # Ensure activations stay in [0,1]

        # Check if better than current best
        self.obj.update_cs()

        # Update best object, if improved
        if self.obj.cs <= self.best.cs:
            self.best = object_copy(self.obj, self.obj.data, self.obj.k_max,
                                    self.obj.metric, self.obj.pref_dist)

        # Track best at this iteration
        self.best_list.append(self.best)

        return self


    def __str__(self):
        return 'Object {}:. Best: {}'.format(self.obj, self.best)


    def plot(self, nrows, fig_size, fonts, save=None):
        """ Plots best local for cell through time """
        plot_grid(data=[("Object", self.best_list)], nrows=nrows, fig_size=fig_size,
                  fonts=fonts, save=save)


class Cell(object):

    """ Represents an agent in the system that contains several solutions """

    def __init__(self, c_id, m, data, k_max, lower_b, upper_b, metric, pref_dist):
        """ Builds a cell. For further information of arguments check Object constructor """
        self.id = c_id
        self.m = m
        self.objects = []
        for i in range(m):
            print('Creating object %d' % i)
            self.objects.append(Object(data, k_max, lower_b, upper_b, metric, pref_dist))
        self.best_ext = None
        self.best_cell = None
        self.bests = []


    def evolve_cell(self, objs):
        """ Evolves cell by obtaining new evolved objects """
        self.objects = objs
        # Get best so far and track it
        self.best_cell = self._get_best_object()
        self.bests.append(self.best_cell)


    def _get_best_object(self):
        """ Returns the best object found by the cell so far """
        best = min(self.objects, key=lambda x: x.best.cs)
        options = [obj.best for obj in self.objects if obj.best.cs == best.best.cs]
        index = np.random.randint(0, len(options))
        return object_copy(options[index], data=options[index].data,
                k_max=options[index].k_max, metric=options[index].metric, pref_dist=options[index].pref_dist)


    def send_external_best(self, objs):
        """ Receives the best options from other cells and updates best external object"""
        _, self.best_ext = get_random_best(objs)


    def plot(self, nrows, fig_size, fonts, save=None):
        """ Plots best local for cell through time """
        plot_grid(data=[("Cell %d" % self.id, self.bests)], nrows=nrows,
                  fig_size=fig_size, fonts=fonts, save=save)


    def plot_objects(self, nrows, fig_size, fonts, save=None):
        """ Plots best local for objetcs in the cell through time"""
        obj_data = [("Cell %d, Object %d" % (self.id, i), obj.best_list)
                    for (i, obj) in enumerate(self.objects)]
        plot_grid(data=obj_data, nrows=nrows, fig_size=fig_size, fonts=fonts, save=save)


    def __str__(self):
        out = 'Cell ' + str(self.id) + ':\n'
        for obj in self.objects:
            out += '--> ' + str(obj) + '\n'
        return out


class Environment(object):

    """ Environment class that controlls the whole system and that saves
    the best clustering solution. The default parameters are the ones extracted
    from the paper experiments """

    def __init__(self,
                 data,
                 k_max,
                 lower_b,
                 upper_b,
                 nworkers=None,
                 max_t=300,
                 q=4,
                 m=20,
                 min_w=0.02,
                 max_w=0.9,
                 metric=Metric.CLUSTER_SEPARATION,
                 pref_dist='euclidean',
                 c1=1.0,
                 c2=1.0,
                 c3=1.0):
        """ Builds environment
        Args:
            data: Numpy N x M data matrix where N are rows and M columns
            k_max: Maximum number of clusters to consider
            lower_b: Lower bound of the data
            upper_b: Upper bound of the data
            nworkers: Number of parallel processes to use. By default, uses all possible CPUs
            max_t: Maximum number of iterations
            q: Number of cells
            m: Objects per cell
            min_w: Initial inertia parameter
            max_w: Maximum intertia
            metric: Metric to guide evolution of objects' cells
            pref_dist: Distance to use. Default is euclidean
            c1, c2, c3: Learning coefficients
        """
        self.data = data
        self.q = q
        self.m = m
        self.nworkers = -1 if nworkers is None else nworkers
        self.max_t = max_t
        self.k_max = k_max
        self.min_w = min_w
        self.w = min_w
        self.max_w = max_w
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.metric = metric
        self.pref_dist = pref_dist
        self.best_global = None
        self.globals_list = []
        self._initialize_cells(data, lower_b, upper_b, metric, pref_dist)


    def _initialize_cells(self, data, lower_b, upper_b, metric, pref_dist):
        """ Creates cells using independent processes """
        with Parallel(n_jobs=self.nworkers) as parallel:
            jobs = []
            for c in range(self.q):
                print('Creating cell %d' % c)
                jobs.append(delayed(create_cell)(c, self.m, data, self.k_max,
                                          lower_b, upper_b, metric, pref_dist))
            self.cells = parallel(jobs)


    def fit(self):
        """ Starts to iterate clustering the data """
        for t in tqdm(range(0, self.max_t)):
            self._perform_step(t)
        return self


    def predict(self, X):
        """ Returns the clusters for each input row """
        return self.best_global.assign(X, self.pref_dist)


    def _perform_step(self, t):
        """ Performs a single iteration in the system """
        with Parallel(n_jobs=self.nworkers) as parallel:

            # Iterate cells
            jobs = []
            for c in self.cells:
                cell_id = c.id
                best_cell, best_ext = c.best_cell, c.best_ext

                # One job per object cell
                for obj in c.objects:
                    jobs.append(delayed(evolve_obj)
                                (cell_id, obj, self.c1, self.c2, self.c3, self.w, best_cell, best_ext))

            # Objects are returned as a tuple (cell_id, object)
            evolved = parallel(jobs)

            # Sort by cells and assign them
            cells = {}
            for (c_id, obj) in evolved:
                if c_id in cells:
                    cells[c_id].append(obj)
                else:
                    cells[c_id] = [obj]

            # Assign objects to each cell
            for c in self.cells:
                c.evolve_cell(cells[c.id])

            # Get cell id and best solution for each
            best_cells = [(c.id, c.best_cell) for c in self.cells]

            # 'Send' best q-1 options to each cell
            self._send_objects(best_cells)

            # Update inertia
            self.w = self.max_w - (self.max_w - self.min_w) * (t/self.max_t)

        # Update global solution
        self._update_global(best_cells)

        # Keep track of the best option so far
        self.globals_list.append(self.best_global)


    def _send_objects(self, best_cells):
        """ Send a random object from the best external cells """
        for c in self.cells:
            best = [(cid, obj) for (cid, obj) in best_cells if cid != c.id]  # q-1 cells
            c.send_external_best(best)


    def _update_global(self, best):
        """ Updates the global solution using the best """
        _, best_option = get_random_best(best)
        if self.best_global is None or best_option.cs <= self.best_global.cs:
            # Update global because first found or improved by other (copy)
            self.best_global = object_copy(best_option, self.data, self.k_max, self.metric, self.pref_dist)


    def plot(self, cell_rows=2, object_rows=4, fig_size=(20,8), fonts=15, save=None):
        """ Plots the best global solution fitness at each step. If path provided,
        it saves the figure into the location
        Args:
            cell_rows: Number of rows to use for the plot of the globals of the cells
            object_rows: Number of rows to use for the plot of the objects of each cell
            fig_size: Tuple with dimensions of the figure
            fonts: Size of the fonts in the plots
            save: Folder where to store the figures. Set to None to display
        """
        # Plot bets global
        global_path = os.path.join(save, "global.png") if save is not None else None
        plot_grid(data=[('Global', self.globals_list)],
                  nrows=1,
                  fig_size=fig_size,
                  fonts=fonts,
                  save=global_path)

        # Plot cells
        cell_path = os.path.join(save, "cells.png") if save is not None else None
        cell_info = [('Cell: %d' % c.id, c.bests) for c in self.cells]
        plot_grid(data=cell_info,
                  nrows=cell_rows,
                  fig_size=fig_size,
                  fonts=fonts,
                  save=cell_path)

        # Plot cell's objects
        for c in self.cells:
            objs_path = os.path.join(save, "cell_%d.png" % c.id) if save is not None else None
            c.plot_objects(nrows=object_rows, fig_size=fig_size, fonts=fonts, save=objs_path)


def plot_grid(data, nrows, fig_size=(8, 3), fonts=15, save=None):
    """" Plots the input data as a grid. Data must be a list of tuples
    of the form (title, best_objects) """

    num = len(data)

    # Compute grid dimensions
    ncols = int(np.ceil(num / float(nrows)))
    nrows = nrows if num > ncols else 1

    # Subplot image with margins
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)
    fig.tight_layout()
    axes = axes if isinstance(axes, np.ndarray) else np.array(axes)

    # Fill grid with plots
    for i in range(nrows):
        for j in range(ncols):

            # Compute axes index and column to plot
            index = np.ravel_multi_index((i, j), dims=(nrows, ncols), order='F')

            if index >= num:
                # May be that grid is larger than number of colums
                # Remaining columns to be left blank
                fig.delaxes(axes.ravel()[index])
            else:
                # Send column to plot in the corresponding axis
                title, objects = data[index]
                steps = range(len(objects))
                ax = axes.ravel()[index]
                ax.plot(steps, [x.cs for x in objects])

                # Adjust title and font
                ax.set_title(title, fontsize=fonts)
                ax.yaxis.label.set_size(fonts)
                ax.xaxis.label.set_size(fonts)

    if save is not None:
        fig.savefig(save)
    else:
        plt.show()
    plt.close(fig)


def create_cell(c, m, data, k_max, lower_b, upper_b, metric, pref_dist):
    """ Wrapper function to create a new cell """
    return Cell(c, m, data, k_max, lower_b, upper_b, metric, pref_dist)


def evolve_obj(c_id, obj, c1, c2, c3, w, best_cell, best_ext):
    """ Auxiliar function to evolve the input cell """
    return c_id, obj.evolve(c1, c2, c3, w, best_cell, best_ext)


def object_copy(obj, data, k_max, metric, pref_dist):
    """ Creates a copy of the given solution"""
    return ObjectStr(data=data, k_max=k_max, metric=metric, pref_dist=pref_dist,
          position=obj.position.copy(), velocity=obj.velocity.copy(), cs=obj.cs.copy())


def get_random_best(objs):
    """ Given a set of tuples (cell_id, Object), returns a random one from the
    ones with lowest cluster separation """
    c_id, best = min(objs, key=lambda x: x[1].cs)
    options = [(c_id, obj) for (c_id, obj) in objs if obj.cs == best.cs]
    index = np.random.randint(0, len(options))
    return options[index]


def square_to_condensed(i, j, n):
    """ Returns the index for the condensed distance matrix. It is faster than computing
    its square form. Code from:
    http://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist"""
    if i < j:
        i, j = j, i
    return n * j - j * (j + 1) / 2 + i - 1 - j


def data_bounds(data):
    """ Returns array of lower and upper bounds of the data columns.
    Minimum and maximum values are taken as the 5 and 95 percentiles """
    return np.percentile(data, 10, axis=0), np.percentile(data, 90, axis=0)
