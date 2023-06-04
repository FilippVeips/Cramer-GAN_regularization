import numpy as np
from sklearn.neighbors import KernelDensity
from tqdm import tqdm


class ConditionalKDE:
    def __init__(self, **kwargs):
        self.kde_kwargs = kwargs

    def fit(self, target, train_weights, condition):
        assert target.ndim == 2
        assert len(target) == len(condition)

        self.target_data = target
        self.target_weights = train_weights
        self.kde_condition = KernelDensity(**self.kde_kwargs).fit(condition)
        assert self.kde_condition.kernel == "gaussian"

        return self

    def sample(self, condition, sample_bw_factor, neighbors, progress=False):
        bw = self.kde_condition.bandwidth
        bw2 = bw * bw

        result = np.random.normal(
            size=(len(condition), self.target_data.shape[1])
        ) * (bw/sample_bw_factor)

        weights = np.zeros(len(condition))

        # maybe it breaks from memory
        ids, dists = self.kde_condition.tree_.query_radius(
            condition, 3 * bw, return_distance=True, count_only=False, sort_results = True
        )

        indexes = range(0, len(dists))
        elements = zip(indexes, ids, dists, condition)

        new_ids_list = []
        new_dists_list = []
        for x in elements:
            index, id, dist, cond = x
            if len(dist) == 0:
                # calculate new
                new_dist, new_id = self.kde_condition.tree_.query([cond], k=neighbors, sort_results = True)
                new_dist, new_id = new_dist[0], new_id[0]
            else:
                new_dist, new_id = dist, id
            new_ids_list.append(new_id)
            new_dists_list.append(new_dist)

        ids = np.array(new_ids_list)
        dists = np.array(new_dists_list)

        alpha = np.random.uniform(size=len(condition))
        iterator = (
            tqdm(enumerate(zip(ids, dists)), total=len(ids))
            if progress else
            enumerate(zip(ids, dists))
        )
        for i, (ids_i, dists_i) in iterator:
            # checking for sorting
            assert np.array_equal(dists_i, np.sort(dists_i))
            assert np.argmin(dists_i) == 0

            log_weights = -0.5 * (dists_i * dists_i) / bw2
            # When bw to small, cdf all zeros and i_sample = 0,
            cdf = np.cumsum(np.exp(log_weights))
            # for this to be the nearest neighbor, you need to sort and check it
            i_sample = ids_i[(alpha[i] * cdf[-1] < cdf).argmax()]
            result[i] += self.target_data[i_sample]
            weights[i] = self.target_weights[i_sample]

        # return result, weights
        return result


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from itertools import combinations as comb

    # generate train data
    X = np.random.uniform(size=(100000, 5))
    X[:, 4] *= X[:, :3].sum(axis=1)
    X[:, 2] *= X[:, 1]

    for i1, i2 in comb(range(4), 2):
        plt.figure()
        plt.scatter(X[:, i1], X[:, i2], s=0.1, alpha=0.1)
        plt.title(f"{i1} - {i2}")
        plt.show()

    # train in formats (target, condition)
    kde = ConditionalKDE(bandwidth=0.05).fit(X[:, 2:], X[:, :2])

    # generate test conditions
    X_cond_test = np.random.uniform(size=(100000, 2))

    # get targets
    X_targ_test = kde.sample(X_cond_test, progress=True)
    X_test = np.concatenate([X_cond_test, X_targ_test], axis=1)

    for i1, i2 in comb(range(4), 2):
        plt.figure()
        plt.scatter(X_test[:, i1], X_test[:, i2], s=0.1, alpha=0.1)
        plt.title(f"{i1} - {i2}")
        plt.show()
