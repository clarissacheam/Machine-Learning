import math
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import copy
import matplotlib.pyplot as plt


# k = 0

# returns Euclidean distance between vectors a dn b
def euclidean(a, b):
    # print('a:', a)
    # print('b:', b)
    # a = list(map(int, a))
    # b = list(map(int, b))
    summ = sum((u - v) ** 2 for u, v in zip(a, b))
    dist = math.sqrt(float(summ))
    return dist


# returns Cosine Similarity between vectors a dn b
def cosim(a, b):
    # print('a:', a)
    # print('b:', b)
    # a = list(map(int, a))
    # b = list(map(int, b))
    dot = sum(u * v for u, v in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    dist = float(dot) / (mag_a * mag_b)
    return dist


'test 2 dist function'


def unit_test(a, b):
    euc = euclidean(a, b)
    cos = cosim(a, b)
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.metrics.pairwise import cosine_similarity
    sk_euc = euclidean_distances([a], [b])
    print('Euclidean match sk-learn') if sk_euc == euc else 'Euclidean doesnt match'
    sk_cos = cosine_similarity([a], [b])
    print('Cos similarity match sk-learn') if sk_cos == cos else 'Cos similarity doesnt match'


# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train, query, metric):
    k = 8  # hyper-parameter, could tone
    labels = []
    for query_dat in query:
        query_pt = query_dat[1]
        tuple_lst = []  # the list of tuple (dist, label)
        for train_dat in train:
            train_pt = train_dat[1]
            # print('train_pt:', train_pt, ', query_pt:', query_pt)
            dist = euclidean(train_pt, query_pt) if metric == 'euclidean' \
                else cosim(train_pt, query_pt)
            tuple_lst.append((dist, train_dat[0]))
        tuple_lst.sort()
        # Find closest point and take the majority vote
        # print('KNN\'s labels:', [x[1] for x in tuple_lst[:k]])
        vote = Counter([x[1] for x in tuple_lst[:k]]).most_common()[0][0]
        labels.append(vote)
    return labels


# returns a list of labels for the query dataset based upon observations in the train dataset.
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train, query, metric):
    k = 14
    tol = 0.001
    max_iter = 20
    # train_dat = copy.copy(train)
    # query_dat = copy.copy(query)
    # train = [x[1] for x in train]
    # query = [x[1] for x in query]
    centroids = np.array([])
    classification = {}
    check = True
    # pick centroid
    centroids = np.empty([0, len(train[0])])
    for i in range(k):
        centroids = np.vstack((centroids, list(map(float, train[-i]))))
        classification[i] = np.empty([0, len(train[0])])
    count = 0
    while check:
        # empty current assignments
        for i in range(k):
            classification[i] = np.empty([0, len(train[0])])
        # iterate through train points
        for pt in train:
            min_dist = float('inf')
            for i in range(k):
                if metric == 'euclidean':
                    dist = euclidean(pt, centroids[i])
                else:
                    dist = cosim(pt, centroids[i])
                if dist < min_dist:
                    min_dist = dist
                    classification[i] = np.vstack((classification[i], pt))
            # classification[dist.index(min(dist))] = np.vstack(
            #     (classification[dist.index(min(dist))], pt))
        # for k, val in classification.items():
        #     print('classification', k, ' values:', len(val))
        # recalculate centroids and break condition
        # new_centroids = np.empty([0, len(train[0])])
        check = False
        for i in range(k):
            new_centroid = np.nanmean(classification[i], axis=0)
            # print('Nan:', new_centroid) if np.isnan(new_centroid[0]) else ''
            # print('new centroid:', new_centroid.shape)
            # print(classification[i].astype(np.int).shape)
            # convergence test
            # new_centroids = np.append(new_centroids, new_centroid)
            print('centriods:', centroids[i])
            # print('new_centriod:', new_centroid)
            # try:
            #     tol_score = np.sum((abs(centroids[i] - new_centroid) / (centroids[i])))
            # except RuntimeWarning:
            #     print('zero division')
            tol_score = np.sum((abs(centroids[i] - new_centroid) / (centroids[i] + 0.001)))
            if tol_score > tol:
                check = True
            # ---- only check for all convergence
            centroids[i] = new_centroid
        count += 1
        print('done iteration', count)
        if count >= max_iter:
            check = False

    # predict
    # -- plot centroids
    fig, axs = plt.subplots(k)
    for i in range(k):
        # print('cluster', i, ':', centroids[i])
        axs[i].imshow(centroids[i].reshape(28, 28))
    plt.show()
    clusters = []
    for pt in query:
        min_dist = float('inf')
        cluster = -1
        for i in range(k):
            if metric == 'euclidean':
                dist = euclidean(pt, centroids[i])
            else:
                dist = cosim(pt, centroids[i])
            # print('dist:', dist)
            if dist < min_dist:
                min_dist = dist
                cluster = i
        clusters.append(cluster)
    return clusters


'Output 200x2x784 matrix'
def read_data(file_name):
    data_set = []
    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i + 1])
            data_set.append([label, attribs])
    return data_set


def show(file_name, mode):
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ', end='')
                else:
                    print('*', end='')
            else:
                print('%4s ' % data_set[obs][1][idx], end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0], end='')
        print(' ')


def dimensionality_reduction(filename):
    train_data = read_data(filename)
    train_2d = []
    train_labels = []
    for row in range(len(train_data)):
        X_train = train_data[row][1]
        X_label = train_data[row][0]
        train_2d.append(X_train)
        train_labels.append(X_label)

    pca_trans_data = PCA(n_components=50, svd_solver='randomized', whiten=True).fit(train_2d)
    X_train_pca = pca_trans_data.transform(train_2d)
    X_train = list(np.round(X_train_pca, 2))
    train = [e for e in zip(train_labels, X_train)]
    pca_variance = pca_trans_data.explained_variance_ratio_.sum()

    return train, train_labels


def main():
    # show('valid.csv', 'pixels')
    # ------------- test parameters -------------
    function = 'knn'
    metric = 'euclidean'
    # k = 2
    dimension_red = False
    mannual_label = False    # visiualize the centroids and mannually assign labels
    # -------------------------------------------
    if dimension_red:
        dat_train,train_labels = dimensionality_reduction('train.csv')
        dat_val,val_labels  = dimensionality_reduction('valid.csv')
        dat_test,test_labels = dimensionality_reduction('test.csv')
    else:
        dat_train = read_data('train.csv')
        dat_val = read_data('valid.csv')
        dat_test = read_data('test.csv')
    '''Output 200x2x784 matrix'''
    if function == 'knn':
        pred = knn(dat_train, dat_val, metric)
        actual = test_labels
    elif function == 'kmeans':
        train = [list(map(int, x[1])) for x in dat_train]
        valid = [list(map(int, x[1])) for x in dat_val]
        test = [list(map(int, x[1])) for x in dat_test]
        pred = kmeans(train, valid, metric)
        #actual = test_labels
    # print('predictions:', pred)
    # print('labels:', [x[0] for x in dat_test])
    correct = 0
    if mannual_label:
        clusters = [3, 8, 9, 1, 2, 0, 1]
        labels = list(map(clusters.__getitem__, pred))
        print('labels', len(labels))
    else:
        labels = pred
    for i in range(len(dat_test)):
        correct = correct + (str(labels[i]) == str(dat_val[i][0]))
    acc = float(correct) / len(dat_test)
    print('accuracy:', acc)
    #print(confusion_matrix(actual, pred))
    if function =='softkmeans':
        train1 = [x[1] for x in dat_train]
        valid1 = [x[1] for x in dat_val]
        test1 = [x[1] for x in dat_test]
        pred=soft_kmeans(test1)
        print("Data point highest probabilities (sample set only):")
        print(pred[1:30])


if __name__ == "__main__":
    main()


###Soft k means

def cluster_fn(centers, x, beta):
    # N, D = x.shape
    N = len(x)
    _ = len(x[0])
    K, D = centers.shape
    # K = len(centers)
    # D = len(centers[0])
    R = np.zeros((N, K))

    for n in range(N):
        R[n] = np.exp(-beta * np.linalg.norm(centers - x[n], 2, axis=1))
    R /= R.sum(axis=1, keepdims=True)

    return R


def soft_kmeans(x, k=3, max_iters=20, beta=1.):
    # Initializing centers
    # N, D = x.shape
    N = len(x)
    D = len(x[0])
    centers = np.zeros((k, D))
    arr = []
    for i in range(k):
        j = np.random.choice(N)
        while j in arr:
            j = np.random.choice(N)
        arr.append(j)
        centers[i] = x[j]

    prev_cost = 0

    count = 0
    for _ in range(max_iters):
        count += 1
        print('iteration', count)
        r = cluster_fn(centers, x, beta)

        # Updating centers
        # N, D = x.shape
        N = len(x)
        D = len(x[0])
        centers = np.zeros((k, D))
        for i in range(k):
            centers[i] = r[:, i].dot(x) / r[:, i].sum()

        # Calculating cost
        cost = 0
        for i in range(k):
            norm = np.linalg.norm(x - centers[i], 2)
            cost += (norm * np.expand_dims(r[:, i], axis=1)).sum()

        # Break condition
        if np.abs(cost - prev_cost) < 1e-5:
            break
        prev_cost = cost
    return r