import numpy as np
from machine_learning import dataset
import time

'''
train_x: shape (m1, n)
train_y: shape (m, )
test_x: shape (m2, n)
k: top k nearst neighbour
'''
def knn(train_x, train_y, test_x, k):
    dist = np.square(test_x[:, np.newaxis, :] - train_x[np.newaxis, :, :]).sum(axis=2)  #  (m2, m1)
    sort_dist_indexes = np.argsort(dist, axis=1)
    top_k_indxes = sort_dist_indexes[:, 0:k]
    top_k_y = train_y[top_k_indxes]
    pred = [np.bincount(y).argmax() for y in top_k_y]
    return pred


mnist = dataset.mnist.load(one_hot=False)
train_xs, train_ys, test_xs, test_ys = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# not enough memory so ....
true_count = 0
batch = 10
index = 0
for i in range(int(len(test_ys) / batch) + 1):
    end_index = len(test_ys) if index + batch > len(test_ys) else index + batch
    tx = test_xs[index:end_index]
    ty = test_ys[index:end_index]
    pred_y = knn(train_xs.reshape(len(train_xs), -1), train_ys, tx.reshape(len(tx), -1), 3)
    true_count += (pred_y == ty).sum()
    accuracy = true_count / end_index
    print("knn in minst accuracy is {}%".format(accuracy * 100))
    if end_index >= len(test_ys): break
    index = end_index
