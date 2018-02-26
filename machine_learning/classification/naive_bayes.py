import copy
import numpy as np
from machine_learning import dataset
from collections import namedtuple

'''
num_features: count of features
enums: enumerations basesd on featrues
class_prob: A ndarray with shape (num_categories, num_features, num_feature_enums) 
'''
_DiscreteFeatures = namedtuple('DiscreteFeatures', ['num_features', 'enums', 'class_prob'])
_ContinuousFeatures = namedtuple('ContinuousFeatures', ['num_features', 'class_mu', 'class_sigma'])
_Model = namedtuple('Model', ['features_discrete', 'features_continuous', 'label_enums', 'label_probs'])

def train_naive_bayes(xs_discretes, xs_continuous, ys):
    '''
    :param xs_discrete: A list, each item as discrete features will be used for calculating of enumeration
    :param xs_continuous: A list, each item as continuous features will be used for calculating of mu and sigma
    :param ys: labels
    :return: model

    P(c|x) = P(c)P(x|c)/P(x) = P(c)/P(x) * mutiply(P(x_i|c)

    '''
    enum_y = np.unique(ys)
    enum_y_indexes = [np.where(ys == e)[0] for e in enum_y]

    xs_discretes = xs_discretes or []
    xs_continuous = xs_continuous or []
    features_discrete, features_continuous = [], []
    for x in xs_discretes:
        x = x.reshape((x.shape[0], -1))
        m = len(x)
        num_features = x.shape[1]
        enum_x = np.unique(x)

        # calculate probabilities of featrues in all categories with laplace smoothing
        class_prob = np.array([
            ((x[index, :] == ex).sum(axis=0) + 1) / (m + len(enum_x))
            for ex in enum_x
            for index in enum_y_indexes
        ]).reshape([len(enum_y), len(enum_x), num_features]).transpose(0, 2, 1)

        features_discrete.append(_DiscreteFeatures(num_features, enum_x, class_prob))


    # calculate probabilities of categories with laplace smoothing
    p_c = np.array([
        ((ys == e).sum() + 1) / (len(ys) + len(enum_y))
        for e in enum_y
    ])

    return _Model(features_discrete, features_continuous, enum_y, p_c)


def inference_naive_bayes(xs_discretes, xs_continuous, model):
    feature_probs = []

    xs_discretes = xs_discretes or []
    xs_continuous = xs_continuous or []

    for i in range(len(xs_discretes)):
        x = xs_discretes[i].reshape((xs_discretes[i].shape[0], -1))
        f_x = model.features_discrete[i]
        num_feature_indexes = range(f_x.num_features)
        cond = x[:, :, np.newaxis] == f_x.enums[np.newaxis, np.newaxis, :]
        m_indexes, k_indexes, enum_indexes = np.where(cond)
        enum_indexes = enum_indexes.reshape([len(x), f_x.num_features])  # m x f
        probs = f_x.class_prob[:, num_feature_indexes, enum_indexes] # k x m x f
        feature_probs.append(probs.transpose(1, 0, 2))  # m x k x f

    feature_probs = np.concatenate(feature_probs, axis=2)
    category_log_probs = np.log(feature_probs).sum(axis=2)
    label_log_probs = category_log_probs + np.log(model.label_probs)
    return model.label_enums[label_log_probs.argmax(axis=1)]



def _binarization(x):
    x = copy.deepcopy(x)
    c = x > 0.5
    x[c == True] = 1
    x[c == False] = 0
    return x

def train_mnist(train_xs, train_ys, data_type):
    xs_discretes, xs_continuous = [], []
    if data_type == 'discrete':
        xs_discretes.append(_binarization(train_xs))
    else:
        xs_continuous.append(train_xs)

    return train_naive_bayes(xs_discretes=xs_discretes, xs_continuous=xs_continuous, ys=train_ys)


def test_mnist(test_xs, test_ys, model, data_type):
    xs_discretes, xs_continuous = [], []
    if data_type == 'discrete':
        xs_discretes.append(_binarization(test_xs))
    else:
        xs_continuous.append(test_xs)

    pred_ys = inference_naive_bayes(xs_discretes=xs_discretes, xs_continuous=xs_continuous, model=model)
    accuracy = (test_ys == pred_ys).sum() / len(test_xs)
    return accuracy


DATA_TYPE = 'discrete'
#DATA_TYPE = 'continuous'

mnist = dataset.mnist.load(one_hot=False)
train_xs, train_ys, test_xs, test_ys = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
model = train_mnist(train_xs, train_ys, DATA_TYPE)
accuracy = test_mnist(test_xs, test_ys, model, DATA_TYPE)
print('Accuracy {}%'.format(accuracy * 100))
