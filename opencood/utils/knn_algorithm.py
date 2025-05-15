from collections import Counter

import numpy as np


class KNN:
    def __init__(self, task_type="classification"):
        self.train_data = None
        self.train_label = None
        self.task_type = task_type

    def fit(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label

    def predict(self, test_data, k=3, distance="l2"):
        x = test_data
        # preds = []
        # for x in test_data:
        if distance == "l1":
            dists = self.l1_distance(x)
        elif distance == "l2":
            dists = self.l2_distance(x)
        else:
            raise ValueError("wrong distance type")
        sorted_idx = np.argsort(dists)
        knearnest_labels = self.train_label[sorted_idx[:k]]
        pred = None
        if self.task_type == "regression":
            pred = np.mean(knearnest_labels)
        elif self.task_type == "classification":
            pred = Counter(knearnest_labels).most_common(1)[0][0]
        # preds.append(pred)
        return pred  # s

    def l1_distance(self, x):
        return np.sum(np.abs(self.train_data - x))

    def l2_distance(self, x):
        return np.sum(np.square(self.train_data - x))


if __name__ == "__main__":
    train_data = [[1, 1, 1], [2, 2, 2], [10, 10, 10], [13, 13, 13]]
    # train_label = ['aa', 'aa', 'bb', 'bb']
    train_label = [1, 2, 30, 60]
    test_data = [[3, 2, 4], [9, 13, 11], [10, 20, 10]]
    knn = KNN(task_type="regression")
    knn.fit(train_data, train_label)
    preds = knn.predict(test_data, k=2)
    print(preds)
