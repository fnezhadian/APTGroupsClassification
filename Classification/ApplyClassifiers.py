import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def split_dataset(vectors, target):
    return train_test_split(vectors, target, test_size=0.2)


def apply_GaussianNB(X_train, X_test, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test)


def apply_KNN(X_train, X_test, y_train):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test)


def apply_classifier(vectors, target):
    X_train, X_test, y_train, y_test = split_dataset(vectors, target)
    GaussianNB_prediction_results = apply_GaussianNB(X_train, X_test, y_train)
    KNN_prediction_results = apply_KNN(X_train, X_test, y_train)


def main():
    target = np.loadtxt('target.txt', dtype=str)
    vectors = np.loadtxt('vector.txt', dtype=float)
    apply_classifier(vectors, target)


if __name__ == "__main__":
    main()
