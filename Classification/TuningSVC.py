import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001]
    , 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
    , 'degree': [0, 1, 2, 3, 4, 5, 6]}


def split_dataset(vectors, target):
    return train_test_split(vectors, target, test_size=0.2)


def predict(classifier, X_test):
    return classifier.predict(X_test)


def get_score(classifier, X_test, y_test):
    return classifier.score(X_test, y_test)


def apply_model(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    prediction_results = predict(classifier, X_test)
    score = get_score(classifier, X_test, y_test)
    return prediction_results, score


def get_param_tuning(vectors, target):
    X_train, X_test, y_train, y_test = split_dataset(vectors, target)
    grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    grid.score(X_test, y_test)
    print(grid.best_estimator_)
    print(grid.best_score_)


def apply_classifiers(vectors, target):
    X_train, X_test, y_train, y_test = split_dataset(vectors, target)
    kernels = ['poly', 'rbf', 'sigmoid', 'linear']

    for kernel in kernels:
        classifier = svm.SVC(kernel=kernel, C=100, gamma=0.1, degree=0)
        prediction, score = apply_model(classifier, X_train, X_test, y_train, y_test)
        print(type(classifier), classifier.kernel, round(score, ndigits=2))


def main():
    dataset_path = "D:\\Material\\Current\\Flow\\Dataset\\GL2Vec"
    target_path = os.path.join(dataset_path, "target.txt")
    vector_path = os.path.join(dataset_path, "vector.txt")

    target = np.loadtxt(target_path, dtype=int)
    vectors = np.loadtxt(vector_path, dtype=float)

    # get_param_tuning(vectors, target)  # result: best estimator -> SVC(C=100, degree=0, gamma=0.1)
    apply_classifiers(vectors, target)  # result: best kernel -> rbf


if __name__ == "__main__":
    main()
