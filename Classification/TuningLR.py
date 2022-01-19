import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Hyperparameters range initialization for tuning
param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'penalty': ['none', 'l1', 'l2', 'elasticnet'],
              'C': [100, 10, 1.0, 0.1, 0.01],
              'multi_class': ['auto', 'ovr', 'multinomial'],
              'max_iter': [100, 400, 1000, 2000]}

# Warning The choice of the algorithm depends on the penalty chosen: Supported penalties by solver:
# param_grid_1 = {'solver': ['newton-cg'], 'penalty': ['none', 'l2']}
# param_grid_2 = {'solver': ['lbfgs'], 'penalty': ['none', 'l2']}
# param_grid_3 = {'solver': ['liblinear'], 'penalty': ['l1', 'l2']}
# param_grid_4 = {'solver': ['sag'], 'penalty': ['none', 'l2']}
# param_grid_5 = {'solver': ['saga'], 'penalty': ['l1', 'l2', 'elasticnet']}

# Solver newton-cg supports only 'l2' or 'none' penalties
# penalty='none' will ignore the C and l1_ratio parameters
penalties = ['none', 'l2']

test_sizes = [0.2, 0.3, 0.4, 0.5]
max_iters = [100, 400, 1000, 2000]


def split_dataset(vectors, target, test_size):
    return train_test_split(vectors, target, test_size=test_size)


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
    X_train, X_test, y_train, y_test = split_dataset(vectors, target, test_size=0.2)

    grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)
    grid.score(X_test, y_test)
    print(grid.best_estimator_)
    print(grid.best_score_)


def apply_classifiers(vectors, target):
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = split_dataset(vectors, target, test_size)
        for penalty in penalties:
            for max_iter in max_iters:
                classifier = LogisticRegression(C=100, solver='newton-cg', max_iter=max_iter, penalty=penalty)
                prediction, score = apply_model(classifier, X_train, X_test, y_train, y_test)
                print(test_size, type(classifier), classifier.max_iter, classifier.penalty, classifier.multi_class, round(score, ndigits=2))


def main():
    dataset_path = "D:\\Material\\Current\\Dataset\\Flow\\GL2Vec"
    target_path = os.path.join(dataset_path, "target.txt")
    vector_path = os.path.join(dataset_path, "vector.txt")

    target = np.loadtxt(target_path, dtype=int)
    vectors = np.loadtxt(vector_path, dtype=float)

    # get_param_tuning(vectors, target)  # result: best estimator -> LogisticRegression(C=100, max_iter=1000, solver='newton-cg')
                                                                   # LogisticRegression(C=100, solver='newton-cg')
    apply_classifiers(vectors, target)  # result: best penalty -> l2, test_size: 0.2


if __name__ == "__main__":
    main()
