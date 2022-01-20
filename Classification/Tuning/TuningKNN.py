import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

#Hyperparameters range initialization for tuning
param_grid = {'n_neighbors': range(1, 21, 2),
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan', 'minkowski'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
              }

test_sizes = [0.2, 0.3, 0.4, 0.5]
algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']


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

    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # grid = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)
    grid.score(X_test, y_test)
    print(grid.best_estimator_)
    print(grid.best_score_)


def apply_classifiers(vectors, target):
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = split_dataset(vectors, target, test_size)
        for algorithm in algorithms:
            classifier = KNeighborsClassifier(metric='euclidean', n_neighbors=3, weights='distance', algorithm=algorithm)
            prediction, score = apply_model(classifier, X_train, X_test, y_train, y_test)
            print(test_size, type(classifier), classifier.algorithm, round(score, ndigits=2))


def main():
    dataset_path = "D:\\Material\\Current\\Dataset\\Flow\\GL2Vec"
    target_path = os.path.join(dataset_path, "target.txt")
    vector_path = os.path.join(dataset_path, "vector.txt")

    target = np.loadtxt(target_path, dtype=int)
    vectors = np.loadtxt(vector_path, dtype=float)

    # get_param_tuning(vectors, target)  # result: best estimator -> KNeighborsClassifier(metric='euclidean', n_neighbors=3, weights='distance')
    apply_classifiers(vectors, target)  # result: best algorithm -> makes no difference, test_size: 0.2


if __name__ == "__main__":
    main()
