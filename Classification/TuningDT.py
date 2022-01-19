import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import GridSearchCV

# Hyperparameters range initialization for tuning
# param_grid = {"splitter": ["best", "random"],
#               "max_depth": [1, 3, 5, 7, 9, 11, 12],
#               "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#               "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#               "max_features": ["auto", "log2", "sqrt", None],
#               "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90]}

param_grid = {'splitter': ['best', 'random'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [1, 3, 5, 7, 9, 11, 12],
              'max_features': ["auto", "log2", "sqrt", None],
              }

test_sizes = [0.2, 0.3, 0.4, 0.5]


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

    # tuning_model = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid
    #                             , scoring='neg_mean_squared_error', cv=3, verbose=3)
    # scoring='neg_mean_squared_error', cv=3
    grid = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)
    grid.score(X_test, y_test)
    print(grid.best_estimator_)
    print(grid.best_score_)


def apply_classifiers(vectors, target):
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = split_dataset(vectors, target, test_size)

        classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=13)
        prediction, score = apply_model(classifier, X_train, X_test, y_train, y_test)
        print(test_size, type(classifier), classifier.kernel, round(score, ndigits=2))


def main():
    dataset_path = "D:\\Material\\Current\\Flow\\Dataset\\GL2Vec"
    target_path = os.path.join(dataset_path, "target.txt")
    vector_path = os.path.join(dataset_path, "vector.txt")

    target = np.loadtxt(target_path, dtype=int)
    vectors = np.loadtxt(vector_path, dtype=float)

    # get_param_tuning(vectors, target)  # result: best estimator -> tree.DecisionTreeClassifier(criterion='entropy', max_depth=13)
    apply_classifiers(vectors, target)  # result: test_size: 0.2


if __name__ == "__main__":
    main()
