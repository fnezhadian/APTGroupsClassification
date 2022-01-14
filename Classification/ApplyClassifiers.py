import math
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network._multilayer_perceptron import MLPClassifier
from sklearn import metrics


target_names = ["APT1", "APT10", "APT19", "APT21", "APT28", "APT29", "APT30"
                , "DarkHotel", "EnergeticBear", "EquationGroup", "GorgonGroup", "Winnti"]


def split_dataset(vectors, target):
    return train_test_split(vectors, target, test_size=0.2)


def predict(classifier, X_test):
    return classifier.predict(X_test)


def get_score(classifier, X_test, y_test):
    return classifier.score(X_test, y_test)


def get_classification_report(actual_data, predicted_data):
    report = metrics.classification_report(y_true=actual_data, y_pred=predicted_data
                                           , zero_division=0, target_names=target_names)
    return report


def apply_model(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    prediction_results = predict(classifier, X_test)
    score = get_score(classifier, X_test, y_test)
    print(type(classifier), round(score, ndigits=2))
    report = get_classification_report(y_test, prediction_results)
    print(report)
    return prediction_results, score


def apply_classifiers(vectors, target):
    X_train, X_test, y_train, y_test = split_dataset(vectors, target)

    classifiers = [
        LogisticRegression(max_iter=1000, random_state=0),
        GaussianNB(),
        KNeighborsClassifier(n_neighbors=round(math.sqrt(len(X_train)))),
        tree.DecisionTreeClassifier(),
        svm.SVC(),
        # svm.LinearSVC(max_iter=1000),
        # svm.NuSVC(),
        svm.SVR(),
        # svm.LinearSVR(max_iter=1000),
        # svm.NuSVR(),
        MLPClassifier(max_iter=1000)
    ]

    for classifier in classifiers:
        prediction, score = apply_model(classifier, X_train, X_test, y_train, y_test)


def main():
    datasets_path = [
        "D:\\Material\\Current\\Dataset\\Call\\FeatherGraph",
        "D:\\Material\\Current\\Dataset\\Call\\Graph2Vec",
        "D:\\Material\\Current\\Dataset\\Call\\GL2Vec",
        "D:\\Material\\Current\\Dataset\\Flow\\FeatherGraph",
        "D:\\Material\\Current\\Dataset\\Flow\\Graph2Vec",
        "D:\\Material\\Current\\Dataset\\Flow\\GL2Vec"
    ]

    for dataset_path in datasets_path:
        print(dataset_path)
        target_path = os.path.join(dataset_path, "target.txt")
        vector_path = os.path.join(dataset_path, "vector.txt")

        target = np.loadtxt(target_path, dtype=int)
        vectors = np.loadtxt(vector_path, dtype=float)
        apply_classifiers(vectors, target)


if __name__ == "__main__":
    main()
