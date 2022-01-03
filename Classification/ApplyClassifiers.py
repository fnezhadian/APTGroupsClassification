import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifierer


def split_dataset(vectors, target):
    return train_test_split(vectors, target, test_size=0.2)


def apply_GaussianNB(X_train, X_test, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    prediction_results = classifier.predict(X_test)
    score = classifier.score(X_train, y_train)
    return prediction_results, score


def apply_KNN(X_train, X_test, y_train):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    prediction_results = classifier.predict(X_test)
    score = classifier.score(X_train, y_train)
    return prediction_results, score


def apply_KMeans(X_train, X_test, y_train):
    classifier = KMeans(n_clusters=3, random_state=0)
    classifier.fit(X_train)
    prediction_results = classifier.predict(X_test)
    score = classifier.score(X_train, y_train)
    return prediction_results, score


def apply_RandomForest(X_train, X_test, y_train):
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    prediction_results = classifier.predict(X_test)
    score = classifier.score(X_train, y_train)
    return prediction_results, score


def apply_classifier(vectors, target):
    X_train, X_test, y_train, y_test = split_dataset(vectors, target)
    NB_prediction_results, NB_Score = apply_GaussianNB(X_train, X_test, y_train)
    KNN_prediction_results, KNN_Score = apply_KNN(X_train, X_test, y_train)
    KMeans_prediction_results, KMeans_Score = apply_KMeans(X_train, X_test, y_train)
    RF_prediction_results, RF_Score = apply_RandomForest(X_train, X_test, y_train)


def main():
    target = np.loadtxt('target.txt', dtype=int)
    vectors = np.loadtxt('vector.txt', dtype=float)
    apply_classifier(vectors, target)


if __name__ == "__main__":
    main()
