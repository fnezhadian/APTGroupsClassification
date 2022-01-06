import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifierer


def split_dataset(vectors, target):
    return train_test_split(vectors, target, test_size=0.2)


def predict(classifier, X_test):
    return classifier.predict(X_test)


def check_score(classifier, X_test, y_test):
    return classifier.score(X_test, y_test)


def apply_model(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    prediction_results = predict(classifier, X_test)
    score = check_score(classifier, X_test, y_test)
    return prediction_results, score


def apply_classifiers(vectors, target):
    X_train, X_test, y_train, y_test = split_dataset(vectors, target)
    NB_classifier = GaussianNB()
    KNN_classifier = KNeighborsClassifier(n_neighbors=3)
    RF_classifier = RandomForestClassifier()

    NB_prediction_results, NB_Score = apply_model(NB_classifier, X_train, X_test, y_train, y_test)
    KNN_prediction_results, KNN_Score = apply_model(KNN_classifier, X_train, X_test, y_train, y_test)
    RF_prediction_results, RF_Score = apply_model(RF_classifier, X_train, X_test, y_train, y_test)


def main():
    target = np.loadtxt('target.txt', dtype=int)
    vectors = np.loadtxt('vector.txt', dtype=float)
    apply_classifiers(vectors, target)

    
if __name__ == "__main__":
    main()
