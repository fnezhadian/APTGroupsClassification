import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics


target_names = ["APT1", "APT10", "APT19", "APT21", "APT28", "APT29", "APT30", "DarkHotel", "EnergeticBear",
                "EquationGroup", "GorgonGroup", "Winnti"]


def split_dataset(vectors, target):
    return train_test_split(vectors, target, test_size=0.2)


def predict(classifier, X_test):
    return classifier.predict(X_test)


def get_score(classifier, X_test, y_test):
    return classifier.score(X_test, y_test)


def get_classification_report(actual_data, predicted_data):
    report = metrics.classification_report(y_true=actual_data, y_pred=predicted_data, zero_division=0
                                           , target_names=target_names)
    return report


def apply_model(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    prediction_results = predict(classifier, X_test)
    score = get_score(classifier, X_test, y_test)
    return prediction_results, score


def apply_classifiers(vectors, target):
    X_train, X_test, y_train, y_test = split_dataset(vectors, target)
    NB_classifier = GaussianNB()
    KNN_classifier = KNeighborsClassifier(n_neighbors=50)
    RF_classifier = RandomForestClassifier()
    SVM_classifier = svm.SVC()
    DT_classifier = tree.DecisionTreeClassifier()
    LR_classifier = LogisticRegression(max_iter=1000, random_state=0)
    LDA_classifier = LinearDiscriminantAnalysis()

    NB_prediction, NB_Score = apply_model(NB_classifier, X_train, X_test, y_train, y_test)
    KNN_prediction, KNN_Score = apply_model(KNN_classifier, X_train, X_test, y_train, y_test)
    RF_prediction, RF_Score = apply_model(RF_classifier, X_train, X_test, y_train, y_test)
    SVM_prediction, SVM_Score = apply_model(SVM_classifier, X_train, X_test, y_train, y_test)
    DT_prediction, DT_Score = apply_model(DT_classifier, X_train, X_test, y_train, y_test)
    LR_prediction, LR_Score = apply_model(LR_classifier, X_train, X_test, y_train, y_test)
    LDA_prediction, LDA_Score = apply_model(LDA_classifier, X_train, X_test, y_train, y_test)

    print("GaussianNB: ", NB_Score)
    print("KNeighbors: ", KNN_Score)
    print("RandomForest: ", RF_Score)
    print("SVM: ", SVM_Score)
    print("DecisionTree: ", DT_Score)
    print("LogisticRegression: ", LR_Score)
    print("LinearDiscriminantAnalysis: ", LDA_Score)


def main():
    target = np.loadtxt('target.txt', dtype=int)
    vectors = np.loadtxt('vector.txt', dtype=float)
    apply_classifiers(vectors, target)

    
if __name__ == "__main__":
    main()
