from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

# Train Random Forest Classifier
def train_model(features, labels, **kwargs):
    # instantiate model
    model = RandomForestClassifier(n_estimators=50)

    # train model
    model.fit(features, labels)

    return model

# Try out several different classifiers with mostly default settings
# Option for k-fold cross validation to get representative accuracy
def classifier_comparison(feature_sets, labels, cross_validation=None):
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", #"Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]

    data_train, data_test, label_train, label_test = train_test_split(feature_sets, labels, test_size=0.33)
    classifier_scores = []

    for name, clf in zip(names, classifiers):
        clf.fit(data_train, label_train)
        max_score = clf.score(data_test, label_test)

        if cross_validation:
            scores = cross_val_score(clf, feature_sets, labels, cv=cross_validation)
            max_score = scores.mean()

        classifier_scores.append((max_score,clf))
        print('{}: {}'.format(name, max_score))

    return max(classifier_scores, key=lambda x: x[0])
