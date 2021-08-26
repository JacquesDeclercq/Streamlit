import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


st.title("# Welcome to my Classifier Page")

st.write("""
# Explore different machine learning classifiers
Which one is the best ? For Which Dataset ?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris Dataset", "Breast Cancer Dataset", "Wine Dataset"))
st.write(dataset_name)

classifier_name = st.sidebar.selectbox("Select a Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris Dataset":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer Dataset":
        data = datasets.load_breast_cancer()
    else :
        data = datasets.load_wine()

    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write("Shape of the Dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = {}
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1,20)
        weights = st.sidebar.selectbox("weights",("uniform", "distance"))
        algorithm = st.sidebar.selectbox("algorithm",("auto","ball_tree","kd_tree","brute"))
        params['K'] = K
        params['weights'] = weights
        params['algorithm'] = algorithm
    elif clf_name == "SVM":
        C = st.sidebar.slider("C",0.01, 10.0)
        kernel = st.sidebar.selectbox("kernel",("linear", "poly", "rbf", "sigmoid", "precomputed"))
        params['C'] = C
        params['gamma'] = "scale"
        if kernel not in ("linear",'precomputed'):
            gamma = st.sidebar.selectbox("gamma",("scale","auto"))
            params['gamma'] = gamma

    else:
        max_depth = st.sidebar.slider("max_depth", 2, 20)
        n_estimators = st.sidebar.slider("n_estimator",1,200)
        criterion = st.sidebar.selectbox("Select a measure metric", ("gini","entropy"))
        max_features = st.sidebar.selectbox("n_features",("auto", "sqrt", "log2"))
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        params["max_features"] = max_features

    return params


params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"], weights=params["weights"], algorithm=params["algorithm"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"], gamma=params["gamma"])
    else:
        clf = RandomForestClassifier(max_depth=params["max_depth"],
                                     n_estimators=params["n_estimators"],
                                     criterion=params["criterion"],
                                     max_features=params["max_features"], random_state=42)

    return clf

clf = get_classifier(classifier_name, params)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f" Prediction Accuracy of the Model = {accuracy}")

#PLOT

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]
plt.scatter(x1,x2, c=y, alpha = 0.8, cmap="viridis")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
