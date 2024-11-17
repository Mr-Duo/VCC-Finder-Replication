import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_svmlight_file
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef
)
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import re, os

import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
  
def get_commit_id_list(file):
    commit_id_list = list()
    with open(file, "r") as input_file:
        all_lines = input_file.readlines()
    for line in all_lines:
        dirty_id = (line.split('#'))[1].replace('\n', '')
        commit_id_list.append(dirty_id)
    return commit_id_list
   
def load_data(train_file, test_file):
    print("\tStart load")
    (Xgt_train, Ygt_train) = load_svmlight_file(train_file, dtype=bool)
    #Ground Truth Test Data
    (Xgt_test, Ygt_test) = load_svmlight_file(test_file, dtype=bool)
    
    max_features_all_files = max( Xgt_train.shape[1], Xgt_test.shape[1])

    Xgt_train = csr_matrix( Xgt_train, shape=(Xgt_train.shape[0], max_features_all_files ) )
    Xgt_test = csr_matrix( Xgt_test, shape = (Xgt_test.shape[0], max_features_all_files ) )
    print("\tDone load")
    return Xgt_train, Ygt_train, Xgt_test, Ygt_test

def train(Xgt_train, Ygt_train):
    print("\tStart train")
    params_weighted={"max_iter":200000,"class_weight":{0: 1,1: 100}}
    classif = LinearSVC()
    classif.set_params(**params_weighted)
    classif.fit(Xgt_train, Ygt_train)
    print("\tDone train")
    
    return classif

def test(classif, Xgt_test, Ygt_test):
    print("\tStart eval")
    preds_test = classif.predict(Xgt_test)
    probas_test = classif.decision_function(Xgt_test)

    commit_id = get_commit_id_list(test_file)
    out_df = pd.DataFrame(zip(commit_id, Ygt_test, probas_test, preds_test), columns=["commit_hash", "label", "proba", "pred"])
    
    f1 = f1_score(y_true=Ygt_test, y_pred=preds_test)
    accuracy = accuracy_score(y_true=Ygt_test, y_pred=preds_test)
    rc = recall_score(y_true=Ygt_test, y_pred=preds_test)
    pr = precision_score(y_true=Ygt_test, y_pred=preds_test)
    mcc = matthews_corrcoef(y_true=Ygt_test, y_pred=preds_test)
    
    roc_auc = roc_auc_score(y_true=Ygt_test,  y_score=probas_test)
    precision, recall, _ = precision_recall_curve(y_true=Ygt_test, probas_pred=probas_test)
    # recall, precision = zip(*sorted(zip(recall, precision)))
    pr_auc = auc(recall, precision)
    
    score_df = pd.DataFrame([[accuracy, roc_auc, pr_auc, f1, pr, rc, mcc]], columns= ["acc", "roc_auc", "pr_auc", "f1", "prc", "rc", "mcc"])
    print("\tDone eval")
    return out_df, score_df

def run(train_file, test_file, mode):
    if mode not in ["train_only", "test_only", "all"]:
        print("Specify the mode!")
        return None
    
    Xgt_train, Ygt_train, Xgt_test, Ygt_test = load_data(train_file, test_file)
    if mode != "test_only":
        classif = train(Xgt_train, Ygt_train)
    if mode != "train_only":
        out_df, score_df = test(classif, Xgt_test, Ygt_test)
        

    return classif, out_df, score_df
if __name__ == "__main__":
    sampling = "unsampling"
    for setup in ["SETUP1", "SETUP2", "SETUP3", "SETUP4", "SETUP5"]:
        for sampling in ["rus", "unsampling", "ros"]:
            print(f"{setup} - {sampling}: ")
            train_file = f"E:\JIT-VP-Data\FFmpeg\{setup}\{sampling}\{setup}-FFmpeg-features-train.libsvm"
            test_file = f"E:\JIT-VP-Data\FFmpeg\{setup}\{setup}-FFmpeg-features-test.libsvm"
            classif, out_df, score_df = run(train_file, test_file, "all")
            with open(f"model/vccfinder_{setup}_{sampling}.pkl", "wb") as f:
                pickle.dump(classif, f)
            out_df.to_csv(f"predict_scores/vccfinder_{setup}_{sampling}.csv", index=False)
            score_df.to_csv(f"results/results_{setup}_{sampling}.csv", index=False)


    # names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
    #         "Decision Tree",
    #         "Random Forest_depth5__10",
    #         "Random ForestF_depth2_0_200",
    #         "RF_6_0_200",
    #         "Neural Net",
    #         "AdaBoost"
    #         ]

    # classifiers = [
    #     KNeighborsClassifier(3),
    #     SVC(kernel="linear", C=0.025,probability=True),
    #     SVC(gamma=2, C=1,probability=True),
    #     DecisionTreeClassifier(max_depth=5),
    #     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #     RandomForestClassifier(max_depth=2,random_state=0,n_estimators=200),
    #     RandomForestClassifier(max_depth=6,random_state=0,n_estimators=200),
    #     MLPClassifier(alpha=1, max_iter=1000),
    #     AdaBoostClassifier()
    # ]

    # index=0


    # #list of classifiers
    # for name, clf, colour in zip(names, classifiers,colours):
    #     print ("----"+name)
    #     if name == "Gaussian Process" or name == "Decision Tree" or name == "Naive Bayes":
    #         Xtrain = Xgt_train.toarray()
    #         Xtest = Xgt_test.toarray()
    #     else:
    #         Xtrain = Xgt_train
    #         Xtest = Xgt_test
    #     clf.fit(Xtrain, Ygt_train)

    #     if hasattr(clf, "decision_function"):
    #         clf_score_test = clf.decision_function(Xtest)
    #     else:
    #         clf_score_test = clf.predict_proba(Xtest)[:, 1]
        
    #     clf_preds_test = clf.predict(Xtest)
    #     clf_probas_test = clf.predict_proba(Xtest)
    #     clf_precision_test, clf_recall_test, clf_threshold_test = precision_recall_curve(
    #         Ygt_test, clf_score_test
    #     )