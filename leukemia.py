import json 
import numpy as np

verbose = True

with open('mappings/patient_to_type.json', 'r') as f:
    patient_to_type = json.load(f)
with open('mappings/num_to_label.json', 'r') as f:
    num_to_label = json.load(f)
    num_to_label = {int(num):num_to_label[num] for num in num_to_label}
with open('mappings/label_to_num.json', 'r') as f:
    label_to_num = json.load(f)
with open('features/p0.05_feature_indices.txt', 'r') as f:
    selected_features = np.array([int(v) for v in f.readline().split(',')])
with open('mappings/num_to_feature.json', 'r') as f:
    num_to_feature = json.load(f)
    num_to_feature = {int(num):num_to_feature[num] for num in num_to_feature}

labels = [num_to_label[num] for num in sorted(num_to_label)] 

X_raw = []
with open('data/GSE13159.U133Plus2_EntrezCDF.MAS5.log2.pcl', 'r') as f:
    y = np.array([patient_to_type[patient] for patient in f.readline().strip().split('\t')[2:]])
    for line in f:
        data = line.strip().split('\t')[2:]
        X_raw.append([float(entry) for entry in data])
X = np.vstack(X_raw).T

print('y.shape = {}'.format(y.shape))
print('X.shape = {}'.format(X.shape))

from collections import Counter 
if verbose: 
    counts = Counter(y)
    print()
    print("There are:")
    for class_num in sorted(counts):
        print('  {} patients that have {}'.format(counts[class_num], num_to_label[class_num]))

from sklearn.model_selection import train_test_split, StratifiedKFold

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=527, stratify=y)

if verbose: 
    print()
    print('X_train.shape = {}, X_test.shape = {}'.format(X_train.shape, X_test.shape))
    print('y_train.shape = {}, y_test.shape = {}'.format(y_train.shape, y_test.shape))
    print('Train/Test split for patients')
    counts_train = Counter(y_train)
    counts_test = Counter(y_test)
    for class_num in sorted(counts):
        print("  {}/{} patients that have {}".format(counts_train[class_num], counts_test[class_num], num_to_label[class_num]))

from sklearn.linear_model import LogisticRegression

def get_top_5_acc(probs, y):
    top_5_log = np.argsort(probs, axis=1)[:,-5:]
    correct = 0
    for i in range(top_5_log.shape[0]):
        if y[i] in top_5_log[i,:]:
            correct += 1
    return correct / y.shape[0]

def normalize_all(X_train, X_test):
    means = np.mean(X_train, axis=0)
    std_dev = np.std(X_train, axis=0)
    return (X_train - means) / std_dev, (X_test - means) / std_dev

def normalize_by_normal(X_train, X_test, y_train):
    selected = X_train[y_train == label_to_num['Non-leukemia and healthy bone marrow']]
    means = np.mean(selected, axis=0)
    std_dev = np.std(selected, axis=0)
    return (X_train - means) / std_dev, (X_test - means) / std_dev

print()
verbose2 = False
def cross_validate(X_train, y_train):
    logistic_accuracies = {'top1':0, 'top5':0}
    logistic_norm_all_accuracies = {'top1':0, 'top5':0}
    logistic_norm_accuracies = {'top1':0, 'top5':0}

    sss = StratifiedKFold(n_splits=5)
    i = 0
    for train_index, validation_index in sss.split(X_train, y_train):
        print('fold {}'.format(i))
        i += 1
        X_cv_train, X_cv_validation = X_train[train_index], X_train[validation_index]
        y_cv_train, y_cv_validation = y_train[train_index], y_train[validation_index]
        if verbose2:
            print()
            counts_cv_train = Counter(y_cv_train)
            counts_cv_validation = Counter(y_cv_validation)
            for class_num in sorted(counts):
                print("  {}/{} patients that have {}".format(counts_cv_train[class_num], counts_cv_validation[class_num], num_to_label[class_num]))

        # test logistic regression
        clf = LogisticRegression(penalty='l1', solver='liblinear').fit(X_cv_train, y_cv_train)
        score = clf.score(X_cv_validation, y_cv_validation)
        print('logistic: {}'.format(score))
        logistic_accuracies['top1'] += score / 5
        logistic_accuracies['top5'] += get_top_5_acc(clf.predict_proba(X_cv_validation), y_cv_validation) / 5

        # featurize by normalizing each gene expression level
        X_cv_train_norm_all, X_cv_validation_norm_all = normalize_all(X_cv_train, X_cv_validation)
        clf = LogisticRegression(penalty='l1', solver='liblinear').fit(X_cv_train_norm_all, y_cv_train)
        score = clf.score(X_cv_validation_norm_all, y_cv_validation)
        print('logistic, normalize all: {}'.format(score))
        logistic_norm_all_accuracies['top1'] += score / 5
        logistic_norm_all_accuracies['top5'] += get_top_5_acc(clf.predict_proba(X_cv_validation_norm_all), y_cv_validation) / 5

        # featurize by normalizing each gene expression level by "Non-leukemia and healthy bone marrow" [9]
        X_cv_train_norm_norms, X_cv_validation_norm_norms = normalize_by_normal(X_cv_train, X_cv_validation, y_cv_train)
        clf = LogisticRegression(penalty='l1', solver='liblinear').fit(X_cv_train_norm_norms, y_cv_train)
        score = clf.score(X_cv_validation_norm_norms, y_cv_validation)
        print('logistic, normalize by normal: {}'.format(score))
        logistic_norm_accuracies['top1'] += score / 5
        logistic_norm_accuracies['top5'] += get_top_5_acc(clf.predict_proba(X_cv_validation_norm_norms), y_cv_validation) / 5

    # compare accuracies of logistic regression, 
    print('logistic')
    print(logistic_accuracies)
    print('logistic - featurize by normalizing each gene expression level')
    print(logistic_norm_all_accuracies)
    print('logistic - featurize by normalizing each gene expression level by "Non-leukemia and healthy bone marrow"')
    print(logistic_norm_accuracies)

do_cv = False
if do_cv: 
    print('cross-validation with all features')
    cross_validate(X_train, y_train)
    print()
    print('cross-validation with selected features')
    cross_validate(X_train[:, selected_features], y_train)

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

def graph_confusion_matrix(clf, X, y, title='', filename=''):
    disp = plot_confusion_matrix(clf, X, y,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues)

    disp.ax_.set_title(title)
    plt.xticks(rotation=90)
    disp.figure_.set_size_inches(15,16)
    plt.savefig('graphs/{}.png'.format(filename), bbox_inches='tight')
    plt.close()

def eval_model(X_train, X_test, y_train, aug=''):
    logistic_accuracies = {'train':{'top1':0, 'top5':0}, 'test':{'top1':0, 'top5':0}}
    logistic_norm_all_accuracies = {'train':{'top1':0, 'top5':0}, 'test':{'top1':0, 'top5':0}}
    logistic_norm_accuracies = {'train':{'top1':0, 'top5':0}, 'test':{'top1':0, 'top5':0}}

    # test logistic regression
    clf = LogisticRegression(penalty='l1', solver='liblinear').fit(X_train, y_train)
    logistic_accuracies['train']['top1'] += clf.score(X_train, y_train)
    logistic_accuracies['train']['top5'] += get_top_5_acc(clf.predict_proba(X_train), y_train)
    score = clf.score(X_test, y_test)
    logistic_accuracies['test']['top1'] += score
    logistic_accuracies['test']['top5'] += get_top_5_acc(clf.predict_proba(X_test), y_test)
    graph_confusion_matrix(clf, X_train, y_train, title='Confusion Matrix of Logistic Regression {}\nw/ L1 regularization (train)'.format(aug), filename='cm_logreg_l1_train_{}'.format(aug))
    graph_confusion_matrix(clf, X_test, y_test, title='Confusion Matrix of Logistic Regression {}\nw/ L1 regularization (test)'.format(aug), filename='cm_logreg_l1_test_{}'.format(aug))

    # featurize by normalizing each gene expression level
    X_train_norm_all, X_test_norm_all = normalize_all(X_train, X_test)
    clf = LogisticRegression(penalty='l1', solver='liblinear').fit(X_train_norm_all, y_train)
    logistic_norm_all_accuracies['train']['top1'] += clf.score(X_train_norm_all, y_train)
    logistic_norm_all_accuracies['train']['top5'] += get_top_5_acc(clf.predict_proba(X_train_norm_all), y_train)
    score = clf.score(X_test_norm_all, y_test)
    logistic_norm_all_accuracies['test']['top1'] += score
    logistic_norm_all_accuracies['test']['top5'] += get_top_5_acc(clf.predict_proba(X_test_norm_all), y_test)
    graph_confusion_matrix(clf, X_train_norm_all, y_train, title='Confusion Matrix of Logistic Regression {}\nw/ L1 regularization + normalizing w/ all (train)'.format(aug), filename='cm_logreg_l1_train_norm_all_{}'.format(aug))
    graph_confusion_matrix(clf, X_test_norm_all, y_test, title='Confusion Matrix of Logistic Regression {}\nw/ L1 regularization + normalizing w/ all (test)'.format(aug), filename='cm_logreg_l1_test_norm_all_{}'.format(aug))

    # featurize by normalizing each gene expression level by "Non-leukemia and healthy bone marrow" [9]
    X_train_norm_norms, X_test_norm_norms = normalize_by_normal(X_train, X_test, y_train)
    clf = LogisticRegression(penalty='l1', solver='liblinear').fit(X_train_norm_norms, y_train)
    logistic_norm_accuracies['train']['top1'] += clf.score(X_train_norm_norms, y_train)
    logistic_norm_accuracies['train']['top5'] += get_top_5_acc(clf.predict_proba(X_train_norm_norms), y_train)
    score = clf.score(X_test_norm_norms, y_test)
    logistic_norm_accuracies['test']['top1'] += score
    logistic_norm_accuracies['test']['top5'] += get_top_5_acc(clf.predict_proba(X_test_norm_norms), y_test)
    graph_confusion_matrix(clf, X_train_norm_norms, y_train, title='Confusion Matrix of Logistic Regression {}\nw/ L1 regularization + normalizing w/ healthy (train)'.format(aug), filename='cm_logreg_l1_train_norm_health_{}'.format(aug))
    graph_confusion_matrix(clf, X_test_norm_norms, y_test, title='Confusion Matrix of Logistic Regression {}\nw/ L1 regularization + normalizing w/ healthy (test)'.format(aug), filename='cm_logreg_l1_test_norm_health_{}'.format(aug))

    # compare accuracies of logistic regression, 
    print('logistic')
    print(logistic_accuracies)
    print('logistic - featurize by normalizing each gene expression level')
    print(logistic_norm_all_accuracies)
    print('logistic - featurize by normalizing each gene expression level by "Non-leukemia and healthy bone marrow"')
    print(logistic_norm_accuracies)

do_eval = False
if do_eval:
    print('eval without significant features')
    eval_model(X_train, X_test, y_train)
    print()
    print('eval with significant features')
    eval_model(X_train[:, selected_features], X_test[:, selected_features], y_train, aug='+ significant features')

from sklearn.metrics.pairwise import cosine_similarity

def check_weights(X_train, y_train):
    clf = LogisticRegression(penalty='l1', solver='liblinear').fit(X_train, y_train)
    types, weights = clf.coef_.shape

    similarities = {}

    for i in range(types):
        for j in range(i, types):
            A = clf.coef_[i].reshape(1, -1)
            B = clf.coef_[j].reshape(1, -1)
            similarities[(i, j)] = cosine_similarity(A, B)[0][0]
            similarities[(j, i)] = similarities[(i, j)]
    
    print('cosine similarity of weights')
    for i in range(types):
        for j in range(types):
            print(round(similarities[(i, j)], 4), end=' ')
        print()
    for i in range(types):
        print(i, min(np.abs(clf.coef_[i])))
        print(np.sum(clf.coef_[i] == 0))
    
    for i in range(types):
        A = clf.coef_[i]
        index_array = np.argsort(A)
        worst = index_array[:10]
        best = index_array[-10:]
        print(num_to_label[i])
        print('best')
        for ind in reversed(best):
            print("{}|{}".format(num_to_feature[ind], A[ind]))
        print('worst')
        for ind in reversed(worst):
            print("{}|{}".format(num_to_feature[ind], A[ind]))


examine_weights = False
if examine_weights:
    check_weights(X_train, y_train)

def get_top_5_acc_knn(probs, y):
    top_5_log = np.argsort(probs, axis=1)[:,-5:]
    top_5 = []
    for x_i in range(len(top_5_log)):
        top_5.append(set([i for i in top_5_log[x_i] if probs[x_i][i] > 0]))
    correct = 0
    for i in range(len(top_5)):
        if y[i] in top_5[i]:
            correct += 1
    return correct / y.shape[0]

from sklearn.neighbors import KNeighborsClassifier

def k_nearest_neighbors(k, Xtrain, ytrain, Xtest, ytest, graph=False):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(Xtrain, ytrain)
    preds = neigh.predict(Xtest)
    correct = 0
    for i in range(len(preds)):
        if ytest[i] == preds[i]:
            correct += 1
    print('top 1 accuracy: {}'.format(correct/len(preds)))
    print('top 5 accuracy: {}'.format(get_top_5_acc_knn(neigh.predict_proba(Xtest), ytest)))
    if graph:
        graph_confusion_matrix(neigh, Xtest, ytest, title='Confusion Matrix of K-Nearest Neighbors (test)', filename='cm_knn_test_{}'.format(k))

do_knn = False
if do_knn:
    print('k nearest neighbors')
    X_train_sig = X_train[:, selected_features]
    X_test_sig = X_test[:, selected_features]
    for k in [3, 5, 10, 15, 20]:
        print(k)
        print('w/o significant')
        k_nearest_neighbors(k, X_train, y_train, X_test, y_test)
        print('w/ significant')
        k_nearest_neighbors(k, X_train_sig, y_train, X_test_sig, y_test)

    k = 10

    k_nearest_neighbors(k, X_train, y_train, X_test, y_test, graph=True)


examine_most_heavy = False
if examine_most_heavy:
    selected = X_train[y_train == label_to_num['mature B-ALL with t(8;14)']]
    print(selected[:,[2978, 6343, 7006, 257]])