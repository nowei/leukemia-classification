import json 
import numpy as np
from scipy.stats import ttest_ind
from collections import Counter
from sklearn.model_selection import train_test_split

verbose = True

with open('mappings/patient_to_type.json', 'r') as f:
    patient_to_type = json.load(f)
with open('mappings/num_to_label.json', 'r') as f:
    num_to_label = json.load(f)
with open('mappings/label_to_num.json', 'r') as f:
    label_to_num = json.load(f)
with open('mappings/num_to_feature.json', 'r') as f:
    num_to_feature = json.load(f)

X_raw = []
with open('data/GSE13159.U133Plus2_EntrezCDF.MAS5.log2.pcl', 'r') as f:
    y = np.array([patient_to_type[patient] for patient in f.readline().strip().split('\t')[2:]])
    for line in f:
        data = line.strip().split('\t')[2:]
        X_raw.append([float(entry) for entry in data])
X = np.vstack(X_raw).T
counts = Counter(y)

print('y.shape = {}'.format(y.shape))
print('X.shape = {}'.format(X.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=527, stratify=y)

normal_label = label_to_num['Non-leukemia and healthy bone marrow']

normal_samples = X_train[y_train == normal_label]

label_to_features = {}

def bonferroni(pvalues):
    m = len(pvalues)
    adj_pvalues = []
    for p in pvalues:
        adj_pvalues.append(min(p * m, 1))
    return adj_pvalues

for label in sorted(counts): 
    if label == normal_label:
        continue

    comp_samples = X_train[y_train == label]

    # Welch's t-test
    pvalues = []
    for i in range(len(X_train)):
        statistic, pvalue = ttest_ind(normal_samples[:, i], comp_samples[:,i], equal_var=False)
        pvalues.append(pvalue)
    label_to_features[label] = bonferroni(pvalues)

def featurize(p_threshold):
    total_features = set()
    label_to_selected_features = {}
    for label in label_to_features:
        p = label_to_features[label]
        selected_features = set([i for i in range(len(p)) if p[i] < p_threshold])
        label_to_selected_features[label] = selected_features
        print(label, len(selected_features))
        total_features |= selected_features

    print('total number of features: {}'.format(len(total_features)))

    with open('features/p{}_feature_indices.txt'.format(p_threshold), 'w') as f:
        s = ','.join([str(f) for f in total_features])
        # print(s)
        f.write(s)

    print('Jaccard index')
    label_list = sorted(label_to_selected_features)
    similarities = {}
    for i in range(len(label_list)):
        for j in range(i + 1, len(label_list)):
            A = label_to_selected_features[label_list[i]]
            B = label_to_selected_features[label_list[j]]
            inter = A.intersection(B)
            similarities[(label_list[i], label_list[j])] = len(inter) / (len(A) + len(B) - len(inter))

    for pair in similarities:
        print(pair, similarities[pair])

    feature_map = Counter()
    for label in label_to_selected_features:
        for feature in label_to_selected_features[label]:
            feature_map[feature] += 1

    features_sorted = sorted(feature_map, key=lambda x: -feature_map[x])
    for feature in features_sorted: 
        print("{}: {}, ".format(feature, feature_map[feature]), end='')

    print()
    shared_counts = Counter()
    for feature in feature_map:
        shared_counts[feature_map[feature]] += 1
    for count in sorted(shared_counts, key=lambda x: -x): 
        print("{}: {}".format(count, shared_counts[count]))

    count_to_feature_to_diagnosis = {}
    for feature in feature_map:
        count = feature_map[feature]
        if count < 10: continue
        if count not in count_to_feature_to_diagnosis:
            count_to_feature_to_diagnosis[count] = {}
        count_to_feature_to_diagnosis[count][feature] = []
        for label in sorted(label_to_selected_features):
            if feature in label_to_selected_features[label]:
                count_to_feature_to_diagnosis[count][feature].append(label)

    for count in sorted(count_to_feature_to_diagnosis, key=lambda x:-x):
        print(count)
        for feature in count_to_feature_to_diagnosis[count]:
            feature_name = num_to_feature[str(feature)]
            print("{}|{}".format(feature_name, ", ".join([num_to_label[str(label)] for label in count_to_feature_to_diagnosis[count][feature]])))

featurize(0.05)
# featurize(0.10)