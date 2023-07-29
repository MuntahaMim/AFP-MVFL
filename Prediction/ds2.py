import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, accuracy_score
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
# from sktime.classification.sklearn import RotationForest
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

# Load the datasets
positive = pd.read_csv('processed_positive_train_dataset2.csv')
negative = pd.read_csv('processed_negative_train_dataset2.csv')
positive_test = pd.read_csv('processed_positive_test_dataset2.csv')
negative_test = pd.read_csv('processed_negative_test_dataset2.csv')

# Convert labels to binary for training sets
positive["#"] = positive["#"].apply(lambda x: 1)
negative["#"] = negative["#"].apply(lambda x: 0)

# Combine the positive and negative datasets
X_train = pd.concat([positive, negative])
y_train = pd.concat([positive["#"], negative["#"]])

# Preprocess the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

positive_test["#"] = positive_test["#"].apply(lambda x: 1)
negative_test["#"] = negative_test["#"].apply(lambda x: 0)

# Preprocess the test data
X_test = pd.concat([positive_test, negative_test])
y_test = pd.concat([positive_test["#"], negative_test["#"]])
X_test_scaled = scaler.transform(X_test)

# Perform feature selection using Random Forest feature selection algorithm
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_train_scaled, y_train)
selected_features = X_train.columns[rf_selector.feature_importances_ > np.mean(rf_selector.feature_importances_)]

# Apply feature selection to training and test data
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Initialize the stratified k-fold cross-validator
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Create a list of classification algorithms
classifiers = [{'name': 'SVM', 'model': SVC(kernel='linear', C=1, probability=True)},
               {'name': 'Random Forest', 'model': RandomForestClassifier(n_estimators=100)},
               {'name': 'AdaBoost', 'model': AdaBoostClassifier()},
               {'name': 'Naive Bayes', 'model': GaussianNB()},
               {'name': 'Logistic Regression', 'model': LogisticRegression(random_state=0)},
               {'name': 'Bernoulli NB', 'model': BernoulliNB(force_alpha=True)},
               {'name': 'Decision Tree', 'model': DecisionTreeClassifier(random_state=0)}]

# Train and evaluate each classifier using stratified k-fold cross-validation on the selected features
results = []
for clf in classifiers:
    accuracy_scores = cross_val_score(clf['model'], X_train_selected, y_train, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(clf['model'], X_train_selected, y_train, cv=skf, scoring='f1')
    mcc_scores = cross_val_score(clf['model'], X_train_selected, y_train, cv=skf, scoring='matthews_corrcoef')
    precision_scores = cross_val_score(clf['model'], X_train_selected, y_train, cv=skf, scoring='precision')
    recall_scores = cross_val_score(clf['model'], X_train_selected, y_train, cv=skf, scoring='recall')
    clf['accuracy'] = accuracy_scores.mean()
    clf['f1'] = f1_scores.mean()
    clf['mcc'] = mcc_scores.mean()
    clf['precision'] = precision_scores.mean()
    clf['recall'] = recall_scores.mean()

    clf['model'].fit(X_train_selected, y_train)
    y_pred = clf['model'].predict(X_test_selected)
    clf['test_accuracy'] = accuracy_score(y_test, y_pred)
    clf['test_f1'] = f1_score(y_test, y_pred)
    clf['test_mcc'] = matthews_corrcoef(y_test, y_pred)
    clf['test_precision'] = precision_score(y_test, y_pred)
    clf['test_recall'] = recall_score(y_test, y_pred)

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    clf['sensitivity'] = sensitivity
    clf['specificity'] = specificity

    results.append(clf)

# Print the results in a table format
headers = ['Algorithm', 'Train Accuracy', 'Train F1', 'Train MCC', 'Train Precision', 'Train Recall',
           'Test Accuracy', 'Test F1', 'Test MCC', 'Test Precision', 'Test Recall', 'Sensitivity', 'Specificity']

table = []
for clf in results:
    row = [clf['name'], clf['accuracy'], clf['f1'], clf['mcc'], clf['precision'], clf['recall'],
           clf['test_accuracy'], clf['test_f1'], clf['test_mcc'], clf['test_precision'], clf['test_recall'],
           clf['sensitivity'], clf['specificity']]
    table.append(row)

print(tabulate(table, headers=headers))

# # Plot ROC curves
# for clf in results:
#     if clf['roc_auc'] is not None:
#         plt.plot(clf['fpr'], clf['tpr'], label=f"{clf['name']} (AUC = {clf['roc_auc']:.2f})")
# plt.plot([0, 1], [0, 1], 'k--', label='Random')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()