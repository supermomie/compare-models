#!/usr/bin/env python
# coding: utf-8

# ### IMPORT LIBS

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored
from tqdm import tqdm


# ### IMPORT DATABASE



data = pd.read_csv("/home/fakhredineatallah/Documents/microsoft/DB/CSV/credit.csv")

print(data)
print(data.columns)


# ### CHECK CORRELATIONS


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

# ### DATA CLEANING


data = data.drop(['Unnamed: 0', 'Cle'], axis=1)
print(data)


# ### ONE HOT ENCODING

data = pd.get_dummies(data)
print(data)

# ### CREATING TRAIN/TEST SET


from sklearn.model_selection import train_test_split

X = data.drop('Cible',axis=1)
y = data['Cible']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### IMPORT MODELS

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV


# ### IMPORT METRICS


from sklearn.metrics import accuracy_score, log_loss, make_scorer, r2_score, mean_squared_error, f1_score, precision_score, jaccard_score, recall_score, roc_auc_score


# ### FEED MODELS AND PARAMETERS


seed=1
models = [
            'ADB',
            'DTC',
            'GBC',
            'RFC',
            'KNC',
            'SVC',
            'LOGREG',
         ]
clfs = [
        AdaBoostClassifier(random_state=seed),
        DecisionTreeClassifier(random_state=seed),
        GradientBoostingClassifier(random_state=seed),
        RandomForestClassifier(random_state=seed, n_jobs=-1),
        KNeighborsClassifier(n_jobs=-1),
        SVC(random_state=seed, probability = True),
        LogisticRegression(random_state=seed, n_jobs=-1),
        ]

params = {
            models[0]: {'learning_rate':[1, 0.5, 0.1, 0.02, 0.01, 0.002, 0.001], 'n_estimators':np.arange(1, 150)},
            models[1]: {'criterion':['gini', 'entropy'], 'splitter' : ['best', 'random'], 
                        'max_depth':np.arange(1, 30), 'min_samples_split':np.arange(2, 30), 
                       'min_samples_leaf': np.arange(1, 30)},
            models[2]: {'learning_rate':[1, 0.5, 0.1, 0.02, 0.01, 0.002, 0.001],'n_estimators':np.arange(1, 150), 
                       'max_depth':np.arange(1, 30), 'min_samples_split':np.arange(2, 30), 
                       'min_samples_leaf': np.arange(1, 30)},
            models[3]: {'n_estimators':np.arange(1, 250), 'criterion':['gini', 'entropy'],
                        'min_samples_split':np.arange(2, 30), 'min_samples_leaf': np.arange(1, 30)},
            models[4]: {'n_neighbors':np.arange(1, 30), 'weights':['distance'],'leaf_size':np.arange(1, 30)},
            models[5]: {'C': np.arange(1, 150), 'tol': [0.005], 
                        'kernel':['sigmoid', 'linear', 'poly', 'rbf', 'precomputed'], 'degree' : np.arange(1, 20)},
            models[6]: {'C':[2000], 'tol': [0.0001]},
         }



scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}


# ### TRAINING


test_scores = []

start = time.perf_counter()
for name, estimator in zip(models,clfs):
    print("Training for :",colored(name, "green", attrs=['reverse']))
    startTrain = time.perf_counter()
    
    clf = GridSearchCV(estimator, params[name], scoring=scoring, refit='AUC', n_jobs=-1, cv=5, verbose=1, 
                       return_train_score=False)
    clf.fit(X_train, y_train)

    print("================================================================")
    print("================================================================")
    print(colored("best params: " + str(clf.best_params_)), "magenta")
    print(colored("best scores: " + str(clf.best_score_)), "cyan")
    estimates = clf.predict_proba(X_test)

    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    f1 = f1_score(y_test, y_pred)
    precisionScore = precision_score(y_test, y_pred)
    jaccardScore = jaccard_score(y_test, y_pred)
    recallScore = recall_score(y_test, y_pred)
    rocAucScore =  roc_auc_score(y_test, y_pred)
    
    
    print("--------------------------------")
    print("Accuracy: {:.4%}".format(acc))
    print("R2 : {:.4%}".format(r2))
    print("MSE : {:.4%}".format(mse))
    print("RMSE : {:.4%}".format(rmse))
    print("F1 : {:.4%}".format(f1))
    print("precision_score : {:.4%}".format(precisionScore))
    print("jaccard_score : {:.4%}".format(jaccardScore))
    print("recall_score : {:.4%}".format(recallScore))
    print("roc_auc_score : {:.4%}".format(rocAucScore))
    
    print("--------------------------------\n")
    endTrain = time.perf_counter()
    print("time for {} is {}s ".format(name, endTrain-startTrain))
    print()

    print("================================================================")
    print("================================================================")
    
    test_scores.append((name ,acc, r2, precisionScore, jaccardScore, recallScore, rocAucScore, 
                        clf.best_score_, clf.best_params_))
    
end = time.perf_counter()
print("time {}s ".format(end-start))



print(test_scores)
