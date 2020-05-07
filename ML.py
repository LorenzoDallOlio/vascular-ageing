# -*- coding: utf-8 -*-

# =============================================================================
# ML Analysis
# =============================================================================


# standard libraries
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import SpectralEmbedding
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import json


# %%

# Load db
filename = "cleaned_db.json"  # cleaned db with:
                              # 38 extracted features, 
                              # 4 covariates 
                              # and chronological age

d = json.load(open(filename))
d = pd.DataFrame(d).T
d = d.set_index(np.arange(len(d)))

# quality thresholding
quality_threshold = .01
d = d[np.logical_and(d.quality>=0, d.quality<quality_threshold)]

# remove quality from features
d = d.drop(columns=['quality'])




# %%



age = d.age

# STANDARDIZE DATA
features = d.drop(columns=['age'])

RobustScaler(copy=False).fit_transform(features)


# Age thresholds, young will be in [18,std_thr_y[ 
# while old will be in ]std_thr_o1,to std_thr_02[ 
std_thr_y = 40
std_thr_o1 = 59
std_thr_o2 = 80



# %%

# IMPORTANT, NEED TO RUN THIS CELL BEFORE RUNNING OTHER FILES


# creating stratified train/test datasets
k = np.where(d.age < std_thr_y, 1, np.where(d.age < std_thr_o1, -1, 0))
features, features_test, age, age_test = train_test_split(features, age,
                                                          stratify=k,
                                                          test_size=0.25,
                                                          shuffle=True,
                                                          random_state=42)

# indexes are needed to keep track of patients during slicing that happens in "prepare_db_for_DL.py"
train_idx = features.index
train_idx = train_idx[np.logical_or(age[train_idx]<std_thr_y,
                                    np.logical_and(age[train_idx]>std_thr_o1, age[train_idx]<std_thr_o2))]


test_idx = features_test.index
test_idx = test_idx[np.logical_or(age_test[test_idx]<std_thr_y,
                                    np.logical_and(age_test[test_idx]>std_thr_o1, age_test[test_idx]<std_thr_o2))]





# %%



# BINARIZE AGE and remove middle age samples
# 0 == young, 1 == old

bin_features = features[np.logical_or(age<std_thr_y,np.logical_and(age>std_thr_o1,
                                                               age<std_thr_o2))]
bin_age = age[np.logical_or(age<std_thr_y,np.logical_and(age>std_thr_o1,
                                                               age<std_thr_o2))]
bin_age = np.where(bin_age>std_thr_o1, 0, 1)


bin_features_test = features_test[np.logical_or(age_test<std_thr_y, np.logical_and(age_test>std_thr_o1,
                                                               age_test<std_thr_o2))]
bin_age_test = age_test[np.logical_or(age_test<std_thr_y, np.logical_and(age_test>std_thr_o1,
                                                               age_test<std_thr_o2))]
bin_age_test = np.where(bin_age_test>std_thr_o1, 0, 1)




# %%


n_regressions = 100  # it means 100 linear + 100 logistic

# FIND MOST RELEVANT FEATURES


# =============================================================================
# LNEAR REGRESSION
# =============================================================================


rankingr = np.array([np.asarray(features.columns.values),
                     np.zeros(len(features.columns.values))]).T
intercepts_r = []
best_alpha = []


for _ in range(n_regressions):
  print(_)
  X_train, X_test, y_train, y_test = train_test_split(features, age,
                                                      test_size=0.33,
                                                      shuffle=True,
                                                      random_state=_)

  # RIDGE CV
  alphs = 10**np.linspace(-1, 3, 9)
  ridge = RidgeCV(alphas=alphs, cv=3)
  ridge.fit(X_train, y_train)

  names = features.columns.values
  all_params = []

  for _ in range(len(names)):
    all_params.append([names[_], ridge.coef_[_]])
  
  
  intercepts_r.append(ridge.intercept_)
  best_alpha.append(ridge.alpha_)

  all_params = np.asarray(all_params).T
  values = np.asarray(all_params[1], dtype=float)
  order = np.argsort(np.abs(values))

  for i, p in zip(order, range(len(order))):
    rankingr[i][1] += p

final_pointsr = np.array(len(rankingr)-np.asarray(rankingr.T[1])/n_regressions)




# %%



# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================


rankingl = np.array([np.asarray(features.columns.values),
                     np.zeros(len(features.columns.values))]).T

intercepts_l = []
best_C = []


C_s = np.logspace(-2, 5, 8)

for _ in range(n_regressions):
  print(_)
  X_train, X_test, y_train, y_test = train_test_split(bin_features, bin_age,
                                                      test_size=0.33,
                                                      shuffle=True,
                                                      random_state=_)


  class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
  class_w = {np.unique(y_train)[0]:class_weights[0],
             np.unique(y_train)[1]:class_weights[1]}

  logit = LogisticRegressionCV(solver='saga',
                               max_iter=10000, 
                               random_state=_, 
                               Cs=C_s,
                               cv=3,
                               class_weight=class_w)
  logit.fit(X_train, y_train)

  names = features.columns.values
  all_params = []

  for _ in range(len(names)):
    all_params.append([names[_], logit.coef_[0][_]])

  intercepts_l.append(logit.intercept_)
  best_C.append(logit.C_)

  all_params = np.asarray(all_params).T
  values = np.asarray(all_params[1], dtype=float)
  order = np.argsort(np.abs(values))

  for i, p in zip(order, range(len(order))):
    rankingl[i][1] += p

final_pointsl = np.array(len(rankingl)-np.asarray(rankingl.T[1])/n_regressions)
final_points = final_pointsl + final_pointsr


# %%


# =============================================================================
# OBTAIN FINAL SCORE
# =============================================================================

final_pointsl = final_pointsl[np.argsort(final_points)]
final_pointsr = final_pointsr[np.argsort(final_points)]

final_rank = np.array(rankingl.T[0][np.argsort(final_points)])
final_points = np.sort(final_points)





# %%

# =============================================================================
# ALL the following code cells are related to plots and ML/DL comparison
# =============================================================================


# Creating Spectral Embedding components fro further plots

# 2 components
features_SE2 = SpectralEmbedding(n_components=2).fit_transform(bin_features)
se2_1 = features_SE2.T[0]
se2_2 = features_SE2.T[1]


# 3 components 
features_SE3 = SpectralEmbedding(n_components=3).fit_transform(bin_features)
se3_1 = features_SE3.T[0]
se3_2 = features_SE3.T[1]
se3_3 = features_SE3.T[2]



# %%

# 2-D PLOT:  component vs component
fig, ax = plt.subplots()
a = ax.scatter(se2_1*1000, se2_2*1000, s=5, alpha=1, c=bin_age, cmap='jet_r')
cb = fig.colorbar(a)
cb.set_label("chronological age (years)")
ax.grid()
ax.set_ylabel("2nd component (a.u.)")
ax.set_xlabel("1st component (a.u.)")


# %%

# 3-D PLOT: 3 components
fig, ax = plt.subplots()
ax = Axes3D(fig)
a = ax.scatter(se3_1*1e3, se3_2*1e3, se3_3*1e3, s=5, c=bin_age, cmap='jet_r')

ax.set_xlabel("1st component (a.u.)")
ax.set_ylabel("2nd component (a.u.)")
ax.set_zlabel("3rd component (a.u.)")
cbar = fig.colorbar(a)
cbar.set_label("chronological age (years)")
fig.show()


# %%


# SVM hyperparameter's tuning (gamma and C) with CV on train set

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cov = ['length', 'weight', 'sex', 'smoke']
saved = []
par = []
f = ['tpr', 'a'] # + cov  # uncomment the " + cov" to add covariates


min_par1 = -3
min_par2 = 0.5

max_par1 = -1
max_par2 = 1.5

num1 = 13
num2 = 13

k = num1*num2+1

for i in 10**np.linspace(min_par1, max_par1, num=num1):
    for j in 10**np.linspace(min_par2, max_par2, num=num2):
        scores = []

        classif = SVC(gamma=i,
                      C=j, 
                      random_state=42)

        # REMOVE "[f]" if you want to use all the variables in the db
        new_bin_features = bin_features[f]        
        
        for train_index, test_index in kf.split(new_bin_features, bin_age):
          classif.fit(new_bin_features.iloc[train_index], bin_age[train_index])
        
          scores.append(classif.score(new_bin_features.iloc[test_index], bin_age[test_index]))
        
        saved.append(np.mean(scores))
        par.append([i, j])
        k-=1
        print("\nmissing iterations: ", k)
        

print(max(saved), "  gamma = ", str(par[np.argmax(saved)][0]), "  C = ", str(par[np.argmax(saved)][1]), "  for ", f)



# %%

# You need this cell only if you want to compare performances of
# Machine Learning approach (ML) and Deep Learning apporach (DL)

# FIRST YOU NEED TO RUN "prepare_db_for_DL.py" 
# AND "DL.py" FILES IN ORDER TO RUN THIS CELL
# N.W you need to keep the variables saved in the running enviroment


separate_guys.append(len(y_predicted))
NN_probs = []

for i,j in zip(separate_guys, separate_guys[1:]):
    
    NN_probs.append(np.mean(y_predicted[i:j], axis=0))

NN_probs = np.asarray(NN_probs)


y_pred_CNN = np.argmax(NN_probs, axis=1)




# %%

# Cell needed only if you want to plot more AUC curves together

labls = []
lines = []


# %%





# UNCOMMENT ONE OF THE FOLLOWING CLASSIFIERS IN ORDER TO CHECK ITS AUC

# classifier = SVC(gamma=0.01, C=12.11528, probability=True, random_state=42)  # ALL
# classifier = 'CNN'
classifier = SVC(gamma=0.0051, C=695, probability=True, random_state=42)  # tpr+a+cov
# classifier = SVC(gamma=9, C=1, probability=True, random_state=42)  # tpr+a
# classifier = SVC(gamma=42, C=1, probability=True, random_state=42)  # cov

# classifier = SVC(gamma=0.01145, C=657.93322, probability=True, random_state=42)  # ac_slope+tpr+cov
# classifier = SVC(gamma=77, C=.2, probability=True, random_state=42)  # ac_slope
# classifier = SVC(gamma=60, C=.7, probability=True, random_state=42)  # a
# classifier = SVC(gamma=3.2, C=.4, probability=True, random_state=42)  # tpr
# classifier = SVC(gamma=0.03162, C=1.77828, probability=True, random_state=42)  # ibi
# classifier = SVC(gamma=21.54435, C=.1, probability=True, random_state=42)  # pnn20



cov = ['length', 'weight', 'sex', 'smoke']
f = ['tpr', 'a']# + cov  # uncomment " + cov" to add covariates


# REMOVE "[f]" if you want to use all the variables in the db
X_train, y_train = np.asarray(bin_features[f]), bin_age
X_test, y_test = np.asarray(bin_features_test[f]), bin_age_test



if classifier=='CNN':
    probas_ = NN_probs
else:
    probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)


# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])

roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2, alpha=0.8,
          label='roc curve (AUC = %0.3f)' % (roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          alpha=.99)


plt.rc('font', size=15)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()



# %%

# RUN this cell if you want to include last plotted 
# roc curve in the final AUCs comparison plot

labls.append(r'39 (AUC = %0.3f)' % (roc_auc))
lines.append(fpr)
lines.append(tpr)


# %%

# RUN this cell only when you want to see 
# the final AUCs comparison plot

for i in range(len(labls)):
  if i==1:
      plt.plot(lines[2*i], lines[2*i+1], 'k-.', label=labls[i], lw=2, alpha=.8)
  else:
      plt.plot(lines[2*i], lines[2*i+1], label=labls[i], lw=2, alpha=.8)


plt.rc('font', size=15)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         alpha=.99)
plt.xlim([-0.001, 1.001])
plt.ylim([-0.001, 1.001])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=20)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=20)

plt.legend(loc="lower right", fontsize=20)
plt.grid()
plt.show()
