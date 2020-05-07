# -*- coding: utf-8 -*-

# ======================================================================================
# preparing db for DL approach, slicing every signal in sequence of 15 consecutive peaks
# ======================================================================================


# cardio stuff
# you can recover this function from https://github.com/Nico-Curti/cardio
# it simply finds the x coordinate of all the minima in y it can detect
from double_gaussian_features import find_x_of_minima

# standard libraries
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import pickle
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
d = d['age', 'quality', 'signal', 'time']



# %%


d = d[np.logical_and(d.quality>=0, d.quality<.01)]
age = d.age
d = d[np.logical_or(age<40, np.logical_and(age<80, age>59))]

# %%

final_peaks = []
final_labels = []
separate_guys = []

# if you want to recover which patient are used as test set in "ML.py"
# replace "train_idx" with "test_idx"
# To obtain train_idx and test_idx you first need to run "ML.py" cell 4
for guy in train_idx:
  print(guy)
  separate_guys.append(len(final_peaks))
  sample = d.loc[guy]  # use this when using test_idx
  splits = find_x_of_minima(sample.time, sample.signal)
  peaks = np.split(sample.signal, splits[5::15])[1:-1]
  times = np.split(sample.time, splits[5::15])[1:-1]
  labels = sample.drop(['signal', 'time'])

  try:
    new_peaks = list(map(lambda tx, sy: interp1d(tx, sy, kind="cubic")(np.linspace(min(tx), max(tx), 1024)), times, peaks))
    final_peaks = final_peaks + new_peaks
    final_labels = final_labels + [labels]*len(new_peaks)
  except ValueError:
    pass
print("ended")
DCNN_db = {"samples": final_peaks, "labels": final_labels}


# %%

print("updating db")
with open('cleaned_db_for_DL_15p.pickle', 'wb') as file:
    pickle.dump(DCNN_db, file)