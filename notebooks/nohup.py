import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# tmp
df = pd.read_csv('../outputs/model_v3/full.input.X.csv', index_col=0)
counts = df['CDR3'].value_counts()

from Levenshtein import distance
from tqdm import tqdm
# our goal here is to find the similar and different peptides in our dataset
df_l = pd.DataFrame(columns=['pep1','pep2','d'])
for idx, pep1 in tqdm(enumerate(counts.index[:-1]), total=counts.shape[0]-1):
    for pep2 in counts.index[idx+1:]:
        d = distance(pep1, pep2)
        df_l.loc[df_l.shape[0]] = pep1, pep2, d

# write down this table
df_l.to_csv('../outputs/model_v3/full.cdr3.levenshtein.tab.csv')