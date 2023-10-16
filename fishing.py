# silence any tf messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from tensorflow import keras
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd


## DEFINE CONSTANTS
# > biochemical and biophysical
amino_acid_charge = {
    'A': 0,  # Alanine
    'R': 1,  # Arginine
    'N': 0,  # Asparagine
    'D': -1,  # Aspartic Acid
    'C': 0,  # Cysteine
    'E': -1,  # Glutamic Acid
    'Q': 0,  # Glutamine
    'G': 0,  # Glycine
    'H': 0,  # Histidine
    'I': 0,  # Isoleucine
    'L': 0,  # Leucine
    'K': 1,  # Lysine
    'M': 0,  # Methionine
    'F': 0,  # Phenylalanine
    'P': 0,  # Proline
    'S': 0,  # Serine
    'T': 0,  # Threonine
    'W': 0,  # Tryptophan
    'Y': 0,  # Tyrosine
    'V': 0,  # Valine
}
amino_acid_hydrophobicity = {
    'A': 1.8,  # Alanine
    'R': -4.5,  # Arginine
    'N': -3.5,  # Asparagine
    'D': -3.5,  # Aspartic Acid
    'C': 2.5,  # Cysteine
    'E': -3.5,  # Glutamic Acid
    'Q': -3.5,  # Glutamine
    'G': -0.4,  # Glycine
    'H': -3.2,  # Histidine
    'I': 4.5,  # Isoleucine
    'L': 3.8,  # Leucine
    'K': -3.9,  # Lysine
    'M': 1.9,  # Methionine
    'F': 2.8,  # Phenylalanine
    'P': -1.6,  # Proline
    'S': -0.8,  # Serine
    'T': -0.7,  # Threonine
    'W': -0.9,  # Tryptophan
    'Y': -1.3,  # Tyrosine
    'V': 4.2,  # Valine
}
amino_acid_weights = {
    'A': 89.09,   # Alanine
    'R': 174.20,  # Arginine
    'N': 132.12,  # Asparagine
    'D': 133.10,  # Aspartic Acid
    'C': 121.16,  # Cysteine
    'E': 147.13,  # Glutamic Acid
    'Q': 146.15,  # Glutamine
    'G': 75.07,   # Glycine
    'H': 155.16,  # Histidine
    'I': 131.18,  # Isoleucine
    'L': 131.18,  # Leucine
    'K': 146.19,  # Lysine
    'M': 149.21,  # Methionine
    'F': 165.19,  # Phenylalanine
    'P': 115.13,  # Proline
    'S': 105.09,  # Serine
    'T': 119.12,  # Threonine
    'W': 204.23,  # Tryptophan
    'Y': 181.19,  # Tyrosine
    'V': 117.15,  # Valine
}
has_sulfur = ['C','M']
is_aromatic = ['F','Y','W']
# > encoding
stretch_length_cdr3 = 30
stretch_length_ag = 15


## DEFINE HELPER FUNCTIONS
def retrieve_ncpus(n_cpus):
    # define the number of processors to use
    try:
        # attempt to cast as int
        n_cpus = int(n_cpus)
        # if -1 then use all available
        if n_cpus == -1:
            n_cpus = mp.cpu_count()
        return n_cpus
    except:
        # if not castable then throw error
        raise ValueError('argument "--processors" cannot be interpreted as an integer, please try again thank you!')

def retrieve_bait(bait):
    # check if the filename is available
    if not os.path.exists(bait):
        raise ValueError('argument "--bait" does not point to an actual file, please double check thank you!')
    # read in the file
    # > let's assume there is no index columns
    df = pd.read_csv(bait)
    # > if that has too many columns then assume there is an index column
    if df.shape[1] >= 2:
        df = pd.read_csv(bait, index_col=0)
    # check if shape is alright
    if df.shape[0] == 0:
        raise ValueError('argument "--bait" does not appear to have any data, please check thank you!')
    if df.shape[1] <= 1:
        raise ValueError('argument "--bait" does not have enough columns, please check it has one for CDR3 one for Antigen')
    if df.shape[1] >= 3:
        raise ValueError('argument "--bait" appears to have too many columns even when accounting for indexes, please check it is correct thank you!')
    # if it appears correct continue
    return df
    

## DEFINE ENCODERS
def _encode_cdr3s(peptide):
    # define columns
    columns = cdr3_alphabet+['charge','hydrophobicity','weight','sulfur','aromatic']
    columns_returned = [f'{idx}{col}' for idx in range(stretch_length_cdr3) for col in columns]
    # create the tracking dataframe
    X = pd.DataFrame(np.nan, index=range(stretch_length_cdr3), columns=columns)
    step = stretch_length_cdr3 / len(peptide)
    for idx, aa in enumerate(peptide):
        # find start and end of each peptide
        start, end = round(idx * step), round((idx + 1) * step)
        if idx == len(peptide) - 1:
            end = stretch_length_cdr3
        # map to the coordinates, weight is scaled by 0.01 to reduce to single digit values, same with hydrop with 0.5
        charge = amino_acid_charge[aa]
        hydrophobicity = amino_acid_hydrophobicity[aa] / 2
        weight = amino_acid_weights[aa] / stretch_length_cdr3
        sulfur = 1 * (aa in has_sulfur)
        aromatic = 1 * (aa in is_aromatic)
        X.loc[start:end, cdr3_alphabet] = 0
        X.loc[start:end, aa] = 1
        X.loc[start:end, ['charge','hydrophobicity','weight','sulfur','aromatic']] = charge, hydrophobicity, weight, sulfur, aromatic
    assert not X.isna().any().any()
    return pd.Series(X.values.flatten(), index=columns_returned, name=peptide)
def encode_cdr3s(cdr3s, n_cpus, verbose):
    if verbose: print('\n\tencoding TCRs...', end='')
    # the first step is to create a conversion map for CDR3 sequences
    global cdr3_alphabet
    cdr3_alphabet = sorted(set([el for x in cdr3s for el in list(x)]))
    # work through cdr3s
    els = []
    with mp.Pool(n_cpus) as pool:
        for el in pool.imap_unordered(_encode_cdr3s, cdr3s):
            els.append(el)
    cdr3_to_X = pd.concat(els, axis=1).T
    if verbose: print('done!')
    return cdr3_to_X


def _encode_ags(peptide):
    # define columns
    columns = ag_alphabet+['charge','hydrophobicity','weight','sulfur','aromatic']
    columns_returned = [f'{idx}{col}' for idx in range(stretch_length_ag) for col in columns]
    # create the tracking dataframe
    X = pd.DataFrame(np.nan, index=range(stretch_length_ag), columns=columns)
    step = stretch_length_ag / len(peptide)
    for idx, aa in enumerate(peptide):
        # find start and end of each peptide
        start, end = round(idx * step), round((idx + 1) * step)
        if idx == len(peptide) - 1:
            end = stretch_length_ag
        # map to the coordinates, weight is scaled by 0.01 to reduce to single digit values, same with hydrop with 0.5
        charge = amino_acid_charge[aa]
        hydrophobicity = amino_acid_hydrophobicity[aa] / 2
        weight = amino_acid_weights[aa] / stretch_length_ag
        sulfur = 1 * (aa in has_sulfur)
        aromatic = 1 * (aa in is_aromatic)
        X.loc[start:end, ag_alphabet] = 0
        X.loc[start:end, aa] = 1
        X.loc[start:end, ['charge','hydrophobicity','weight','sulfur','aromatic']] = charge, hydrophobicity, weight, sulfur, aromatic
    assert not X.isna().any().any()
    return pd.Series(X.values.flatten(), index=columns_returned, name=peptide)
def encode_ags(ags, n_cpus, verbose):
    if verbose: print('\tencoding antigens...', end='')
    # the first step is to create a conversion map for ag sequences
    global ag_alphabet
    ag_alphabet = sorted(set([el for x in ags for el in list(x)]))
    # work through antigens
    els = []
    with mp.Pool(n_cpus) as pool:
        for el in pool.imap_unordered(_encode_ags, ags):
            els.append(el)
    ag_to_X = pd.concat(els, axis=1).T
    if verbose: print('done!')
    return ag_to_X


## DEFINE PREDICTION
def predict(X_cdr3s, X_ags, n_cpus, verbose):
    # load in the model
    if verbose: print('\n\tloading model...', end='')
    model = keras.models.load_model('outputs/model_v5/model_v5.keras')
    if verbose: print('done!')

    # combine the cdr3 and antigens
    if verbose: print('\tnormalizing data...', end='')
    X_cdr3s.columns = 'cdr3:' + X_cdr3s.columns
    X_ags.columns = 'ag:' + X_ags.columns
    X = X_cdr3s.join(X_ags)

    # read in normalization factors
    means = pd.read_csv('outputs/model_v5/model_v5.means.csv', index_col=0).iloc[:, 0]
    stds = pd.read_csv('outputs/model_v5/model_v5.stds.csv', index_col=0).iloc[:, 0]
    # subset for relevant columns
    X = X.T.reindex(means.index).fillna(0).T
    # normalize
    X -= means
    X /= stds
    if verbose: print('done!')

    # retrieve the appropriate columns
    if verbose: print('\tpredicting data...', end='')
    cols_cdr3 = X.columns[X.columns.str.startswith('cdr3')]
    cols_ag = X.columns[X.columns.str.startswith('ag')]
    # predict the binding
    pred = model.predict([X[cols_cdr3], X[cols_ag]], workers=n_cpus, use_multiprocessing=n_cpus > 1, verbose=0)
    pred = pd.Series(pred[:, 0]).reset_index()
    if verbose: print('done!')
    return pred


## DEFINE MAIN
if __name__ == '__main__':
    # define argument parser
    parser = argparse.ArgumentParser(
                    prog='TARPON',
                    description='"fishes" for TCR-Ag binders in a given "bait" set of TCR-Ag pairs')
    parser.add_argument('-b', '--bait', required=True, help='input CSV filename with first column for CDR3s and second for Ags')
    parser.add_argument('-c', '--catch', required=True, help='output CSV filename to store predictions, will be written in same row order as inputted bait')
    parser.add_argument('-p', '--processors', required=False, help='number of processors to use (-1 for all)', default=-1)
    parser.add_argument('-v', '--verbose', action='store_true', help='whether to be verbose, helps with debugging')
    
    # parse arguments
    args = parser.parse_args()
    print('preparing bait...', end='')
    verbose = args.verbose
    if verbose: print('\n\treading input TCR-Ag pairs...', end='')
    bait = retrieve_bait(args.bait)
    if verbose: print('done!')
    if verbose: print('\treading n-processors...', end='')
    n_cpus = retrieve_ncpus(args.processors)
    if verbose: print('done!')
    print('done!')
    
    # encode tcrs and ags
    print('casting line...', end='')
    cdr3s = bait.iloc[:, 0].unique()
    ags = bait.iloc[:, 1].unique()
    cdr3_to_X = encode_cdr3s(cdr3s, n_cpus, verbose)
    ag_to_X = encode_ags(ags, n_cpus, verbose)
    print('done!')
    
    # predict final data
    print('reeling in...', end='')
    X_cdr3s = cdr3_to_X.loc[cdr3s].reset_index().iloc[:, 1:]
    X_ags = ag_to_X.loc[ags].reset_index().iloc[:, 1:]
    pred = predict(X_cdr3s, X_ags, n_cpus, verbose)
    print('done!')
    
    # write the data
    pred.to_csv(args.catch)
    print('check your catch! TARPON was successful and saved your catch at "' + args.catch + '"')
    print('''\
         /\\
o      _/./
 o  ,-'    `-:..-'/
  o: o )      _  (
   "`-....,--; `-.\\
       `''')