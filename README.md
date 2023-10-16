![Logo of the TARPON tool with TARPON written in big block letters colored in a blue to black gradient and a fish in the "O"](logo.png)
## TARPON – TCR-Antigen Relations Predicted Over a Neural-Network
TARPON predicts TCR-Antigen binding via a 0-1 score, where 1 indicates strong confidence of binding, for variable length TCR beta chain CDR3 sequences (10-22 ideal, maximum of 30) and variable length antigen peptide sequences (8-11 ideal, maximum of 15). It utilizes a parsimonious neural network structure that produces separate learned TCR and Antigen embeddings then combines them via a learned rule set that is then softmaxed into a probability-like score.

## Usage Instructions
1. Download the packages listed in `requirements.txt`, these are pandas, numpy and tensorflow.
2. Assemble your input CSV with the first column as CDR3 sequences from TCR beta chains, and second column as antigen sequences. 
3. Run TARPON! For example, with an input CSV named "example.csv", provided in the TARPON directory, and an output CSV called "out.csv" we would run the following: `python fishing.py --bait example.csv --catch out.csv --verbose` and receive the following output:
```
preparing bait...
        reading input TCR-Ag pairs...done!
        reading n-processors...done!
done!
casting line...
        encoding TCRs...done!
        encoding antigens...done!
done!
reeling in...
        loading model...done!
        normalizing data...done!
        predicting data...done!
done!
check your catch! TARPON was successful and saved your catch at "out.csv"
         /\
o      _/./
 o  ,-'    `-:..-'/
  o: o )      _  (
   "`-....,--; `-.\
       `
```
You can remove the `--verbose` tag to get more succint outputs, best of luck with the predictions and good luck fishing!!!

## Data Selection
pMHC-TCRb pairs are derived from VDJdb, IEDB, and McPAS. High confidence curated and productive TCR pairs were utilized to peptides that are bound to HLA-A\*02:01, due to its standing as the most prevalent class-I HLA. Peptides are utilized that are 8-11 amino acids (AAs) in length because of literature evidence demonstrating that in vast majority only these peptides lengths can load onto classical class-I HLAs ([1](https://elifesciences.org/articles/54558)). Only CDR3s between 10-22 AAs in length are utilized, similarly due to literature evidence that this is the typical distribution length of CDR3 regions for TCRb chains ([2](https://nature.com/articles/srep29544)). No other data was filtered to ensure strong replicability and accessibility of the model.

## Model Descriptions
### Model V5 (Final)
- **Source:** VDJdb, IEDB, McPAS
- **Input:** TRB and AA sequence mapped to 30 and 15 spans, respectively, based on AA identity and physicochemical characteristics
- **Transformation:** Z-score normalization and constant/zero-sum column removal based on training data
- **Output:** reported to bind the given pMHC-TCR pair (1) or not bind (0)
- **Modeling Strategy:** 216,401 parameter NN \[\[\[I742-2H\],\[I375-100\]\]-1H-O\] (tensorflow)
- **Validation Strategy:** validation with 100k+ rows of pairs completely uninvolved in training and (in-training testing)

### Model V4
- **Source:** VDJdb, IEDB, McPAS
- **Input:** TRB and AA sequence mapped to 100 percentiles based on AA identity and physicochemical characteristics
- **Transformation:** Z-score normalization and constant/zero-sum column removal based on training data
- **Output:** reported to bind the given pMHC-TCR pair (1) or not bind (0)
- **Modeling Strategy:** 2,973,001 parameter NN \[I-1K-5H-O\] (tensorflow)
- **Validation Strategy:** train test split utilizing 25% testing ratio

### Model V3
- **Source:** VDJdb, IEDB, McPAS
- **Input:** TRB and AA sequence mapped to 100 percentiles based on AA identity and physicochemical characteristics
- **Transformation:** Z-score normalization and constant/zero-sum column removal based on training data
- **Output:** reported to bind the given pMHC-TCR pair (1) or not bind (0)
- **Modeling Strategy:** MLP with a hidden layer of 100 nodes (sklearn)
- **Validation Strategy:** train test split utilizing 25% testing ratio

### Model V2
- **Source:** VDJdb, IEDB
- **Input:** TRB sequence mapped to 100 percentiles based on AA identity and physicochemical characteristics
- **Transformation:** Z-score normalization and constant/zero-sum column removal based on training data
- **Output:** reported to bind CMV pp65 antigen NLVPMVATV (1) or not bind (0)
- **Modeling Strategy:** logit (sklearn), random forest (sklearn)
- **Validation Strategy:** 1000-fold cross-validation

### Model V1
- **Source:** VDJdb (Human, Confidence ≥ 2, HLA-A\*02:01)
- **Input:** TRB sequence mapped to 100 percentiles based on AA identity and physicochemical characteristics
- **Transformation:** Z-score normalization and constant/zero-sum column removal based on training data
- **Output:** reported to bind CMV pp65 antigen NLVPMVATV (1) or not bind (0)
- **Modeling Strategy:** logit (sklearn), random forest (sklearn)
- **Validation Strategy:** 1000-fold cross-validation

## References
1. Bisrat J Debebe, Lies Boelen, James C Lee, IAVI Protocol C Investigators, Chloe L Thio, Jacquie Astemborski, Gregory Kirk, Salim I Khakoo, Sharyne M Donfield, James J Goedert, Becca Asquith (2020) Identifying the immune interactions underlying HLA class I disease associations eLife 9:e54558
2. Ma, L. et al. Analyzing the CDR3 Repertoire with respect to TCR—Beta Chain V-D-J and V-J Rearrangements in Peripheral T Cells using HTS. Sci. Rep. 6, 29544; doi: 10.1038/srep29544 (2016).