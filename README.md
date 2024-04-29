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

## Model Description
- **Source:** VDJdb, IEDB, McPAS
- **Input:** TRB and AA sequence mapped to 30 and 15 spans, respectively, based on AA identity and physicochemical characteristics
- **Transformation:** Z-score normalization and constant/zero-sum column removal based on training data
- **Output:** reported to bind the given pMHC-TCR pair (1) or not bind (0)
- **Modeling Strategy:** 216,401 parameter NN \[\[\[I742-2H\],\[I375-100\]\]-1H-O\] (tensorflow)
- **Validation Strategy:** validation with 100k+ rows of pairs completely uninvolved in training and (in-training testing)

## Customization
### How do I train the model upon my own training dataset of interest?
Please see the notebook "model_v5.ipynb" in the "notebooks" folder for end-to-end process of training your own version of TARPON; a more succint version is in the "model_v5.cross_validation.ipynb" notebook under the same folder. In brief, replace the "hit.csv" (positive controls, binders) and "irr.csv" (negative controls, non-binders) with your own files of interest and run the model either in a single pass (the first notebook mentioned) or in a cross-validated fashion (the second mentioned notebook).
### Where is the paired version of the model?
Public datasets for antigen-resolved TCR sequences with both alpha and beta chains are substantially smaller than antigen-resolved TCR sequences with just CDR3 beta sequences; thus, they are not able to achieve the same level of learning as the current TARPON model. That said, we are happy to provide our version of dual-chain TARPON (TARPON-ab), as presented in the manuscript, in the notebook "revision.r2.4.dual_chain_system.ipynb" under the "notebooks" folder. As with the original TARPON model, all input and output data files are available on the GitHub or upon reasonable request to the authors. We look forward to future iterations of this model as full-featured TCR-Ag datasets become more ubiquitous and approach the scale of current TCRb-Ag datasets.

## References
1. Bisrat J Debebe, Lies Boelen, James C Lee, IAVI Protocol C Investigators, Chloe L Thio, Jacquie Astemborski, Gregory Kirk, Salim I Khakoo, Sharyne M Donfield, James J Goedert, Becca Asquith (2020) Identifying the immune interactions underlying HLA class I disease associations eLife 9:e54558
2. Ma, L. et al. Analyzing the CDR3 Repertoire with respect to TCR—Beta Chain V-D-J and V-J Rearrangements in Peripheral T Cells using HTS. Sci. Rep. 6, 29544; doi: 10.1038/srep29544 (2016).
