# create constant maps
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

# https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/midas/hydrophob.html (kd)
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