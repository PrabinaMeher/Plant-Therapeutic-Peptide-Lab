from Bio.SeqUtils.ProtParam import ProteinAnalysis

def evaluate_peptide(sequence,
                     min_len=8,
                     max_len=60,
                     min_charge=2,
                     max_charge=9,
                     min_hydro=-1,
                     max_hydro=1,
                     min_aromatic=0.05):

    analysis = ProteinAnalysis(sequence)

    length = len(sequence)
    mw = analysis.molecular_weight()
    charge = analysis.charge_at_pH(7)
    aromatic = analysis.aromaticity()
    hydro = analysis.gravy()

    rules = {
        "Length_OK": min_len <= length <= max_len,
        "Charge_OK": min_charge <= charge <= max_charge,
        "Hydrophobicity_OK": min_hydro <= hydro <= max_hydro,
        "Aromatic_OK": aromatic >= min_aromatic
    }

    score = sum(rules.values()) / len(rules)

    return {
        "length": length,
        "mw": mw,
        "charge": charge,
        "hydrophobicity": hydro,
        "aromatic": aromatic,
        "rules": rules,
        "drug_score": score
    }
