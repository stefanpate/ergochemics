from ergochemics.similarity import MorganFingerprinter, MolFeaturizer, ReactionFeaturizer
import numpy as np
from rdkit import Chem

sma1 = "NC(=O)C1=CN(C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(OP(=O)(O)O)C3O)C(O)C2O)C=CC1.O=C(O)CCC(=O)C(=O)O"
sma2 = "NC(=O)C1=CN(C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(OP(=O)(O)O)C3O)C(O)C2O)C=CC1.O=C(O)CCC(=O)C(=O)O"
rc1 = [3, 47, 46, 45, 5, 4, 50, 49]
rc2 = [3, 4, 5, 45, 46, 47, 49, 50]


def test_molecule_fingerprinter_without_rc():
    mfper = MorganFingerprinter(
        radius=2,
        length=2048,
        mol_featurizer=MolFeaturizer(),
    )
    fp1 = mfper.fingerprint(Chem.MolFromSmiles(sma1))
    fp2 = mfper.fingerprint(Chem.MolFromSmiles(sma2))
    assert np.allclose(fp1, fp2)  # Should be True if fingerprints are equal

def test_molecule_fingerprinter_scramble_rc_order():
    mfper = MorganFingerprinter(
        radius=2,
        length=2048,
        mol_featurizer=MolFeaturizer(),
    )
    fp1 = mfper.fingerprint(Chem.MolFromSmiles(sma1), rc1, rc_dist_ub=1)
    fp2 = mfper.fingerprint(Chem.MolFromSmiles(sma2), rc2, rc_dist_ub=1)
    assert np.allclose(fp1, fp2)  # Should be True if fingerprints are equal

sma1 = "CCO.C(=O)O"
sma2 = "C(=O)O.CCO"
rc1 = [1, 2, 3, 4]
rc2 = [0, 1, 4, 5]

def test_molecule_fingerprinter_scramble_smiles_order():
    mfper = MorganFingerprinter(
        radius=2,
        length=2048,
        mol_featurizer=MolFeaturizer(),
    )
    fp1 = mfper.fingerprint(Chem.MolFromSmiles(sma1), rc1)
    fp2 = mfper.fingerprint(Chem.MolFromSmiles(sma2), rc2)
    assert np.allclose(fp1, fp2)  # Should be True if fingerprints are equal

am_rxn = '[NH2:2][CH:1]([CH2:3][OH:4])[C:5](=[O:7])[OH:6]>>[NH2:2][CH2:1][CH2:3][OH:4].[O:6]=[C:5]=[O:7]'
rxn = 'NC(CO)C(=O)O>>NCCO.O=C=O'

def test_reaction_fingerprinter():
    rfper = ReactionFeaturizer(
        radius=2,
        length=2048,
        mol_featurizer=MolFeaturizer(),
    )

    # Both should return fingerprints, but they should be different
    fp_with_rc = rfper.fingerprint(am_rxn, use_rc=True)
    fp_without_rc = rfper.fingerprint(rxn, use_rc=False)
    assert not np.allclose(fp_with_rc, fp_without_rc)