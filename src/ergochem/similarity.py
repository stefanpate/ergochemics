from rdkit import Chem
from rdkit.Chem import rdFMCS
from copy import deepcopy
import re
from typing import Iterable
from itertools import chain

# TODO: test
def rcmcs_score(
        rxn1: str,
        rxn2: str,
        rcs1: Iterable[Iterable[int]],
        rcs2: Iterable[Iterable[int]],
        patts: Iterable[str],
        norm: str ='max',
        average: str = 'weighted',
        enforce_ring_membership: bool = False
    ) -> float:
    '''
    Calculates the reaction center max common substructure score
    between two reactions. Accepts reactions in form 'A.B>>>C.D' 
    or iterable of reactant / product SMILES. Order of reactants / products
    in all reactions, reaction centers, and reaction center patterns must match.
    
    Args
    ----
    rxn1:str | Iterable[str]
        Reaction 1 in form 'A.B>>>C.D' or iterable of reactant / product SMILES
    rxn2:str | Iterable[str]
        Reaction 2 in form 'A.B>>>C.D' or iterable of reactant / product SMILES
    rcs1:Iterable[Iterable[int]]
        Reaction center atom indices for reaction 1
    rcs2:Iterable[Iterable[int]]
        Reaction center atom indices for reaction 2
    patts:Iterable[str]
        SMARTS patterns of reaction center fragments in order they appear in
        rxn1 and rxn2 / rcs1 and rcs2
    norm:str
        Normalization method for rcmcs score. If 'max', score is divided by
        # of atoms in the larger reactant. If 'min', score is divided by
        # of atoms in the smaller reactant.
    average:str
        How to average the rcmcs scores of the reactants. If 'weighted',
        the average is weighted by the number of atoms in each reactant
        / product. If 'simple', the average is the arithmetic mean.

    Returns
    -------
    rcmcs_score:float
        Reaction center max common subgraph score [0, 1]
    '''
    if norm not in ['max', 'min']:
        raise ValueError("Normalization must be 'max' or 'min'")
    
    if average not in ['weighted', 'simple']:
        raise ValueError("Average must be 'weighted' or 'simple'")
    
    if ">>" in rxn1:
        rxn1 = list(chain(*[side.split('.') for side in rxn1.split(">>")]))

    if ">>" in rxn2:
        rxn2 = list(chain(*[side.split('.') for side in rxn2.split(">>")]))

    len_set = set()
    len_set.add(len(rcs1))
    len_set.add(len(rcs2))
    len_set.add(len(patts))
    len_set.add(len(rxn1))
    len_set.add(len(rxn2))

    if len(len_set) != 1:
        raise ValueError("Number of reactants / products, reaction centers, and reaction center patterns do not match")
    
    n_atoms = 0
    rcmcs = 0
    n_mols = 0
    for smi1, smi2, rc1, rc2, patt in zip(rxn1, rxn2, rcs1, rcs2, patts):
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        rcmcs_i = molecule_rcmcs_score([mol1, mol2], [rc1, rc2], patt, norm=norm, enforce_ring_membership=enforce_ring_membership)

        n1 = mol1.GetNumAtoms()
        n2 = mol2.GetNumAtoms()

        if norm == 'max':
            atoms_i = max(n1, n2)
        elif norm == 'min':
            atoms_i = min(n1, n2)

        if average == 'weighted':
            rcmcs += rcmcs_i * atoms_i
        elif average == 'simple':
            rcmcs += rcmcs_i
        
        n_mols += 1
        n_atoms += atoms_i

    if average == 'weighted':
        return rcmcs / n_atoms
    elif average == 'simple':
        return rcmcs / n_mols

def molecule_rcmcs_score(mols: list[Chem.Mol], rcs: list[tuple[int]], patt:str, norm='max', enforce_ring_membership: bool = False):
    '''
    Args
    ----
    mols: list[Chem.Mol]
        Molecules
    rcs: list[tuple[int]]
        Reaction center atom indices
    patt:str
        Reaction center substructure pattern in SMARTS
    enforce_ring_membership:bool
        Whether to enforce that ring atoms can only match ring atoms

    Returns
    -------
    rcmcs:float
        Reaction center max common substructure score [0, 1]
    '''

    res = molecule_rcmcs(mols, rcs, patt, enforce_ring_membership)

    if res is None:
        return 0.0
    elif res.canceled:
        return 0
    elif norm == 'min':
        return res.numAtoms / min(m.GetNumAtoms() for m in mols)
    elif norm == 'max':
        return res.numAtoms / max(m.GetNumAtoms() for m in mols)

def molecule_rcmcs(mols: list[Chem.Mol], rcs: list[tuple[int]], patt:str, enforce_ring_membership: bool = False):
    '''
    Args
    ----
    mols: list[Chem.Mol]
        Molecules
    rcs: list[tuple[int]]
        Reaction center atom indices
    patt:str
        Reaction center substructure pattern in SMARTS
    enforce_ring_membership:bool
        Whether to enforce that ring atoms can only match ring atoms

    Returns
    -------
    res | None
        FindMCS output or None if failed or failed pre-check
    '''
    if len(mols) != len(rcs):
        raise ValueError("Number of molecules and reaction centers do not match")
    
    mols = [deepcopy(m) for m in mols]

    rc_scalar = 100

    def _replace(match):
        atomic_number = int(match.group(1))
        return f"[{atomic_number * rc_scalar}#{atomic_number}"

    def _reset(match):
        atomic_number = int(match.group(1))
        
        if atomic_number % rc_scalar == 0:
            return f"[{int(atomic_number / rc_scalar)}"
        else:
            return f"[{atomic_number}"

    patt = re.sub(r'\[#(\d+)', _replace, patt) # Mark reaction center patt w/ isotope number

    # Mark reaction center vs other atoms in substrates w/ isotope number
    for mol, rc in zip(mols, rcs):
        for atom in mol.GetAtoms():
            if atom.GetIdx() in rc:
                atom.SetIsotope(atom.GetAtomicNum() * rc_scalar) # Rxn ctr atom
            else:
                atom.SetIsotope(atom.GetAtomicNum()) # Non rxn ctr atom

    cleared, patt = _mcs_precheck(mols, rcs, patt, enforce_ring_membership) # Prevents FindMCS default behavior of non-rc-mcs

    if not cleared:
        return None

    # Get the mcs that contains the reaction center pattern
    tmp = rdFMCS.FindMCS(
        mols,
        seedSmarts=patt,
        atomCompare=rdFMCS.AtomCompare.CompareIsotopes,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        matchChiralTag=False,
        ringMatchesRingOnly=enforce_ring_membership,
        completeRingsOnly=False,
        matchValences=True,
        timeout=10
    )

    rcmcs_patt = Chem.MolFromSmarts(tmp.smartsString)
    rcmcs_idxs = [m.GetSubstructMatch(rcmcs_patt) for m in mols]
    smarts_string = re.sub(r'\[(\d+)', _reset, tmp.smartsString) # Remove rc scaling

    res = {
        'smarts_string': smarts_string,
        'rcmcs_idxs': rcmcs_idxs,
        'num_atoms': tmp.numAtoms,
        'num_bonds': tmp.numBonds
    }

    return res

def _mcs_precheck(mols: list[Chem.Mol], rcs: list[tuple[int]], patt: str, enforce_ring_membership: bool):
    '''
    Modifies single-atom patts and pre-checks ring info
    to avoid giving FindMCS a non-common-substructure which
    results in non-reaction-center-inclusive MCSes
    '''
    if patt.count('#') == 1:
        patt = _handle_single_atom_patt(mols, rcs, patt)
    
    if enforce_ring_membership:
        cleared = _check_ring_membership(mols, rcs)
    else:
        cleared = True

    return cleared, patt

def _handle_single_atom_patt(mols: list[Chem.Mol], rcs: list[tuple[int]], patt: str):
    '''
    Pre-pends wildcard atom and bond to single-atom
    patt if mols share a neighbor w/ common isotope,
    ring membership, & bond type between
    '''
    couples = [set() for _ in range(len(mols))]
    for i, (mol, rc) in enumerate(zip(mols, rcs)):
        rc_idx = rc[0]
        for neighbor in mol.GetAtomWithIdx(rc_idx).GetNeighbors():
            nidx = neighbor.GetIdx()
            nisotope = neighbor.GetIsotope()
            in_ring = neighbor.IsInRing()
            bond_type = mol.GetBondBetweenAtoms(rc_idx, nidx).GetBondType()
            couples[i].add((nisotope, in_ring, bond_type))

    if len(set.intersection(*couples)) > 0:
        patt = '*~' + patt
    
    return patt

def _check_ring_membership(mols: list[Chem.Mol], rcs: list[tuple[int]]):
    '''
    Returns false if any "aligned" atom has distinct ring membership
    '''
    alignments = zip(*rcs)
    for elt in alignments:
        ring_membership = [mols[i].GetAtomWithIdx(idx).IsInRing() for i, idx in enumerate(elt)]

        if len(set(ring_membership)) != 1:
            return False
        
    return True
