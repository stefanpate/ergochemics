from rdkit import Chem
from rdkit.Chem import rdFMCS
from copy import deepcopy
import re

# TODO: reaction_rcmcs and reaction_rcmcs_score

# def calc_lhs_rcmcs(
#         rcts_rc1:Iterable,
#         rcts_rc2:Iterable,
#         patts:Iterable[str],
#         norm:str='max'
#     ):
#     '''
#     Calculates atom-weighted reaction rcmcs score of aligned reactions
#     using only reactants, NOT the products of the reaction.

#     Args
#     -------
#     rxn_rc:Iterable of len = 2
#         rxn_rc[0]:Iterable[str] - Reactant SMILES, aligned to operator
#         rxn_rc[1]:Iterable[Iterable[int]] - innermost iterables have reaction
#             center atom indices for a reactant
#     patts:Iterable[str]
#         SMARTS patterns of reaction center fragments organized
#         the same way as rxn_rc[1] except here, one SMARTS string per reactant
#     '''
#     smiles = [rcts_rc1[0], rcts_rc2[0]]
#     rc_idxs = [rcts_rc1[1], rcts_rc2[1]]
#     molecules= [[Chem.MolFromSmiles(smi) for smi in elt] for elt in smiles]
#     mol_rcs1, mol_rcs2 = [list(zip(molecules[i], rc_idxs[i])) for i in range(2)]
    
#     n_atoms = 0
#     rcmcs = 0
#     for mol_rc1, mol_rc2, patt in zip(mol_rcs1, mol_rcs2, patts):
#         rcmcs_i = calc_molecule_rcmcs(mol_rc1, mol_rc2, patt, norm=norm)

#         if norm == 'max':
#             atoms_i = max(mol_rc1[0].GetNumAtoms(), mol_rc2[0].GetNumAtoms())
#         elif norm == 'min':
#             atoms_i = min(mol_rc1[0].GetNumAtoms(), mol_rc2[0].GetNumAtoms())
        
#         rcmcs += rcmcs_i * atoms_i
#         n_atoms += atoms_i

#     return rcmcs / n_atoms

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
