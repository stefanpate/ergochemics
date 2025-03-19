from rdkit import Chem
from rdkit.Chem import rdChemReactions
import re
from typing import Iterable
from pydantic import BaseModel
from itertools import permutations, product, chain
from functools import lru_cache
from ergochemics.standardize import (
    standardize_rxn,
    fast_tautomerize
)

class OperatorMapResult(BaseModel):
    '''
    Result of mapping a reaction to a reaction operator.
    Attributes
    ----------
    did_map:bool
        Whether the mapping was successful
    aligned_smarts:str | None
        Reaction SMARTS with reactants and products
        aligned to the operator reactants and products
    atom_mapped_smarts:str | None
        Reaction SMARTS with atom map numbers
    reaction_center:tuple[tuple[int]] | None
        Reaction center indices. Outer
        iterable is len 2,
        next iterable is len n rcts or n prods,
        next is len(n rc atoms in molecule i)
    '''
    did_map: bool
    aligned_smarts: str | None = None
    atom_mapped_smarts: str | None = None
    reaction_center: tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]] | None = None

MAPPING_STANDARDIZATION_DEFAULTS = {
    'do_canon_taut':False,
    'do_neutralize':True,
    'do_find_parent':False,
    'do_remove_stereo':True,
    'quiet': False,
}

@lru_cache(maxsize=1000)
def _m_standardize_reaction(rxn: str, **kwargs) -> str:
    return standardize_rxn(rxn, **kwargs)

@lru_cache(maxsize=1000)
def _cached_fast_tautomerize(smiles: str) -> str:
    return fast_tautomerize(smiles)

def _tautomer_expand(lhs: Iterable[str], rhs: Iterable[str]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    '''
    Expands reactants and products using fast_tautomerize
    and returns all combinations of reactants and products.

    Args
    ----
    lhs:Iterable[str]
        Reactants
    rhs:Iterable[str]
        Products

    Returns
    -------
    list[tuple[tuple[str, ...], tuple[str, ...]]]
        All combinations of reactants and products
    '''
    lhs = [_cached_fast_tautomerize(r) for r in lhs]
    rhs = [_cached_fast_tautomerize(p) for p in rhs]

    return list(product(product(*lhs), product(*rhs)))

def operator_map_reaction(rxn: str, operator: str, max_outputs=10_000) -> OperatorMapResult:
    '''
    Attempts to map operator to reaction.

    Note: The returned reaction center is really the set of all atoms specified in the 
    operator template. It is only truly the reaction center when the operator is minimal,
    i.e., its template only specifies the reaction center.
    
    Args
    ----
    rxn:str
        Reaction SMILES
    operator:str
        Reaction operator in SMARTS
    max_outputs:int
        Maximum number of outputs to generate w/ operator

    Returns
    -------
    OperatorMapResult
        Result of mapping. See class for details
    '''
    rxn = _m_standardize_reaction(rxn, **MAPPING_STANDARDIZATION_DEFAULTS)
    rcts, pdts = [elt.split('.') for elt in rxn.split('>>')]
    op_lhs, op_rhs = extract_operator_patts(operator)

    if [len(rcts), len(pdts)] != [len(op_lhs), len(op_rhs)]: # First check cardinality
        return OperatorMapResult(did_map=False)
    
    op = rdChemReactions.ReactionFromSmarts(operator) # Make reaction object from smarts string

    for taut_rcts, taut_pdts in _tautomer_expand(rcts, pdts): # Iterate over tautomer combinations
        
        # Mark reactant atoms for atom mapping
        rcts_mol = [Chem.MolFromSmiles(r) for r in taut_rcts]
        for i, m in enumerate(rcts_mol):
            for atom in m.GetAtoms():
                atom.SetIntProp('reactant_idx', i)

        # Preserve mapping of op am numbers to op reactant indices, i.e., which reactant template
        # each atom map number belongs to (will lose this after running operator)
        am_to_reactant_idx ={}
        for ri in range(op.GetNumReactantTemplates()):
            rt = op.GetReactantTemplate(ri)
            for atom in rt.GetAtoms():
                if atom.GetAtomMapNum():
                    am_to_reactant_idx[atom.GetAtomMapNum()] = ri
        
        matched_idxs = permutations([i for i in range(len(taut_rcts))])

        lhs_patts = [Chem.MolFromSmarts(l) for l in op_lhs]
        for idx_perm in matched_idxs:
            perm = [rcts_mol[idx] for idx in idx_perm]
            substruct_matches = [perm[i].GetSubstructMatches(lp) for i, lp in enumerate(lhs_patts)]

            if any([len(elt) == 0 for elt in substruct_matches]): # Check if this permutation of reactants matches operator templates
                continue

            ss_match_combos = product(*substruct_matches) # All combos of putative rcs of n substrates
            all_putative_rc_atoms = [set(chain(*elt)) for elt in substruct_matches] # ith element has set of all putative rc atoms of ith reactant

            for smc in ss_match_combos:

                # Protect all but rc currently considered in each reactant
                for j, reactant_rc in enumerate(smc):
                    all_but = all_putative_rc_atoms[j] - set(reactant_rc) 
                    for protect_idx in all_but:
                        perm[j].GetAtomWithIdx(protect_idx).SetProp('_protected', '1')

                outputs = op.RunReactants(perm, maxProducts=max_outputs)

                correct_output = _compare_operator_outputs_w_products(outputs, taut_pdts)

                if correct_output is not None:
                    aligned_rxn, am_rxn, rhs_rc = _finalize_mapped_reaction(reactants=perm, output=correct_output, permuted_idxs=idx_perm, am_to_reactant_idx=am_to_reactant_idx)
                    reaction_center = tuple([smc, rhs_rc])
                    return OperatorMapResult(
                        did_map=True,
                        aligned_smarts=aligned_rxn,
                        atom_mapped_smarts=am_rxn,
                        reaction_center=reaction_center
                    )

                # Deprotect & try again
                for j, reactant_rc in enumerate(smc):
                    all_but = all_putative_rc_atoms[j] - set(reactant_rc)
                    for protect_idx in all_but:
                        perm[j].GetAtomWithIdx(protect_idx).ClearProp('_protected')

    return OperatorMapResult(did_map=False) # Did not map

def _compare_operator_outputs_w_products(outputs: tuple[tuple[Chem.Mol]], products: list[str]) -> tuple[Chem.Mol] | None:
    '''
    Compares operator outputs to products and
    returns the products in the order of the operator outputs.
    Returns empty list if no match is found.
    '''
    srt_prod = tuple(sorted(products))

    for output in outputs:
        try:
            output_smi = [(Chem.MolToSmiles(mol), i) for i, mol in enumerate(output)]
            srt_out_smi, srt_oidx = zip(*sorted(output_smi, key=lambda x: x[0]))
        except:
            continue

        if srt_out_smi == srt_prod:
            return output    

    return None

def _finalize_mapped_reaction(reactants: Iterable[Chem.Mol], output: Iterable[Chem.Mol], permuted_idxs: list[int], am_to_reactant_idx: dict[int, int]) -> tuple[str, str, tuple[tuple[int, ...], tuple[int, ...]]]:
    '''
    Args
    ----
    reactants:Iterable[Chem.Mol]
        Reactants. Note: must be ordered as they match
        lhs operator templates
    output:Iterable[Chem.Mol]
        Output from operator.RunReactants(reactants) that
        matches the actual products
    permuted_idxs:list[int]
        Original indices of reactants in reaction,
        permuted to match the operator
    am_to_reactant_idx:dict[int, int]
        Mapping of atom map numbers to reactant indices
        (i.e. which reactant the atom map number belongs to)
    Returns
    -------
    :tuple[str, str, tuple[tuple[int, ...], tuple[int, ...]]]
        Operator aligned reaction without atom mapping
        Operator aligned reaction WITH atom mapping
        Reaction center indices
    '''
    aligned_no_am = '.'.join([Chem.MolToSmiles(m) for m in reactants]) + '>>' + '.'.join([Chem.MolToSmiles(m) for m in output])

    am = 1
    rhs_rc = []
    for prod in output:
        prod_rc = []
        for atom in prod.GetAtoms():
            atom.SetAtomMapNum(am)
            props = atom.GetPropsAsDict()
            rct_atom_idx = props.get('react_atom_idx')
            rct_idx = props.get('reactant_idx')
            
            if rct_idx is not None:
                reactants[permuted_idxs[rct_idx]].GetAtomWithIdx(rct_atom_idx).SetAtomMapNum(am)
            else:
                old_am = props.get('old_mapno')
                rct_idx = am_to_reactant_idx[old_am]
                reactants[rct_idx].GetAtomWithIdx(rct_atom_idx).SetAtomMapNum(am)
                prod_rc.append(atom.GetIdx())
            
            am += 1

        rhs_rc.append(prod_rc)
    
    aligned_with_am = '.'.join([Chem.MolToSmiles(m) for m in reactants]) + '>>' + '.'.join([Chem.MolToSmiles(m) for m in output])
    rhs_rc = tuple(tuple(elt) for elt in rhs_rc)
    return aligned_no_am, aligned_with_am, rhs_rc

def extract_operator_patts(smarts: str) -> tuple[tuple[str]]:
    '''
    Pulls SMARTS patterns from a reaction SMARTS string.
    
    Args
    ----
    smarts:str
        Reaction SMARTS
    Returns
    -------
    tuple[tuple[str]]
        Tuple of tuples of SMARTS patterns. Each tuple corresponds
        to a side of the reaction operator. Each pattern corresponds
        to the part of the reaction center contained within a given molecule.
    '''
    patts = []
    for side in smarts.split('>>'):
        smarts = re.sub(r':[0-9]+]', ']', side)

        # Identify each fragment
        side_patts = []
        temp_fragment = []

        # Append complete fragments only
        for fragment in smarts.split('.'):
            temp_fragment += [fragment]
            if '.'.join(temp_fragment).count('(') == '.'.join(temp_fragment).count(')'):
                side_patts.append('.'.join(temp_fragment))
                temp_fragment = []

                # Remove component grouping for substructure matching
                if '.' in side_patts[-1]:
                    side_patts[-1] = side_patts[-1].replace('(', '', 1)[::-1].replace(')', '', 1)[::-1]

        patts.append(tuple(side_patts))

    return tuple(patts)

if __name__ == '__main__':
    import json
    import pandas as pd
    from time import perf_counter

    with open('v3_folded_pt_ns.json', 'r') as f:
        data = json.load(f)

    ops = pd.read_csv('minimal1224_all_uniprot.tsv', sep='\t')

    ex_k, ex_rxn = list(data.items())[0]
    op = ops.loc[ops['Name'] == ex_rxn['min_rules'][0], 'SMARTS'].values[0]
    for _ in range(2):
        tic = perf_counter()
        res = operator_map_reaction(ex_rxn['smarts'], op)
        toc = perf_counter()
        print(f"Elapsed time: {toc - tic:.4f} seconds")

    for k, v in data.items():
        op_name = v['min_rules'][0]
        op = ops.loc[ops['Name'] == op_name, 'SMARTS'].values[0]
        rxn = v['smarts']
        res = operator_map_reaction(rxn=rxn, operator=op)

    print('hold')