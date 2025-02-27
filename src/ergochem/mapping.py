from rdkit import Chem
from rdkit.Chem import rdChemReactions
import re
from ergochem.standardize import (
    standardize_rxn,
    standardize_smiles,
    fast_tautomerize
)
from typing import Iterable
from pydantic import BaseModel
from itertools import permutations, product, chain

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

def operator_map_reaction(rxn: str, operator: str, matched_idxs=None, max_outputs=10_000) -> OperatorMapResult:
    '''
    Attempts to map operator to reaction.
    
    Args
    ----
    rxn:str
        Reaction SMILES
    operator:str
        Reaction operator in SMARTS
    matched_idxs:Iterable[Iterable[int]]
        Permutation of reactant indices that such that
        reactants match "roles" of in operator LHS
    max_outputs:int
        Maximum number of outputs to generate w/ operator

    Returns
    -------
    OperatorMapResult
        Result of mapping. See class for details
    '''
    # TODO: incorporate template mapping here? tricky thing is depends on coreactant list
    # TODO: could think about putting this in a class w/ rules and rxns and not standardizing
    # every single time rxns * operators

    # TODO: if you want to treat tautomers, the thing to do is to "tautomer expand"
    # both the reactants and products then do the full mapping over all combos. This way
    # the operator maps iff the operator can transform some tautomer form of reactants
    # into some tautomer form of products ! The old way is wrong I believe.

    rxn = standardize_rxn(rxn, **MAPPING_STANDARDIZATION_DEFAULTS)
    rcts, pdts = [elt.split('.') for elt in rxn.split('>>')]
    op_lhs, op_rhs = extract_operator_patts(operator)

    if [len(rcts), len(pdts)] != [len(op_lhs), len(op_rhs)]: # First check cardinality
        return OperatorMapResult(did_map=False)
    
    # Mark reactant atoms for atom mapping
    rcts_mol = [Chem.MolFromSmiles(r) for r in rcts]
    for i, m in enumerate(rcts_mol):
        for atom in m.GetAtoms():
            atom.SetIntProp('reactant_idx', i)

    op = rdChemReactions.ReactionFromSmarts(operator) # Make reaction object from smarts string
    
    # Preserve mapping of op am numbers to op reactant indices, i.e., which reactant template
    # each atom map number belongs to (will lose this after running operator)
    am_to_reactant_idx ={}
    for ri in range(op.GetNumReactantTemplates()):
        rt = op.GetReactantTemplate(ri)
        for atom in rt.GetAtoms():
            if atom.GetAtomMapNum():
                am_to_reactant_idx[atom.GetAtomMapNum()] = ri
    
    if matched_idxs is None:
        matched_idxs = permutations([i for i in range(len(rcts))])

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

            correct_output = _compare_operator_outputs_w_products(outputs, pdts)

            if correct_output is not None:
                aligned_rxn, am_rxn, rhs_rc = _finalize_mapped_reaction(reactants=perm, output=correct_output, am_to_reactant_idx=am_to_reactant_idx)
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

def _finalize_mapped_reaction(reactants: Iterable[Chem.Mol], output: Iterable[Chem.Mol], am_to_reactant_idx: dict[int, int]) -> tuple[str, str, tuple[tuple[int, ...], tuple[int, ...]]]:
    '''
    Args
    ----
    reactants:Iterable[Chem.Mol]
        Reactants. Note: must be aligned to operator LHS
    output:Iterable[Chem.Mol]
        Output from operator.RunReactants(reactants) that
        matches the actual products
    am_to_reactant_idx:dict[int, int]
        Mapping of atom map numbers to reactant indices
        (i.e. which reactant the atom map number belongs to)
    Returns
    -------
    tuple[str, str]
        Operator aligned reaction without atom mapping
        Operator aligned reaction with atom mapping
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
                reactants[rct_idx].GetAtomWithIdx(rct_atom_idx).SetAtomMapNum(am)
            else: # atom part of reaction center <=> lost rct_idx
                old_am = props.get('old_mapno')
                rct_idx = am_to_reactant_idx[old_am]
                reactants[rct_idx].GetAtomWithIdx(rct_atom_idx).SetAtomMapNum(am)
                prod_rc.append(atom.GetIdx())
            
            am += 1

        rhs_rc.append(prod_rc)
    
    aligned_with_am = '.'.join([Chem.MolToSmiles(m) for m in reactants]) + '>>' + '.'.join([Chem.MolToSmiles(m) for m in output])
    rhs_rc = tuple(tuple(elt) for elt in rhs_rc)
    return aligned_no_am, aligned_with_am, rhs_rc

# TODO: This doesn't need to be a separate function
def _apply_operator_to_known_rc(op: rdChemReactions.ChemicalReaction, reactants: Iterable[Chem.Mol], lhs_rc: Iterable[Iterable[int]]) -> tuple[tuple[Chem.Mol]]:
    '''
    Applies operator to reaction w/ only LHS reaction center unprotected
    
    Args
    ----
    op:rdChemReactions.ChemicalReaction
        Operator
    reactants:Iterable[Chem.Mol]
        Reactants
    lhs_rc:Iterable[Iterable[int]]
        Reaction center indices. Must correspond to order of reactants

    Returns
    -------
    tuple[tuple[Chem.Mol]]
        Ouptput of operator
    '''

    for r, mol_rc in zip(reactants, lhs_rc):
        for a in r.GetAtoms():
            if a.GetIdx() in mol_rc:
                continue
            else:
                a.SetProp('_protected', '1')

    outputs = op.RunReactants(reactants)

    return outputs

# TODO: unify this with helpers above and name something more descriptive
# of the fact that this requires rc and op
def atom_map_reaction(rxn: str, rc: Iterable[Iterable[Iterable[int]]], op: str) -> str:
    '''
    Label reaction with all atom map numbers. Required that the reaction
    center and minimal operatoer are already known.

    Args
    ----
    rxn:str
        Reaction SMILES
    rc:Iterable[Iterable[int]]
        Reaction center indices. Outer iterable is len 2,
        next iterable is len n rcts or n prods,
        next is len(n rc atoms in molecule)
    op:str
        Reaction operator

    Returns
    -------
    rxn:str
        Reaction SMILES with atom map numbers
    '''

    reactants, products = [elt.split('.') for elt in rxn.split('>>')]
    reactants = [Chem.MolFromSmiles(r) for r in reactants]
    products = sorted(
        [
            standardize_smiles(p, do_canon_taut=False, do_find_parent=False, quiet=True)
            for p in products
        ]
    )

    for i, m in enumerate(reactants):
        for atom in m.GetAtoms():
            atom.SetIntProp('reactant_idx', i)

    op = rdChemReactions.ReactionFromSmarts(op)
    
    am_to_reactant_idx ={}
    for ri in range(op.GetNumReactantTemplates()):
        rt = op.GetReactantTemplate(ri)
        for atom in rt.GetAtoms():
            if atom.GetAtomMapNum():
                am_to_reactant_idx[atom.GetAtomMapNum()] = ri

    outputs = _apply_operator_to_known_rc(op, reactants, rc[0])

    match = False
    for output in outputs:
        try:
            output_smi = sorted([Chem.MolToSmiles(mol) for mol in output])
        except:
            continue
        
        if output_smi == products:
            match = True
            break

    if not match:
        return None

    am = 1
    for prod in output:
        for atom in prod.GetAtoms():
            props = atom.GetPropsAsDict()
            atom.SetAtomMapNum(am)
            rct_atom_idx = props.get('react_atom_idx')
            rct_idx = props.get('reactant_idx')
            
            if rct_idx is not None:
                reactants[rct_idx].GetAtomWithIdx(rct_atom_idx).SetAtomMapNum(am)
            else:
                old_am = props.get('old_mapno')
                rct_idx = am_to_reactant_idx[old_am]
                reactants[rct_idx].GetAtomWithIdx(rct_atom_idx).SetAtomMapNum(am)
            
            am += 1

    return '.'.join([Chem.MolToSmiles(m) for m in reactants]) + '>>' + '.'.join([Chem.MolToSmiles(m) for m in output])

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
    # Test the function
    op = '[C:1].[C:2]>>[C:1][C:2]'
    rxn = 'CCOC.CC>>CCOCC'
    reactants = [Chem.MolFromSmiles('CCOC'), Chem.MolFromSmiles('CC')]
    rc = [[[0,], [0,]], []]


    with open('v3_folded_pt_ns.json', 'r') as f:
        data = json.load(f)

    ops = pd.read_csv('minimal1224_all_uniprot.tsv', sep='\t')

    for k, v in data.items():
        op_name = v['min_rules'][0]
        op = ops.loc[ops['Name'] == op_name, 'SMARTS'].values[0]
        rxn = v['smarts']
        res = operator_map_reaction(rxn=rxn, operator=op, matched_idxs=None)

    for k, v in data.items():
        op_name = v['min_rules'][0]
        op = ops.loc[ops['Name'] == op_name, 'SMARTS'].values[0]
        rxn = v['smarts']
        rc = v['rcs']
        am_smarts = atom_map_reaction(rxn, rc, op)
        if am_smarts is None:
            print(f"Failed to map {rxn} with operator {op}")

    
    # am_smarts = atom_map_reaction(rxn, rc, op)
    print('hold')


    

# def match_template(rxn, rule_reactants_template, rule_products_template, smi2paired_cof, smi2unpaired_cof):
#     '''
#     Returns the permuted indices corresponding to
#     a match between reactant and rule templates
#     '''
#     reactants_smi, products_smi = split_reaction(rxn)
#     rule_reactants_template = tuple(rule_reactants_template.split(';'))
#     rule_products_template = tuple(rule_products_template.split(';'))
#     matched_idxs = [] # Return empty if no matches found
#     # First check the cardinality of reactants, products matches
#     if (len(rule_reactants_template) == len(reactants_smi)) & (len(rule_products_template) == len(products_smi)):

#         reactants_template = ['Any' for elt in reactants_smi]
#         products_template = ['Any' for elt in products_smi]

#         # Search for unpaired cofactors first
#         for i, r in enumerate(reactants_smi):
#             if r in smi2unpaired_cof:
#                 reactants_template[i] = smi2unpaired_cof[r]

#         for i, p in enumerate(products_smi):
#             if p in smi2unpaired_cof:
#                 products_template[i] = smi2unpaired_cof[p]

#         # Search for paired cofactors
#         # Only overwriting should be PPi/Pi as phosphate donor/acceptor
#         for i, r in enumerate(reactants_smi):
#             for j, p in enumerate(products_smi):
#                 if (r, p) in smi2paired_cof:
#                     reactants_template[i] = smi2paired_cof[(r, p)][0]
#                     products_template[j] = smi2paired_cof[(r, p)][1]
#                 elif (p, r) in smi2paired_cof:
#                     reactants_template[i] = smi2paired_cof[(p, r)][1]
#                     products_template[j] = smi2paired_cof[(p, r)][0]

#         reactants_idx_template = [(elt, i) for i, elt in enumerate(reactants_template)]

#         # First try to products templates
#         product_template_match = False
#         for perm in permutations(products_template):
#             if perm == rule_products_template:
#                 product_template_match = True

#         # If product templates match
#         # find permutations of reactant template that match
#         # rule template and keep the indices of those good permutations
#         # Else return empty list
#         if product_template_match:
#             for perm in permutations(reactants_idx_template):
#                 this_template, this_idx = list(zip(*perm))
#                 if this_template == rule_reactants_template:
#                     matched_idxs.append(this_idx)

#     return matched_idxs
