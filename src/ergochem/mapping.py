from rdkit import Chem
from rdkit.Chem import rdChemReactions
import re
from ergochem.standardize import standardize_mol, standardize_smiles
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
    aligned_smarts: str | None
    atom_mapped_smarts: str | None
    reaction_center: tuple[tuple[int]] | None

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
    # TODO: invoke standardization, custom possible w/ default
    # TODO: incorporate template mapping here? tricky thing is depends on coreactant list

    rcts, pdts = [elt.split('.') for elt in rxn.split('>>')]
    op_lhs, op_rhs = extract_operator_patts(operator)

    if [len(rcts), len(pdts)] != [len(op_lhs), len(op_rhs)]:
        return OperatorMapResult(did_map=False)
    
    if matched_idxs is None:
        matched_idxs = permutations([i for i in range(len(rcts))])

    rcts_mol = [Chem.MolFromSmiles(r) for r in rcts]
    lhs_patts = [Chem.MolFromSmarts(l) for l in op_lhs]
    for idx_perm in matched_idxs:
        perm = [rcts_mol[idx] for idx in idx_perm]

        substruct_matches = [perm[i].GetSubstructMatches(lp) for lp, i in enumerate(lhs_patts)]

        if any([len(elt) == 0 for elt in substruct_matches]):
            continue

        ss_match_combos = product(*substruct_matches) # All combos of putative rcs of n substrates
        all_putative_rc_atoms = [set(chain(*elt)) for elt in substruct_matches] # ith element has set of all putative rc atoms of ith reactant

        for smc in ss_match_combos:

            # Protect all but rc currently considered in each reactant
            for j, reactant_rc in enumerate(smc):
                all_but = all_putative_rc_atoms[j] - set(reactant_rc) 
                for protect_idx in all_but:
                    perm[j].GetAtomWithIdx(protect_idx).SetProp('_protected', '1')

            outputs = operator.RunReactants(perm, maxProducts=max_outputs)

            op_compare = _compare_operator_outputs_w_products(outputs, pdts)

            if op_compare is not None:
                aligned_rcts = ".".join([Chem.MolFromSmiles(m) for m in perm])
                aligned_pdts = op_compare
                # TODO: atom map and think thru rc for RHS
                return OperatorMapResult()

            # Deprotect & try again
            for j, reactant_rc in enumerate(smc):
                all_but = all_putative_rc_atoms[j] - set(reactant_rc)
                for protect_idx in all_but:
                    perm[j].GetAtomWithIdx(protect_idx).ClearProp('_protected')



def map_rxn2rule(rxn, rule, return_rc=False, matched_idxs=None, max_products=10000):
    '''
    Maps reactions to SMARTS-encoded reaction rule.
    Args:
        - rxn: Reaction SMARTS string
        - rule: smarts string
        - return_rc: Return reaction center
        - matched_idxs: Indices of reaction reactants in the order they match the smarts
        reactants templates
    Returns:
        - res:dict{
            did_map:bool
            aligned_smarts:str | None
            reaction_center:Tuple[tuple] | None
        }
    '''
    res = {
        'did_map':False,
        'aligned_smarts':None,
        'reaction_center':None,
    }
    reactants, unsorted_products = split_reaction(rxn)
    
    products = sorted(unsorted_products) # Canonical ordering for later comparison
    operator = Chem.rdChemReactions.ReactionFromSmarts(rule) # Make reaction object from smarts string
    reactants_mol = [Chem.MolFromSmiles(elt) for elt in reactants] # Convert reactant smiles to mol obj
    rule_substrate_cts = [len(get_patts_from_operator_side(rule, i)) for i in range(2)] # [n_reactants, n_products] in a rule
    rxn_substrate_cts = [len(reactants), len(products)]

    # Check if number of reactants / products strictly match
    # rule to reaction. If not return false
    if rule_substrate_cts != rxn_substrate_cts:
        return res
    
    # If not enforcing templates,
    # get all permutations of reactant
    # indices
    if matched_idxs is None:
        matched_idxs = list(permutations([i for i in range(len(reactants))]))
        
    # For every permutation of that subset of reactants
    # TODO: What if there are multiple match idxs that product the right outputs?
    for idx_perm in matched_idxs:
        perm = tuple([reactants_mol[idx] for idx in idx_perm]) # Re-order reactants based on allowable idx perms
        outputs = operator.RunReactants(perm, maxProducts=max_products) # Apply rule to that permutation of reactants

        if compare_operator_outputs_w_products(outputs, products):
            res['did_map'] = True
            res['aligned_smarts'] = ".".join([reactants[idx] for idx in idx_perm]) + ">>" + ".".join(unsorted_products)
            break # out of permutations-of-matched-idxs loop

    if res['did_map'] and not return_rc: # Mapped and don't want rc
        return res

    elif res['did_map'] and return_rc: # Mapped and want rc
        patts = get_patts_from_operator_side(rule, 0)
        patts = [Chem.MolFromSmarts(elt) for elt in patts]

        if len(patts) != len(perm):
            raise Exception("Something wrong. There should be same number of operator fragments as reaction reactants")
        
        substruct_matches = [perm[i].GetSubstructMatches(patts[i]) for i in range(len(patts))]
        ss_match_combos = product(*substruct_matches) # All combos of putative rcs of n substrates
        all_putative_rc_atoms = [set(chain(*elt)) for elt in substruct_matches] # ith element has set of all putative rc atoms of ith reactant

        for smc in ss_match_combos:

            # Protect all but rc currently considered in each reactant
            for j, reactant_rc in enumerate(smc):
                all_but = all_putative_rc_atoms[j] - set(reactant_rc) # To protect: "all but current rc"
                for protect_idx in all_but:
                    perm[j].GetAtomWithIdx(protect_idx).SetProp('_protected', '1')

            outputs = operator.RunReactants(perm, maxProducts=max_products) # Run operator with protected atoms

            # If found match
            if _compare_operator_outputs_w_products(outputs, products):
                res['reaction_center'] = smc
                return res
            
            # Deprotect & try again
            for j, reactant_rc in enumerate(smc):
                all_but = all_putative_rc_atoms[j] - set(reactant_rc) # To protect: "all but current rc"
                for protect_idx in all_but:
                    perm[j].GetAtomWithIdx(protect_idx).ClearProp('_protected')

    return res # Did not map or failed getting RC

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

def _compare_operator_outputs_w_products(
        outputs: tuple[tuple[Chem.Mol]],
        products: list[str],
        standardization_params: dict[str, bool] = MAPPING_STANDARDIZATION_DEFAULTS
    ) -> bool:
    '''
    Compare operator outputs to products. If mapped, return True.
    
    '''
    # TODO:


    # Try WITHOUT tautomer canonicalization
    for output in outputs:
        try:
            output = sorted(
                [
                    Chem.MolToSmiles(
                        standardize_mol(
                            mol,
                            do_canon_taut=False,
                            do_neutralize=False,
                            do_find_parent=False,
                            quiet=True
                        )
                    )
                    for mol in output
                ]
            )
        except:
            continue

        # Compare predicted to actual products. If mapped, return True
        if output == products: 
            return True
        
    # # Try WITH tautomer canonicalization TODO: unacceptably slow
    # try:
    #     products = sorted([_poststandardize(Chem.MolFromSmiles(smi), do_canon_taut=True) for smi in products])
    # except:
    #     return False
    
    # for output in outputs:
    #     try:
    #         output = sorted([_poststandardize(mol, do_canon_taut=True) for mol in output]) # Standardize and sort SMILES
    #     except:
    #         continue

    #     # Compare predicted to actual products. If mapped, return True
    #     if output == products: 
    #         return True
            
    return False



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
            output_smi = sorted(
                Chem.MolToSmiles(
                    # TODO unify with standardization defaults
                    [
                        Chem.MolToSmiles(
                            standardize_mol(mol, do_canon_taut=False, do_neutralize=False, do_find_parent=False, quiet=True)
                        )
                        for mol in output
                    ]
                )
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
        rc = v['rcs']
        am_smarts = atom_map_reaction(rxn, rc, op)
        if am_smarts is None:
            print(f"Failed to map {rxn} with operator {op}")

    
    # am_smarts = atom_map_reaction(rxn, rc, op)
    print('hold')
