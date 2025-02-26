from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from typing import Iterable
from itertools import product

def _handle_kwargs(**kwargs):
    default_kwargs = {
        'do_canon_taut':False,
        'do_neutralize':True,
        'do_find_parent':True,
        'do_remove_stereo':True,
        'max_tautomers':50,
        'quiet': False,
    }
    filtered_kwargs = {k : v for k, v in kwargs.items() if k in default_kwargs}
    default_kwargs.update(filtered_kwargs)
    return default_kwargs


def standardize_mol(mol: Chem.Mol, **kwargs) -> Chem.Mol:
    '''
    Standardize a molecule using RDKit's standardization tools.

    Args
    ----
    mol:rdkit.Chem.rdchem.Mol
        Molecule to standardize.
    kwargs:dict
        Keyword arguments to pass to the standardization functions.
        - do_canon_taut:bool
            Whether to return canonical tautomer
        - do_neutralize:bool
            Whether to neutralize charges
        - do_find_parent:bool
            Whether to find the parent molecule
        - do_remove_stereo:bool
            Whether to remove stereochemistry
        - max_tautomers:int
            Maximum number of tautomers to generate
        - quiet:bool
            Whether to suppress warnings
    Returns
    -------
    mol:rdkit.Chem.rdchem.Mol
        Standardized molecule.
    '''
    kwargs = _handle_kwargs(**kwargs)

    if kwargs['quiet']:
        _ = rdBase.BlockLogs()

    if kwargs['do_remove_stereo']:
        Chem.rdmolops.RemoveStereochemistry(mol)

    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    # Also checks valency, that mol is kekulizable
    mol = rdMolStandardize.Cleanup(mol)

    # if many fragments, get the "parent" (the actual mol we are interested in)
    if kwargs['do_find_parent']:
        mol = rdMolStandardize.FragmentParent(mol)

    if kwargs['do_neutralize']:
        mol = neutralize_charges(mol) # Remove charges on atoms matching common patterns

    # Enumerate tautomers and choose canonical one
    if kwargs['do_canon_taut']:
        te = rdMolStandardize.TautomerEnumerator()
        te.SetMaxTautomers(kwargs['max_tautomers'])
        mol = te.Canonicalize(mol)
    
    return mol

def standardize_smiles(smiles:str, **kwargs) -> str:
    '''
    Standardize a molecule using RDKit's standardization tools.
    Args
    ----
    smiles:str
        SMILES string to standardize.
    kwargs:dict
        Keyword arguments to pass to the standardization functions.
        - do_canon_taut:bool
            Whether to return canonical tautomer
        - do_neutralize:bool
            Whether to neutralize charges
        - do_find_parent:bool
            Whether to find the parent molecule
        - do_remove_stereo:bool
            Whether to remove stereochemistry
        - max_tautomers:int
            Maximum number of tautomers to generate
        - quiet:bool
            Whether to suppress warnings
    Returns
    -------
    smiles:str
        Standardized SMILES string.
    '''
    kwargs = _handle_kwargs(**kwargs)
    mol = Chem.MolFromSmiles(smiles)
    mol = standardize_mol(
        mol,
        **kwargs
    )
    return Chem.MolToSmiles(mol)

def standardize_rxn(rxn: str, **kwargs) -> str:
    kwargs = _handle_kwargs(**kwargs)
    '''
    Standardize a reaction using RDKit's standardization tools.

    Args
    ----
    rxn:str
        SMARTS-encoded reaction, 'reactant.reactant>>product.product'
    kwargs:dict
        Keyword arguments to pass to the standardization functions.
        - do_canon_taut:bool
            Whether to return canonical tautomer
        - do_neutralize:bool
            Whether to neutralize charges
        - do_find_parent:bool
            Whether to find the parent molecule
        - do_remove_stereo:bool
            Whether to remove stereochemistry
        - max_tautomers:int
            Maximum number of tautomers to generate
        - quiet:bool
            Whether to suppress warnings
    Returns
    -------
    rxn:str
        Standardized reaction.
    '''
    rcts, pdts = [side.split('.') for side in rxn.split('>>')]
    rcts = [standardize_smiles(r, **kwargs) for r in rcts]
    pdts = [standardize_smiles(p, **kwargs) for p in pdts]
    return f"{'.'.join(rcts)}>>{'.'.join(pdts)}"

def neutralize_charges(mol: Chem.Mol) -> Chem.Mol:
    """Neutralize all charges in an rdkit mol.

    Args
    ----
    mol : rdkit.Chem.rdchem.Mol
        Molecule to neutralize.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        Neutralized molecule.
    """
    patts = (
        ("[n+;H]", "n"), # Imidazoles
        ("[N+;!H0]", "N"), # Amines
        ("[$([O-]);!$([O-][#7])]", "O"), # Carboxylic acids and alcohols
        ("[S-;X1]", "S"), # Thiols
        ("[$([N-;X2]S(=O)=O)]", "N"), # Sulfonamides
        ("[$([N-;X2][C,N]=C)]", "N"), # Enamines
        ("[n-]", "[nH]"), # Tetrazoles
        ("[$([S-]=O)]", "S"), # Sulfoxides
        ("[$([N-]C=O)]", "N"), # Amides
    )

    reactions = [
        (AllChem.MolFromSmarts(x), AllChem.MolFromSmiles(y, False)) for x,y in patts
    ]

    for (reactant, product) in reactions:
        while mol.HasSubstructMatch(reactant):
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    return mol


def fast_tautomerize(smiles: str) -> Iterable[str]:
    '''
    Applies common tautomerization patterns and returns
    any identified tautomers. Input smiles is returned
    at the first index.

    Args
    ----
    smiles:str
        SMILES string to standardize.
    Returns
    -------
    tautomers:Iterable[str]
        List of tautomers.
    '''
    transformations = [
        "[#7H1X3&a:1]:[#6&a:2]:[#7H0X2&a:3]>>[#7H0X2:1]:[#6:2]:[#7H1X3:3]",
        # TODO: add other common patterns
    ]

    tautomer_mols = []
    for trans in transformations:
        rxn = AllChem.ReactionFromSmarts(trans)
        try:
            outputs = rxn.RunReactants((Chem.MolFromSmiles(smiles),))
        except:
            print(f"Warning: rdkit sanitization failed for: {smiles}")
            outputs = rxn.RunReactants((Chem.MolFromSmiles(smiles, sanitize=False),))

        tautomer_mols.extend([o[0] for o in outputs])

    tautomer_smiles = [Chem.MolToSmiles(m) for m in tautomer_mols]
    return [smiles] + list(set(tautomer_smiles))

###############################################################################################   

def postsanitize_smiles(smiles_list):
    """Postsanitize smiles after running SMARTS.
    :returns tautomer list of list of smiles"""
    sanitized_list = []
    tautomer_smarts = "[#7H1X3&a:1]:[#6&a:2]:[#7H0X2&a:3]>>[#7H0X2:1]:[#6:2]:[#7H1X3:3]"
    for s in smiles_list:
        temp_mol = Chem.MolFromSmiles(s, sanitize=False)
        aromatic_bonds = [
            i.GetIdx()
            for i in temp_mol.GetBonds()
            if i.GetBondType() == Chem.rdchem.BondType.AROMATIC
        ]
        for i in temp_mol.GetBonds():
            if i.GetBondType() == Chem.rdchem.BondType.UNSPECIFIED:
                i.SetBondType(Chem.rdchem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(temp_mol)
            Chem.rdmolops.RemoveStereochemistry(temp_mol)
            temp_smiles = Chem.MolToSmiles(temp_mol)
        except Exception as msg:
            if "Can't kekulize mol" in str(msg):
                pyrrole_indices = [
                    i[0] for i in temp_mol.GetSubstructMatches(Chem.MolFromSmarts("n"))
                ]
                # indices to sanitize
                for s_i in pyrrole_indices:
                    temp_mol = Chem.MolFromSmiles(s, sanitize=False)
                    if temp_mol.GetAtomWithIdx(s_i).GetNumExplicitHs() == 0:
                        temp_mol.GetAtomWithIdx(s_i).SetNumExplicitHs(1)
                    elif temp_mol.GetAtomWithIdx(s_i).GetNumExplicitHs() == 1:
                        temp_mol.GetAtomWithIdx(s_i).SetNumExplicitHs(0)
                    try:
                        Chem.SanitizeMol(temp_mol)
                        processed_pyrrole_indices = [
                            i[0]
                            for i in temp_mol.GetSubstructMatches(
                                Chem.MolFromSmarts("n")
                            )
                        ]
                        processed_aromatic_bonds = [
                            i.GetIdx()
                            for i in temp_mol.GetBonds()
                            if i.GetBondType() == Chem.rdchem.BondType.AROMATIC
                        ]
                        if (
                            processed_pyrrole_indices != pyrrole_indices
                            or aromatic_bonds != processed_aromatic_bonds
                        ):
                            continue
                        Chem.rdmolops.RemoveStereochemistry(temp_mol)
                        temp_smiles = Chem.MolToSmiles(temp_mol)
                        break
                    except:
                        continue
                if "temp_smiles" not in vars():
                    Chem.rdmolops.RemoveStereochemistry(temp_mol)
                    temp_smiles = Chem.MolToSmiles(temp_mol)
                    sanitized_list.append([temp_smiles])
                    continue
            else:
                Chem.rdmolops.RemoveStereochemistry(temp_mol)
                temp_smiles = Chem.MolToSmiles(temp_mol)
                sanitized_list.append([temp_smiles])
                continue
        rxn = AllChem.ReactionFromSmarts(tautomer_smarts)
        try:
            tautomer_mols = rxn.RunReactants((Chem.MolFromSmiles(temp_smiles),))
        except:
            try:
                tautomer_mols = rxn.RunReactants(
                    (Chem.MolFromSmiles(temp_smiles, sanitize=False),)
                )
            except:
                continue
        tautomer_smiles = [Chem.MolToSmiles(m[0]) for m in tautomer_mols]
        sanitized_list.append(sorted(set(tautomer_smiles + [temp_smiles])))
    return list(product(*sanitized_list))