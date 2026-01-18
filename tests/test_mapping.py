from ergochemics.mapping import get_reaction_center, operator_map_reaction

decarboxylation_rule = '[#6:1]-[#6:2]-[#8:3]>>[#6:1].[#6:2]=[#8:3]'
decarboxylation_reaction = 'OCC(N)C(=O)O>>OCC(N).O=C=O'

def test_operator_map_reaction():
    res = operator_map_reaction(decarboxylation_reaction, decarboxylation_rule)
    print(res)
    assert res.did_map
    assert res.aligned_smarts == 'NC(CO)C(=O)O>>NCCO.O=C=O'
    assert res.atom_mapped_smarts == '[NH2:2][CH:1]([CH2:3][OH:4])[C:5](=[O:7])[OH:6]>>[NH2:2][CH2:1][CH2:3][OH:4].[O:6]=[C:5]=[O:7]'
    assert res.template_aidxs == (((1, 4, 6),), ((1,), (0, 1)))

tetrahedral_chiral_inversion = '[C:1][C:2][C@H:3]([C:4])[Br:5]>>[C:1][C:2][C@@H:3]([C:4])[Br:5]'
stereo_double_bond_inversion = '[C:1]/[C:2]=[C:3]/[C:4]=[O:5]>>[C:1]/[C:2]=[C:3]\\[C:4]=[O:5]'

def test_reaction_center_tetrahedral_inversion():
    rc_tetrahedral = get_reaction_center(tetrahedral_chiral_inversion, include_stereo=True)
    assert rc_tetrahedral == (((1, 2, 3, 4), ), ((1, 2, 3, 4), ))

def test_reaction_center_stereo_double_bond_inversion():
    rc_double_bond = get_reaction_center(stereo_double_bond_inversion, include_stereo=True)
    assert rc_double_bond == (((0, 1, 2, 3), ), ((0, 1, 2, 3), ))
