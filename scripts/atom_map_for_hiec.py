from ergochem.mapping import atom_map_reaction
import json
import pandas as pd
from rdkit import Chem

def main():
    with open('v3_folded_pt_ns.json', 'r') as f:
        data = json.load(f)

    ops = pd.read_csv('minimal1224_all_uniprot.tsv', sep='\t')

    failed = []
    for k, v in data.items():
        op_name = v['min_rules'][0]
        op = ops.loc[ops['Name'] == op_name, 'SMARTS'].values[0]
        rxn = v['smarts']
        rc = v['rcs']
        am_smarts = atom_map_reaction(rxn, rc, op)
        if am_smarts is None:
            print(f"Failed to map {rxn} with operator {op}")
            failed.append(k)
        else:
            data[k]['am_smarts'] = am_smarts

    for k in failed:
        rcts, pdts = [[Chem.MolFromSmiles(mol) for mol in side.split('.')] for side in data[k]['smarts'].split('>>')]
        for r in rcts:
            for i, m in enumerate(r.GetAtoms()):
                m.SetAtomMapNum(i + 1)
        for p in pdts:
            for i, m in enumerate(p.GetAtoms()):
                m.SetAtomMapNum(i + 1)
        data[k]['am_smarts'] = '.'.join([Chem.MolToSmiles(m) for m in rcts]) + '>>' + '.'.join([Chem.MolToSmiles(m) for m in pdts])

    with open('output.json', 'w') as f:
        json.dump(data, f)

    print('done')

if __name__ == '__main__':
    main()