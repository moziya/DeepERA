# Implemented by Masashi Tsubaki et al., 2018
# Modified by Le Li, 2021
from collections import defaultdict
import os
import pickle
import sys
import pandas as pd
import numpy as np
import csv
import random

from rdkit import Chem


def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)

def extractSelected(dat,lst,idx=0):
    if type(lst) == list:
        key_eles = set(lst)
    else:
        key_eles = set(lst.iloc[:,0])
    all_idx = list(dat.iloc[:,idx])
    kept_idx = [i for i,ele in enumerate(all_idx) if ele in key_eles]
    return dat.iloc[kept_idx,:]

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        radius = 2
        ngram = 3
    else:
        radius, ngram = sys.argv[1:]
        radius, ngram = map(int, [radius, ngram])

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))
    Smiles, compounds, adjacencies, proteins, interactions, cids, uids, tort = '', [], [], [], [], [], [], []

    drug_id, target_id = [],[]
    
    with open(f'Raw_data/Drug_list.txt','r') as f:
        reader = csv.reader(f,delimiter=' ')
        for row in reader:
            drug_id.append(int(row[0]))
            smiles = row[1]
            Smiles += smiles + '\n'
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
            atoms = create_atoms(mol)
            i_jbond_dict = create_ijbonddict(mol)
            fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
            compounds.append(fingerprints)#
            adjacency = create_adjacency(mol)
            adjacencies.append(adjacency)#
            
    with open(f'Raw_data/Protein_list.txt','r') as f:
        reader = csv.reader(f,delimiter=' ')
        for row in reader:
            target_id.append(row[0])
            sequence = row[1]
            words = split_sequence(sequence, ngram)
            proteins.append(words)#
    target_id_set = set(target_id)
    drug_id_set = set(drug_id)

    # phenotype information of proteins
    uid_tab = pd.read_csv('Raw_data/uid_pheno.txt')
    # side-effects of drugs
    cid_tab = pd.read_csv('Raw_data/cid_sideeffect.txt')
    # Pfam domains of proteins
    uid_tab_pfam = pd.read_csv('Raw_data/uid_PFAM.txt',delimiter='\t',header=None,names=['UID','DID'])
    # remove drugs not included in the study
    cid_tab_sel = cid_tab[cid_tab.CID.isin(drug_id)]

    # aggregate phenotypes for each protein
    set_val =[]
    for i in range(uid_tab.shape[0]):
        set_val.append(set(uid_tab.DID[i].split(";")))
    uid_tab['DID'] = set_val

    # aggregate domains for each protein
    set_val =[]
    for i in range(uid_tab_pfam.shape[0]):
        set_val.append(set(uid_tab_pfam.DID[i].split(";")[:-1]))
    uid_tab_pfam['DID'] = set_val

    # remove proteins not included in the study
    uid_tab_sel = uid_tab[uid_tab.UID.isin(target_id)].reset_index(drop=True)
    uid_tab_pfam = uid_tab_pfam[uid_tab_pfam.UID.isin(target_id)].reset_index(drop=True)

    # aggregate side-effects for each drug
    cid_int_sel = cid_tab_sel[cid_tab_sel.type=='PT'][['CID','DID']].reset_index(drop=True)
    res = cid_int_sel.groupby('CID').groups
    ky_lst = []
    vl_lst = []
    for ky in res.keys():
        ky_lst.append(ky)
        vl_lst.append(set(cid_int_sel.DID[res[ky]].values.tolist()))

    dic = {'CID':ky_lst,'DID':vl_lst}
    cid_int_sel = pd.DataFrame(dic)


    # featurize domains of proteins with a matrix (num_protein x num_domain)
    set_all = set()
    for i in range(uid_tab_pfam.shape[0]):
        set_all = set_all.union(uid_tab_pfam['DID'][i])
    set_all = sorted(list(set_all))
    num_pfam = len(set_all)
    res = np.zeros((len(target_id),len(set_all))).astype(float)
    for i in range(uid_tab_pfam.shape[0]):
        sel_row = target_id.index(uid_tab_pfam['UID'][i])
        sel_idx = [set_all.index(ele) for ele in uid_tab_pfam['DID'][i] if ele in set_all]
        res[sel_row,sel_idx] = 1

    # featurize phenotypes/diseases of proteins with a matrix (num_protein x num_disease)
    set_all = set()
    for i in range(uid_tab_sel.shape[0]):
        set_all = set_all.union(uid_tab_sel['DID'][i])
    set_all = sorted(list(set_all))
    res2 = np.zeros((len(target_id),len(set_all))).astype(float)
    for i in range(uid_tab_sel.shape[0]):
        sel_row = target_id.index(uid_tab_sel['UID'][i])
        sel_idx = [set_all.index(ele) for ele in uid_tab_sel['DID'][i]]
        res2[sel_row,sel_idx] = 1

    # featurize sideeffects of drugs with a matrix (num_drug x num_sideeffect)
    set_all = set()
    for i in range(cid_int_sel.shape[0]):
        set_all = set_all.union(cid_int_sel['DID'][i])
    set_all = sorted(list(set_all))
    res3 = np.zeros((len(drug_id),len(set_all))).astype(float)
    for i in range(cid_int_sel.shape[0]):
        sel_row = drug_id.index(cid_int_sel['CID'][i])
        sel_idx = [set_all.index(ele) for ele in cid_int_sel['DID'][i]]
        res3[sel_row,sel_idx] = 1

    # create an empty matrix for the domain features of drugs
    drug_pfam = np.zeros((len(drug_id),num_pfam)).astype(float)

    # read interactions
    with open(f'Raw_data/Drug_protein_interactions.txt', 'r') as f:
        data_list = f.read().strip().split('\n')
    N = len(data_list)
    int_num = 0
    nint_num = 0
    int_pos,int_neg = [],[]
    prot_cnt_lst = np.zeros(len(drug_id)).astype(float)
    for noi, data in enumerate(data_list):
        if len(data)<4:
            cid, uid, interaction = data.strip().split()
            t_or_t = -1
        else:
            t_or_t, cid, uid, interaction = data.strip().split()
        cid_m = drug_id.index(int(cid))
        uid_m = target_id.index(uid)
        if (int(t_or_t) == 1) and (int(interaction) == 1): # only training set is used to estimate domain features of drugs
            drug_pfam[cid_m,:] += res[uid_m,:]
            prot_cnt_lst[cid_m] += 1
#        if cid_m >= len(drug_id) or cid_m < 0 or uid_m >= len(target_id) or uid_m < 0:
#            print([cid_m,uid_m])
        tort.append(np.array([int(t_or_t)]))
        uids.append(uid_m)
        cids.append(cid_m)
        interactions.append(np.array([float(interaction)]))#

    # estimate domain features of drugs
    drug_pfam_score = drug_pfam.copy()
    for i in range(len(drug_id)):
        if prot_cnt_lst[i] > 0:
            drug_pfam[i,:] /= prot_cnt_lst[i]
    # calculate the background domain features for drugs without known 
    back_drug_pfam = drug_pfam[prot_cnt_lst>0,:]
    back_pfam_aff = np.mean(back_drug_pfam)
    for i in range(len(drug_id)):
        if prot_cnt_lst[i]==0:
            drug_pfam[i,:] = back_pfam_aff

    dir_input = ('Processed_data/')
    os.makedirs(dir_input, exist_ok=True)

    # read networks
    PPI = pd.read_csv(f'Raw_data/PPN.txt',delimiter=' ',header=None, dtype=object)
    DDI = pd.read_csv(f'Raw_data/DDN.txt',delimiter=' ',header=None, dtype=int)
    if PPI.shape[1] == 2:
        PPI.columns = ['ID1','ID2']
        PPI['Score'] = np.ones((PPI.shape[0])) # obsolete edge value
    else:
        PPI.columns = ['ID1','ID2','Score']

    if DDI.shape[1] == 2:
        DDI.columns = ['ID1','ID2']
        DDI['Cnt1'] = 1 # obsolete edge value for different types
        DDI['Cnt2'] = 1 #
        DDI['Cnt3'] = 1 #
        DDI['Cnt4'] = 1 #
        DDI['Cnt5'] = 1 #
        DDI['Cnt6'] = 1 #
        DDI['Cnt7'] = 1 #
    else:
        DDI.columns = ['ID1','ID2','Cnt1','Cnt2','Cnt3','Cnt4','Cnt5','Cnt6','Cnt7']

    PPI_net = [[target_id.index(row['ID1']),target_id.index(row['ID2']),float(row['Score'])] for i,row in PPI.iterrows()]#
    DDI_net = [[drug_id.index(row['ID1']),drug_id.index(row['ID2']),row[2],row[3],row[4],row[5],row[6],row[7],row[8]] for i,row in DDI.iterrows()]#


    with open(dir_input + 'Smiles.txt', 'w') as f:
        f.write(Smiles)
    np.save(dir_input + 'compounds', compounds)
    np.save(dir_input + 'adjacencies', adjacencies)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'trainortest', tort)
    np.save(dir_input + 'interactions', interactions)
    np.save(dir_input + 'compounds_node_index', cids)
    np.save(dir_input + 'proteins_node_index', uids)
    np.save(dir_input + 'proteins_network', PPI_net)
    np.save(dir_input + 'compounds_network', DDI_net)
    np.save(dir_input + 'protein_domain_feature', res)
    np.save(dir_input + 'drug_domain_feature', drug_pfam)
    np.save(dir_input + 'drug_domain_score_feature', drug_pfam_score)
    np.save(dir_input + 'protein_disease_feature', res2)
    np.save(dir_input + 'drug_sideeffect_feature', res3)
    dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')
    dump_dictionary(word_dict, dir_input + 'word_dict.pickle')
