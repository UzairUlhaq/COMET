#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
import lmdb
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  
import warnings
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool
import json
import random
import shutil
import copy
import torch

from sklearn.model_selection import KFold, StratifiedKFold


# In[ ]:


# random seed
seed = 42
random.seed(seed)
np.random.seed(seed)


# ## Data processing with "component_type*" subclasses: e.g. "component_type_reaction_step", "component_type_component_type"

# In[ ]:


from functools import partial

"""
Create lmdb dataset with:
a) NP compositions: list of components where each component contain molecule_id
b) molecule dataset: like unimol original dataset
"""

def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates


def smi2_3Dcoords(smi,cnt):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list=[]
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)       # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    coordinates = smi2_2Dcoords(smi)            
                    
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)       # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    coordinates = smi2_2Dcoords(smi) 
        except:
            coordinates = smi2_2Dcoords(smi) 

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list

def inner_lnp2data(smi2mol_id, content, pickle_output=True):
    components_list = content['components']
    if "labels" in content:
        raw_labels = content['labels']
    else:
        raw_labels = {}
    dataset_name = content['dataset_name']
    lnp_id = content['lnp_id']

    # handle non-core (optional) attributes of LNPs, e.g. NP_ratio
    np_props = {}
    if "NP_ratio" in content:
        np_props['NP_ratio'] = content['NP_ratio']
    if "actual_ilrna_wt_ratio" in content:
        np_props['actual_ilrna_wt_ratio'] = content['actual_ilrna_wt_ratio']
    if "volumetric_ratio" in content:
        np_props['volumetric_ratio'] = content['volumetric_ratio']        

    labels = raw_labels # raw_labels is already normalized

    output_components_list = []
    
    mol_ids = []
    percents = []
    component_types = []
    # reaction_steps = []

    for component in components_list:

        # component_output = component.copy()
        component_output = copy.deepcopy(component)
        mol_id = smi2mol_id[component['smi']]
        component_output['mol_id'] = mol_id

        output_components_list.append(component_output)

        mol_ids.append(mol_id)

        percent = component['percent']
        percents.append(percent)

        component_type = component['component_type']
        component_types.append(component_type)


    output = {
        'mol_id': mol_ids, 
        'percent': percents, 'component_type': component_types, 
        'target': labels, 
        'dataset_name': dataset_name, 
        'components': output_components_list,
        'lnp_id': lnp_id,
        **np_props # fold non-core (optional) attributes of LNPs into sample dict, e.g. NP_ratio, volumetric_ratio
        }

    # print("inner_lnp2data output: ", output)
    
    if pickle_output:
        return pickle.dumps(output, protocol=-1)
    else:
        output

def lnp2data(smi2mol_id, content):
    try:
        return inner_lnp2data(smi2mol_id, content)
    except:
        print("failed lnp: {}".format(content[0]))
        return None

def inner_smi2coords(content, pickle_output=False):

    smi = content

    cnt = 10 # conformer num,all==11, 10 3d + 1 2d
    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > 400:
        coordinate_list =  [smi2_2Dcoords(smi)] * (cnt+1)
        print("atom num >400,use 2D coords",smi)
    else:
        coordinate_list = smi2_3Dcoords(smi,cnt)
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H 

    output = {'atoms': atoms, 
    'coordinates': coordinate_list, 
    'mol': mol,'smi': smi}

    if pickle_output:
        return pickle.dumps(output, protocol=-1)
    else:
        return output
        
def smi2coords_onlymol(content):
    try:
        return inner_smi2coords_onlymol(content)
    except:
        print("failed smiles: {}".format(content))
        return None

def inner_smi2coords_onlymol(content, pickle_output=True):
    output = inner_smi2coords(content)
    # print("inner_smi2coords_onlymol, output: ", output)
    if pickle_output:
        return pickle.dumps(output, protocol=-1)
    else:
        output



# ## With k-fold CV splits

# ## Function to make test set with top/bottom X% 

# In[ ]:


# with fixnoutputlnp_ids : to generate dataset based on given lnp_ids and output lnp_ids in train, valid and test sets
# and splitlabeldata
def write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(inpath='./', outpath='processed_data_dirs/', nthreads=16, test_ratio=0.1, kfold_valid=None, valid_ratio=None, 
                                                top_heldout_ratio=0, bottom_heldout_ratio=0, target_label_name="in_house_lnp_DC24_luc",  # topbottomheldout args 
                                                random_train_subsample_ratio=None, train_subsample_sample_ids=None, subsample_target_label=None, # trainsubsampling args
                                                split_multilabel_train_sample=False, # whether to split multilabel train samples into multiple samples, each with one label
                                                labels_to_split_into_subfolders=None, # split multilabel data samples into multiple dataset, each with one label, the name of the data folder will be appended with the label name
                                                train_lnp_ids=None, valid_lnp_ids=None, test_lnp_ids=None, # lnp_ids to include in train, valid and test sets
                                                debug=False, shuffle=True,
                                                ):
    
    if all(v != None for v in [train_lnp_ids, valid_lnp_ids, test_lnp_ids]):
        fixed_train_valid_test_split = True
    else:
        fixed_train_valid_test_split = False
    if not (all(v == None for v in [train_lnp_ids, valid_lnp_ids, test_lnp_ids]) or fixed_train_valid_test_split):
        raise ValueError("train_lnp_ids, valid_lnp_ids, test_lnp_ids must be all None or all not None")
    print("fixed_train_valid_test_split A: ", fixed_train_valid_test_split)

    # function creates a test set made up of top and bottom subset of the dataset, if not defined, function create a test set randomly (with size of test_ratio) 
    top_bottom_ratio = 0
    top_bottom_ratio += top_heldout_ratio
    top_bottom_ratio += bottom_heldout_ratio

    if top_bottom_ratio > test_ratio:
        raise ValueError("top_heldout_ratio + bottom_heldout_ratio > test_ratio")
    else:
        total_random_test_ratio = test_ratio - top_bottom_ratio
        remaining_random_test_ratio = total_random_test_ratio / (1 - top_bottom_ratio) # find the ratio of remaining dataset to randomly sample for test set after taking out top and bottom heldout set

    # Data wil be stored in JSON format, e.g. each sample: {components: [{smi: <SMILES>, percent: <%>, name: IL-1}, {..}], label: <label_value>}
    with open(os.path.join(inpath), 'r') as openfile:
        # Reading from json file
        json_obj = json.load(openfile)

    dataset_name_list = []
    dataset_dict = {}

    # collate into list of lnps
    json_list = []
    for lnp_id in json_obj:
        lnp_dict = json_obj[lnp_id]
        lnp_dict['lnp_id'] = lnp_id

        if 'dataset_name' in lnp_dict:
            lnp_dataset_name = lnp_dict['dataset_name']
            if lnp_dataset_name not in dataset_name_list:
                dataset_name_list.append(lnp_dataset_name)
                dataset_dict[lnp_dataset_name] = []
            dataset_dict[lnp_dataset_name].append(lnp_dict)

        # process components' percent value
        np_components = lnp_dict['components']
        # total_weight = 0

        # record percent composition in component dict
        for c_id, component in enumerate(np_components):
            lnp_dict['components'][c_id]['percent'] = lnp_dict['components'][c_id]['mol']

        json_list.append(lnp_dict)

    sz = len(json_list)
    print("sz: ", sz)

    # make master mol lmdb dataset as a list of all unique mols in datasets
    smi_list = []
    for sample_i, np_obj in enumerate(json_list):
        np_components = np_obj['components']
        unique_smi_count = 0
        for component in np_components:
            smi = component['smi']
            if smi not in smi_list:
                unique_smi_count += 1
                smi_list.append(smi)

    mol_filename = "mol.lmdb"
    os.makedirs(outpath, exist_ok=True)
    mol_output_name = os.path.join(outpath, mol_filename)
    try:
        os.remove(mol_output_name)
    except:
        pass
    env_new = lmdb.open(
        mol_output_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)
    with Pool(nthreads) as pool:
        i = 0
        for inner_output in tqdm(pool.imap(smi2coords_onlymol, smi_list)):
            if inner_output is not None:
                print("i=", i, " data = pickle.loads(datapoint_pickled) smi: ", pickle.loads(inner_output)['smi'])
                txn_write.put(f'{i}'.encode("ascii"), inner_output)
                i += 1
        print('{} process {} lines'.format(mol_filename, i))
        txn_write.commit()
        env_new.close()
    
    print("finished processing mol.lmdb")

    # make smi2mol_id dict
    smi2mol_id = {}
    mol_id2smi = {}

    # for ind, lnp_id in enumerate(json_obj):
    for mol_id, smi in enumerate(smi_list):
        smi2mol_id[smi] = mol_id
        mol_id2smi[mol_id] = smi

    lnp2data_w_smi2mol_id = partial(lnp2data, smi2mol_id) # smi2mol_id is the dict to map smi to mol_id in mol.lmdb

    # use index of smi_list as pointer in NP data split and as key to access mol data in mol.lmdb
    # each row in train.lmdb correspond to a NP sample, with its a) components' i) mol_id, ii) percent and b) label
    def get_train_test_split_with_heldout_topbottom(dataset_json_list, target_label_name, bottom_heldout_ratio, top_heldout_ratio, remaining_random_test_ratio):
        # print("get_train_test_split_with_heldout_topbottom RUN!")
        lnp_label_values = []
        dataset_sz = len(dataset_json_list)
        for i, lnp_obj in enumerate(dataset_json_list):
            lnp_label_dict = lnp_obj['labels']
            if len(list(lnp_label_dict.keys())) > 1 and target_label_name not in list(lnp_label_dict.keys()):
                raise ValueError("target_label_name not in label names")
            elif len(list(lnp_label_dict.keys())) == 1:
                target_label_name = list(lnp_label_dict.keys())[0]
            lnp_label_value = lnp_label_dict[target_label_name]
            lnp_label_values.append(lnp_label_value)

        # sort dataset by label values, smallest to largest label value
        dataset_json_list = [x for _,x in sorted(zip(lnp_label_values, dataset_json_list), key = lambda y:y[0])]

        dataset_json_list_wo_heldout = dataset_json_list[int(dataset_sz*bottom_heldout_ratio):int(dataset_sz*(1-top_heldout_ratio))]
        top_heldout_set = dataset_json_list[int(dataset_sz*(1-top_heldout_ratio)):] # last top_heldout_ratio of dataset
        bottom_heldout_set = dataset_json_list[:int(dataset_sz*bottom_heldout_ratio)] # first bottom_heldout_ratio of dataset

        if remaining_random_test_ratio > 0:
            if shuffle:
                np.random.shuffle(dataset_json_list_wo_heldout)
            wo_heldout_dataset_sz = len(dataset_json_list_wo_heldout)
            random_test, train_valid = dataset_json_list_wo_heldout[:int(wo_heldout_dataset_sz*remaining_random_test_ratio)], dataset_json_list_wo_heldout[int(wo_heldout_dataset_sz*remaining_random_test_ratio):]
        else:
            random_test, train_valid = [], dataset_json_list_wo_heldout
        test = random_test + top_heldout_set + bottom_heldout_set

        return train_valid, test
    
    def subsample_train(train_set, random_train_subsample_ratio=None, train_subsample_sample_ids=None, subsample_target_label=None):
        if train_subsample_sample_ids != None:
            sampled_indices = []
            for i, train_sample in enumerate(train_set):
                if train_sample["lnp_id"] in train_subsample_sample_ids:
                    sampled_indices.append(i)

            if subsample_target_label != None: # subsample only the target_label to keep
                new_train_set = []
                for i, train_sample in enumerate(train_set):
                    if i in sampled_indices: # keep all labels
                        new_train_set.append(train_sample)
                    else: # remove subsample_target_label from this sample
                        train_sample['labels'].pop(subsample_target_label, None)
                        if len(train_sample['labels'].keys()) > 0:
                            new_train_set.append(train_sample)
                            
            train_set = new_train_set


        if random_train_subsample_ratio != None:
            # new_train_set = random.sample(train_set, int(len(train_set)*random_train_subsample_ratio))
            # new_train_set = random.sample(train_set, int(len(train_set)*random_train_subsample_ratio))
            train_size = len(train_set)
            sampled_indices = random.sample((range(train_size)), int(train_size*random_train_subsample_ratio))
            if subsample_target_label != None: # subsample only the target_label to keep
                new_train_set = []
                for i, train_sample in enumerate(train_set):
                    if i in sampled_indices: # keep all labels
                        new_train_set.append(train_sample)
                    else: # remove subsample_target_label from this sample
                        train_sample['labels'].pop(subsample_target_label, None)
                        if len(train_sample['labels'].keys()) > 0:
                            new_train_set.append(train_sample)

            else: # subsample all samples, including all labels
                new_train_set = [train_set[i] for i in sampled_indices]
            train_set = new_train_set

        return train_set

    def make_data_lmdb(train, valid, test, dataset_outpath, debug=False):
        for name, content_list in [('train.lmdb', train),
                                    ('valid.lmdb', valid),
                                    ('test.lmdb', test)]:
            
            os.makedirs(dataset_outpath, exist_ok=True)

            if debug:
                output_json_name = os.path.join(dataset_outpath, name.replace(".lmdb", ".json"))
                json_object = json.dumps(content_list, indent=4)
                with open(output_json_name, "w") as outfile:
                    outfile.write(json_object)

            output_name = os.path.join(dataset_outpath, name)
            try:
                os.remove(output_name)
            except:
                pass
            env_new = lmdb.open(
                output_name,
                subdir=False,
                readonly=False,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=int(100e9),
            )
            txn_write = env_new.begin(write=True)
            with Pool(nthreads) as pool:
                i = 0
                for inner_output in tqdm(pool.imap(lnp2data_w_smi2mol_id, content_list)):
                    if inner_output is not None:
                        txn_write.put(f'{i}'.encode("ascii"), inner_output)
                        i += 1
                print('{} process {} lines'.format(name, i))
                txn_write.commit()
                env_new.close()
    # make_data_lmdb end

    def filter_target_label(data_list, label_name):
        new_data_list = []
        for data_sample in data_list:
            if label_name in data_sample['labels'].keys():
                if len(data_sample['labels'].keys()) > 1:
                    new_data_sample = copy.deepcopy(data_sample)
                    new_data_sample['labels'] = {label_name: data_sample['labels'][label_name]}
                    new_data_list.append(new_data_sample)
                else:
                    new_data_list.append(data_sample)

        return new_data_list
    


    for dataset_name in dataset_dict:
        dataset_json_list = dataset_dict[dataset_name]
            
        # Shuffle json_obj
        # TODO NOW: shuffle dataset indices here to get different folds for train/valid/test splits
        if not fixed_train_valid_test_split: # retain the same order of samples for fixed train/valid/test split
            if shuffle:
                np.random.shuffle(dataset_json_list)
        dataset_sz = len(dataset_json_list)

        if kfold_valid == None: # normal train/valid/test split
            dataset_outpath = os.path.join(outpath, dataset_name)
            if valid_ratio == None:
                valid_ratio = test_ratio

            print("fixed_train_valid_test_split B: ", fixed_train_valid_test_split)
            if fixed_train_valid_test_split:
                dataset_json_lnp_id2obj = {}
                for lnp_obj in dataset_json_list:
                    dataset_json_lnp_id2obj[lnp_obj['lnp_id']] = lnp_obj
                train, valid, test = [], [], []
                for split, split_lnp_ids in [(train, train_lnp_ids), (valid, valid_lnp_ids), (test, test_lnp_ids)]:
                    for lnp_id in split_lnp_ids:
                        lnp_obj_to_add = dataset_json_lnp_id2obj[lnp_id]
                        split.append(lnp_obj_to_add)
                print("fixed_train_valid_test_split, train len: ", len(train))
                print("fixed_train_valid_test_split, valid len: ", len(valid))
                print("fixed_train_valid_test_split, test len: ", len(test))
            else:       
                # hold out top and bottom subset of the dataset - START-
                if top_bottom_ratio > 0:
                    train_valid, test = get_train_test_split_with_heldout_topbottom(dataset_json_list, target_label_name, bottom_heldout_ratio, top_heldout_ratio, remaining_random_test_ratio)
                    train, valid = train_valid[:int(dataset_sz*(1-test_ratio-valid_ratio))], train_valid[int(dataset_sz*(1-test_ratio-valid_ratio)):]

                else:
                    train, valid, test = dataset_json_list[:int(dataset_sz*(1-test_ratio-valid_ratio))], dataset_json_list[int(dataset_sz*(1-test_ratio-valid_ratio)):int(dataset_sz*(1-test_ratio))], dataset_json_list[int(dataset_sz*(1-test_ratio)):]

            train = subsample_train(train, random_train_subsample_ratio, train_subsample_sample_ids, subsample_target_label)

            if split_multilabel_train_sample:
                new_train = []
                for train_sample in train:
                    if len(train_sample['labels'].keys()) > 1:
                        for label_name in train_sample['labels'].keys():
                            new_train_sample = copy.deepcopy(train_sample)
                            new_train_sample['labels'] = {label_name: train_sample['labels'][label_name]}
                            new_train.append(new_train_sample)
                    else:
                        new_train.append(train_sample)
                train = new_train
            
            if type(labels_to_split_into_subfolders) == list: # ['in_house_lnp_DC24_luc', 'in_house_lnp_B16F10_luc']
                for label_name in labels_to_split_into_subfolders:
                    label_dataset_name = label_name
                    label_dataset_outpath = os.path.join(outpath, label_dataset_name)

                    label_train = filter_target_label(train, label_name)
                    label_valid = filter_target_label(valid, label_name)
                    label_test = filter_target_label(test, label_name)

                    make_data_lmdb(label_train, label_valid, label_test, label_dataset_outpath, debug)

            else:
                make_data_lmdb(train, valid, test, dataset_outpath, debug)
            

            output_train_lnp_ids = [lnp_obj['lnp_id'] for lnp_obj in train]
            output_valid_lnp_ids = [lnp_obj['lnp_id'] for lnp_obj in valid]
            output_test_lnp_ids = [lnp_obj['lnp_id'] for lnp_obj in test]

            return output_train_lnp_ids, output_valid_lnp_ids, output_test_lnp_ids

        else: # k-fold cross validation split
            fold_dir_paths = []

            # hold out top and bottom subset of the dataset - START-
            if top_bottom_ratio > 0:
                train_valid, test = get_train_test_split_with_heldout_topbottom(dataset_json_list, target_label_name, bottom_heldout_ratio, top_heldout_ratio, remaining_random_test_ratio)
            else:
                train_valid, test = dataset_json_list[:int(dataset_sz*(1-test_ratio))], dataset_json_list[int(dataset_sz*(1-test_ratio)):] # the test set is the same for all k-fold CV splits

            output_test_lnp_ids = [lnp_obj['lnp_id'] for lnp_obj in test]

            print("test set len: ", len(test))
            kf_train = KFold(n_splits=kfold_valid, shuffle=shuffle, random_state=seed)
            kfold_output_train_lnp_ids = {}
            kfold_output_valid_lnp_ids = {}
            for i_valid_fold, (train_index, valid_index) in enumerate(kf_train.split(train_valid)):
                
                fold_subdir_name = "fold_V" + str(i_valid_fold)
                fold_subdir_outpath = os.path.join(outpath, fold_subdir_name)
                os.makedirs(fold_subdir_outpath, exist_ok=True)

                print("i_valid_fold: ", i_valid_fold)
                print("train_index len: ", len(train_index))
                print("valid_index len: ", len(valid_index))
                train = list(np.array(train_valid)[train_index])
                valid = list(np.array(train_valid)[valid_index])

                # fold_dir_path = os.path.join(outpath, "fold_V" + str(i_valid_fold))
                if fold_subdir_outpath not in fold_dir_paths:
                    fold_dir_paths.append(fold_subdir_outpath)

                dataset_outpath = os.path.join(fold_subdir_outpath, dataset_name)

                train = subsample_train(train, random_train_subsample_ratio, train_subsample_sample_ids, subsample_target_label)

                if split_multilabel_train_sample:
                    new_train = []
                    for train_sample in train:
                        if len(train_sample['labels'].keys()) > 1:
                            for label_name in train_sample['labels'].keys():
                                new_train_sample = copy.deepcopy(train_sample)
                                new_train_sample['labels'] = {label_name: train_sample['labels'][label_name]}
                                new_train.append(new_train_sample)
                        else:
                            new_train.append(train_sample)
                    train = new_train
                
                if type(labels_to_split_into_subfolders) == list: # ['in_house_lnp_DC24_luc', 'in_house_lnp_B16F10_luc']
                    for label_name in labels_to_split_into_subfolders:
                        label_dataset_name = label_name
                        label_dataset_outpath = os.path.join(fold_subdir_outpath, label_dataset_name)

                        label_train = filter_target_label(train, label_name)
                        label_valid = filter_target_label(valid, label_name)
                        label_test = filter_target_label(test, label_name)

                        print("label_dataset_outpath: ", label_dataset_outpath)
                        make_data_lmdb(label_train, label_valid, label_test, label_dataset_outpath, debug)

                else:
                    make_data_lmdb(train, valid, test, dataset_outpath, debug)

                fold_output_train_lnp_ids = [lnp_obj['lnp_id'] for lnp_obj in train]
                kfold_output_train_lnp_ids[fold_subdir_name] = fold_output_train_lnp_ids
                fold_output_valid_lnp_ids = [lnp_obj['lnp_id'] for lnp_obj in valid]
                kfold_output_valid_lnp_ids[fold_subdir_name] = fold_output_valid_lnp_ids

            # copy mol.lmdb to each fold dir
            for fold_dir_path in fold_dir_paths:
                shutil.copy(mol_output_name, fold_dir_path)

            return kfold_output_train_lnp_ids, kfold_output_valid_lnp_ids, output_test_lnp_ids


# ============================================================================
# LNPDB heart/kidney dataset generation
# ----------------------------------------------------------------------------
# Converts experiments/data_json/LNPDB_heart_kidney.json into k-fold LMDB datasets under
#   processed_data_dirs/lnpdb_heartkidney_gen/fold_V{0..4}/lnpdb/{train,valid,test}.lmdb
# plus a shared mol.lmdb (3D RDKit conformers) copied into each fold dir.
#
# Run headless from the experiments/ directory:
#   python preprocess_data_LNPDB.py
#
# NOTE: conformer generation is the slow part. Bump --nthreads to the number
# of physical cores available.
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", default="data_json/LNPDB_heart_kidney.json")
    parser.add_argument("--outpath", default="processed_data_dirs/lnpdb_heartkidney_gen")
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument("--kfold-valid", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--no-debug", action="store_true",
                        help="skip writing the human-readable .json alongside each .lmdb")
    args = parser.parse_args()

    train_ids, valid_ids, test_ids = write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=args.inpath,
        outpath=args.outpath,
        nthreads=args.nthreads,
        kfold_valid=args.kfold_valid,
        test_ratio=args.test_ratio,
        top_heldout_ratio=0,
        bottom_heldout_ratio=0,
        shuffle=True,
        debug=not args.no_debug,
    )
    print("Done. Folds written under:", args.outpath)
    print("Test set size:", len(test_ids))
