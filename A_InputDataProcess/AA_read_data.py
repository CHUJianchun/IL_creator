import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = '1'
import json
import pickle
import numpy as np
from tqdm import tqdm, trange
from past.builtins import range
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import LinearRegression

from Util_mol.molecule_tree import annotated_tree_to_mol_tree
from Util_mol.params import DECISION_DIM
from Model.attribute_tree_decoder import create_tree_decoder
from Model.decoder import make_att_masks
from Model.tree_walker import OnehotBuilder
import Util_cfg.cfg_parser as parser
from A_InputDataProcess.bad_smiles import bad_smiles


grammar = parser.Grammar()
walker = OnehotBuilder()
tree_decoder = create_tree_decoder()
co2_x = 44


class DataPoint:
    def __init__(self, temperature_, pressure_, weight_fraction_, ionic_liquid_):
        self.temperature = temperature_
        self.pressure = pressure_
        self.weight_fraction = weight_fraction_
        self.ionic_liquid = ionic_liquid_


class Anion:
    def __init__(self, smiles_, masks_=None, onehot_=None):
        self.smiles = smiles_
        if onehot_ is None or masks_ is None:
            n = annotated_tree_to_mol_tree(parser.parse(self.smiles, grammar)[0])
            self.onehot, self.masks = make_att_masks(n, tree_decoder, walker)
        else:
            self.onehot = onehot_
            self.masks = masks_


class Cation:
    def __init__(self, smiles_, masks_=None, onehot_=None):
        self.smiles = smiles_
        if onehot_ is None or masks_ is None:
            n = annotated_tree_to_mol_tree(parser.parse(self.smiles, grammar)[0])
            self.onehot, self.masks = make_att_masks(n, tree_decoder, walker)
        else:
            self.onehot = onehot_
            self.masks = masks_


class IonicLiquid:
    def __init__(self, anion_, cation_):
        self.anion = anion_
        self.cation = cation_


class DataFrame:
    def __init__(self, il_, data_frame_=None):
        if data_frame_ is None:
            self.data_frame = []
        else:
            self.data_frame = data_frame_
        self.il = il_


class GPDataset:
    def __init__(self, il_, s_333, s_383, s_diff):
        self.il = il_
        self.s_333 = s_333
        self.s_383 = s_383
        self.s_diff = s_diff


def unpickle_gp_dataset():
    gpd_dataset_list_ = []
    with open('InputData/gp_dataset_list.data', 'rb') as f_:
        pickled = pickle.load(f_)
    for gpd in pickled:
        anion = Anion(smiles_=gpd[0], masks_=gpd[1], onehot_=gpd[2])
        cation = Cation(smiles_=gpd[3], masks_=gpd[4], onehot_=gpd[5])
        il = IonicLiquid(anion_=anion, cation_=cation)
        gpd_dataset_list_.append(GPDataset(il_=il, s_333=gpd[6], s_383=gpd[7], s_diff=gpd[8]))
    return gpd_dataset_list_


def pickle_gp_dataset(gp_dataset_list_):
    pickled_gp_dataset_list_ = []
    for gpd in gp_dataset_list_:
        pickled_gp_dataset_list_.append([gpd.il.anion.smiles, gpd.il.anion.masks, gpd.il.anion.onehot,
                                         gpd.il.cation.smiles, gpd.il.cation.masks, gpd.il.cation.onehot,
                                         gpd.s_333, gpd.s_383, gpd.s_diff])
    with open('InputData/gp_dataset_list.data', 'wb') as f_:
        pickle.dump(pickled_gp_dataset_list_, f_)


def unpickle_ionic_liquid_list():
    with open('InputData/ionic_liquid_list.data', 'rb') as f_:
        pickled = pickle.load(f_)
    ionic_liquid_list_ = []
    for il in tqdm(pickled):
        ionic_liquid_list_.append(IonicLiquid(anion_=Anion(il[0], il[2], il[3]), cation_=Cation(il[1], il[4], il[5])))
    return ionic_liquid_list_


def pickle_ionic_liquid_list(ionic_liquid_list_):
    pickled_ionic_liquid_list_ = []
    for ionic_liquid in ionic_liquid_list_:
        pickled_ionic_liquid_list_.append(
            [ionic_liquid.anion.smiles, ionic_liquid.cation.smiles, ionic_liquid.anion.masks, ionic_liquid.anion.onehot,
             ionic_liquid.cation.masks, ionic_liquid.cation.onehot])
    with open('InputData/ionic_liquid_list.data', 'wb') as f_:
        pickle.dump(pickled_ionic_liquid_list_, f_)


def unpickle_ion_list(ion_type):
    ion_list_ = []
    if ion_type == 'Anion':
        with open('InputData/anion_list.data', 'rb') as f_:
            pickled = pickle.load(f_)
        for ion in pickled:
            ion_list_.append(Anion(ion[0], ion[1], ion[2]))
    elif ion_type == 'Cation':
        with open('InputData/cation_list.data', 'rb') as f_:
            pickled = pickle.load(f_)
        for ion in pickled:
            ion_list_.append(Cation(ion[0], ion[1], ion[2]))
    return ion_list_


def pickle_ion_list(ion_type, ion_list):
    pickled_ion_list_ = []
    for ion in ion_list:
        pickled_ion_list_.append([ion.smiles, ion.masks, ion.onehot])
    if ion_type == 'Anion':
        with open('InputData/anion_list.data', 'wb') as f_:
            pickle.dump(pickled_ion_list_, f_)
    elif ion_type == 'Cation':
        with open('InputData/cation_list.data', 'wb') as f_:
            pickle.dump(pickled_ion_list_, f_)


def unpickle_smiles_list(ion_type):
    ion_smiles_list = []
    ion_list = unpickle_ion_list(ion_type)
    for ion in ion_list:
        ion_smiles_list.append(ion.smiles)
    return ion_smiles_list


def unpickle_onehot_list(ion_type):
    ion_onehot_list = []
    ion_list = unpickle_ion_list(ion_type)
    for ion in ion_list:
        ion_onehot_list.append(ion.onehot)
    return np.array(ion_onehot_list)


def unpickle_data_point_list():
    data_point_list_ = []
    with open('InputData/data_point_list.data', 'rb') as f_:
        pickled = pickle.load(f_)
    for data_point in pickled:
        temperature = data_point[0]
        pressure = data_point[1]
        weight_fraction = data_point[2]
        anion_ = Anion(smiles_=data_point[3], masks_=data_point[4], onehot_=data_point[5])
        cation_ = Cation(smiles_=data_point[6], masks_=data_point[7], onehot_=data_point[8])
        ionic_liquid_ = IonicLiquid(anion_, cation_)
        unpickle_data_point = DataPoint(temperature, pressure, weight_fraction, ionic_liquid_)
        data_point_list_.append(unpickle_data_point)
    return data_point_list_


def pickle_data_point_list(data_point_list_):
    pickled_data_point_list_ = []
    for data_point in data_point_list_:
        pickled_data_point_list_.append([data_point.temperature, data_point.pressure, data_point.weight_fraction,
                                         data_point.ionic_liquid.anion.smiles,
                                         data_point.ionic_liquid.anion.masks, data_point.ionic_liquid.anion.onehot,
                                         data_point.ionic_liquid.cation.smiles,
                                         data_point.ionic_liquid.cation.masks, data_point.ionic_liquid.cation.onehot])
    with open('InputData/data_point_list.data', 'wb') as f_:
        pickle.dump(pickled_data_point_list_, f_)


def unpickle_data_frame_list():
    data_frame_list_ = []
    with open('InputData/data_frame_list.data', 'rb') as f_:
        pickled = pickle.load(f_)
    for data_frame in pickled:
        anion_ = Anion(smiles_=data_frame[0], masks_=data_frame[1], onehot_=data_frame[2])
        cation_ = Cation(smiles_=data_frame[3], masks_=data_frame[4], onehot_=data_frame[5])
        il_ = IonicLiquid(anion_=anion_, cation_=cation_)
        data_frame_ = data_frame[6]
        data_frame_list_.append(DataFrame(il_=il_, data_frame_=data_frame_))
    return data_frame_list_


def pickle_data_frame_list(data_frame_list_):
    pickled_data_frame_list_ = []
    for data_frame_ in data_frame_list_:
        pickled_data_frame_list_.append(
            [data_frame_.il.anion.smiles, data_frame_.il.anion.masks, data_frame_.il.anion.onehot,
             data_frame_.il.cation.smiles, data_frame_.il.cation.masks, data_frame_.il.cation.onehot,
             data_frame_.data_frame])
    with open('InputData/data_frame_list.data', 'wb') as f_:
        pickle.dump(pickled_data_frame_list_, f_)


def load_data_origin():
    print('Start: Loading origin data from Data/origin_data_list.data')
    try:
        with open('InputData/origin_data_list.data', 'rb') as f_:
            data_list__ = pickle.load(f_)
    except IOError:
        print('Warning: File origin_data_list.data not found, reinitializing')
        try:
            with open('InputData/data_2021Jul.txt') as f_:
                data_list_ = json.loads(f_.read())
        except IOError:
            print('Error: File InputData/data_2021Jul.txt not found')
            sys.exit()
        else:
            with open('InputData/origin_data_list.data', 'wb') as f_:
                pickle.dump(data_list_, f_)
                print('Finish: Saving origin data from Data/data_2021Jul.txt')
        with open('InputData/origin_data_list.data', 'rb') as f_:
            data_list__ = pickle.load(f_)
    print('Finish: Loading origin data from Data/origin_data_list.data')

    return data_list__


def name_to_il(name_smiles_list_, ionic_liquid_list_, name):
    for il in name_smiles_list_:
        if name == il[0]:
            for il_graph in ionic_liquid_list_:
                if il_graph.anion.smiles == il[1] and il_graph.cation.smiles == il[2]:
                    return il_graph
    return -1


def load_ionic_liquid_list():

    data_list_ = load_data_origin()
    component_name_list = []

    for data in data_list_:
        for component in data[1]['components']:
            if component['name'] not in component_name_list:
                component_name_list.append(component['name'])

    with open('InputData/component_name_list.txt', 'w') as f_:
        for item in component_name_list:
            f_.write(item + '\n')
    os.system('java -jar A_InputDataProcess/opsin.jar -osmi InputData/component_name_list.txt '
              'InputData/component_smiles_list.txt')

    name_smiles_list_ = []
    ionic_liquid_list_ = []
    anion_smiles_list = []
    cation_smiles_list = []
    with open('InputData/component_smiles_list.txt', 'r') as f_:
        component_smiles_list = f_.readlines()
    with open('InputData/component_name_list.txt', 'r') as f_:
        component_name_list = f_.readlines()
    for i_ in trange(len(component_name_list)):
        component_name_list[i_] = component_name_list[i_].replace('\n', '')
        component_smiles_list[i_] = component_smiles_list[i_].replace('\n', '')

        if '.' in component_smiles_list[i_] \
                and component_smiles_list[i_].count('-') == 1 \
                and component_smiles_list[i_].count('+') == 1 \
                and 'Ga' not in component_smiles_list[i_] \
                and 'Re' not in component_smiles_list[i_] \
                and 'Al' not in component_smiles_list[i_] \
                and 'As' not in component_smiles_list[i_] \
                and 'Sb' not in component_smiles_list[i_] \
                and len(component_smiles_list[i_].split('.')[0]) > 6 \
                and len(component_smiles_list[i_].split('.')[1]) > 6:
            part_1 = component_smiles_list[i_].split('.')[0]
            part_2 = component_smiles_list[i_].split('.')[1]
            if '-' in part_1:
                if part_1 not in anion_smiles_list:
                    anion_smiles_list.append(part_1)
                if part_2 not in cation_smiles_list:
                    cation_smiles_list.append(part_2)
                name_smiles_list_.append([component_name_list[i_], part_1, part_2])
            else:
                if part_1 not in cation_smiles_list:
                    cation_smiles_list.append(part_1)
                if part_2 not in anion_smiles_list:
                    anion_smiles_list.append(part_2)
                name_smiles_list_.append([component_name_list[i_], part_2, part_1])

    with open('InputData/name_smiles_list.data', 'wb') as f__:
        pickle.dump(name_smiles_list_, f__)

    print("Notice: Totally %d anions and %d cations to be added to ILs list" % (
        len(anion_smiles_list), len(cation_smiles_list)))

    anion_graph_list = []
    cation_graph_list = []
    anion_sum = 0

    for anion_smiles in anion_smiles_list:
        if anion_smiles not in bad_smiles:
            a = Anion(anion_smiles)
            anion_sum += 1
            anion_graph_list.append(a)
            print('Done generation on No.' + str(anion_sum) + '    ' + anion_smiles)
        else:
            print('Pass a bad smiles')

    for cation_smiles in cation_smiles_list:
        if cation_smiles not in bad_smiles:
            c = Cation(cation_smiles)
            cation_graph_list.append(c)
            print('Done generation on No.' + str(cation_smiles_list.index(cation_smiles)) + '    ' + cation_smiles)
        else:
            print('Pass a bad smiles on No.' + str(cation_smiles_list.index(cation_smiles)))

    for anion_graph in anion_graph_list:
        for cation_graph in cation_graph_list:
            ionic_liquid_list_.append(IonicLiquid(anion_=anion_graph, cation_=cation_graph))

    pickle_ion_list('Anion', anion_graph_list)
    pickle_ion_list('Cation', cation_graph_list)
    pickle_ionic_liquid_list(ionic_liquid_list_)
    return ionic_liquid_list_


def load_data_classified():
    print('Start: Classifying origin data')
    data_list_ = load_data_origin()

    with open('InputData/name_smiles_list.data', 'rb') as f_:
        name_smiles_list_ = pickle.load(f_)
    ionic_liquid_list_ = unpickle_ionic_liquid_list()

    equilibrium_pressure_list = []
    weight_fraction_list = []
    henry_constant_mole_fraction_list = []
    mole_fraction_list = []
    for data_ in data_list_:
        if len(data_[1]['components']) == 2 and data_[1]['solvent'] is None:
            if data_[1]['components'][0]['formula'] == 'CO<SUB>2</SUB>' or data_[1]['components'][1]['formula'] == 'CO<SUB>2</SUB>':
                if data_[1]['title'] == 'Phase transition properties: Equilibrium pressure':  # used
                    equilibrium_pressure_list.append(data_)
                elif data_[1]['title'] == 'Composition at phase equilibrium: Henry\'s Law constant for mole fraction of component':  # used
                    henry_constant_mole_fraction_list.append(data_)
                elif data_[1]['title'] == 'Composition at phase equilibrium: Weight fraction of component':  # used
                    weight_fraction_list.append(data_)
                elif data_[1]['title'] == 'Composition at phase equilibrium: Mole fraction of component':  # used
                    mole_fraction_list.append(data_)
    data_point_list_ = []
    data_frame_list_ = []

    for data_ in mole_fraction_list:
        ionic_liquid_name_ = None
        for component in data_[1]['components']:
            if component['name'] != 'carbon dioxide':
                ionic_liquid_name_ = component['name']
                break
        try:
            assert ionic_liquid_name_ is not None
        except AssertionError:
            print('Data structure error at henry_constant_mole_fraction_list index'
                  + str(equilibrium_pressure_list.index(data_)))
            continue

        ionic_liquid_ = name_to_il(name_smiles_list_, ionic_liquid_list_, ionic_liquid_name_)
        if ionic_liquid_ == -1:
            continue

        if any('Mole fraction of carbon dioxide' in element_ for element_ in data_[1]['dhead']):
            temperature_index, mole_fraction_index, pressure_index = None, None, None
            for i in range(len(data_[1]['dhead'])):
                try:
                    if 'Temperature' in data_[1]['dhead'][i][0]:
                        temperature_index = i
                    elif 'Mole fraction of carbon dioxide' in data_[1]['dhead'][i][0]:
                        mole_fraction_index = i
                    elif 'Pressure' in data_[1]['dhead'][i][0] and 'kPa' in data_[1]['dhead'][i][0]:
                        pressure_index = i
                except TypeError:
                    print(data_[1]['dhead'])

            try:
                assert temperature_index is not None \
                       and mole_fraction_index is not None \
                       and pressure_index is not None \
                       and ionic_liquid_name_ is not None
            except AssertionError:
                print(data_[1]['dhead'])
                # print('Data structure error at equilibrium_pressure_list index' + str(equilibrium_pressure_list.index(data_)))
                continue

            data_frame = DataFrame(il_=ionic_liquid_)
            il_mol = Chem.MolFromSmiles(ionic_liquid_.cation.smiles + '.' + ionic_liquid_.anion.smiles)
            il_x = Descriptors.MolWt(il_mol)
            for point in data_[1]['data']:
                temperature_ = float(point[temperature_index][0])
                pressure_ = float(point[pressure_index][0])
                mole_fraction_ = float(point[mole_fraction_index][0])
                weight_fraction_ = co2_x * mole_fraction_ / (co2_x * mole_fraction_ + il_x * (1 - mole_fraction_))
                data_point_ = DataPoint(
                    temperature_=temperature_,
                    pressure_=pressure_,
                    weight_fraction_=weight_fraction_,
                    ionic_liquid_=ionic_liquid_)
                data_point_list_.append(data_point_)
                data_frame.data_frame.append([temperature_, pressure_, weight_fraction_])
            data_frame_list_.append(data_frame)
    ###
    for data_ in equilibrium_pressure_list:
        ionic_liquid_name_ = None
        for component in data_[1]['components']:
            if component['name'] != 'carbon dioxide':
                ionic_liquid_name_ = component['name']
                break
        try:
            assert ionic_liquid_name_ is not None
        except AssertionError:
            print('Data structure error at henry_constant_mole_fraction_list index'
                  + str(equilibrium_pressure_list.index(data_)))
            continue

        ionic_liquid_ = name_to_il(name_smiles_list_, ionic_liquid_list_, ionic_liquid_name_)
        if ionic_liquid_ == -1:
            continue

        if any('Mole fraction of carbon dioxide' in element_ for element_ in data_[1]['dhead']):
            temperature_index, mole_fraction_index, pressure_index = None, None, None
            for i in range(len(data_[1]['dhead'])):
                if any('Temperature' in dhead for dhead in data_[1]['dhead'][i]):
                    temperature_index = i
                elif any('Mole fraction of carbon dioxide' in dhead for dhead in data_[1]['dhead'][i]):
                    mole_fraction_index = i
                elif any('Equilibrium pressure' in dhead for dhead in data_[1]['dhead'][i]) and \
                        any('kPa' in dhead for dhead in data_[1]['dhead'][i]):
                    pressure_index = i

            try:
                assert temperature_index is not None \
                       and mole_fraction_index is not None \
                       and pressure_index is not None \
                       and ionic_liquid_name_ is not None
            except AssertionError:
                print(data_[1]['dhead'])
                print('Data structure error at equilibrium_pressure_list index'
                      + str(equilibrium_pressure_list.index(data_)))
                continue

            data_frame = DataFrame(il_=ionic_liquid_)
            il_mol = Chem.MolFromSmiles(ionic_liquid_.cation.smiles + '.' + ionic_liquid_.anion.smiles)
            il_x = Descriptors.MolWt(il_mol)
            for point in data_[1]['data']:
                temperature_ = float(point[temperature_index][0])
                pressure_ = float(point[pressure_index][0])
                mole_fraction_ = float(point[mole_fraction_index][0])
                weight_fraction_ = co2_x * mole_fraction_ / (co2_x * mole_fraction_ + il_x * (1 - mole_fraction_))
                data_point_ = DataPoint(
                    temperature_=temperature_,
                    pressure_=pressure_,
                    weight_fraction_=weight_fraction_,
                    ionic_liquid_=ionic_liquid_)
                data_point_list_.append(data_point_)
                data_frame.data_frame.append([temperature_, pressure_, weight_fraction_])
            data_frame_list_.append(data_frame)
    
    for data_ in henry_constant_mole_fraction_list:
        ionic_liquid_name_ = None
        for component in data_[1]['components']:
            if component['name'] != 'carbon dioxide':
                ionic_liquid_name_ = component['name']
                break
        try:
            assert ionic_liquid_name_ is not None
        except AssertionError:
            print('Data structure error at henry_constant_mole_fraction_list index'
                  + str(equilibrium_pressure_list.index(data_)))
            continue
        ionic_liquid_ = name_to_il(name_smiles_list_, ionic_liquid_list_, ionic_liquid_name_)
        if ionic_liquid_ == -1:
            continue
        data_frame = DataFrame(il_=ionic_liquid_)
        il_mol = Chem.MolFromSmiles(ionic_liquid_.cation.smiles + '.' + ionic_liquid_.anion.smiles)
        il_x = Descriptors.MolWt(il_mol)
        for point in data_[1]['data']:
            temperature_ = float(point[0][0])
            pressure_ = 101.325
            mole_fraction_ = pressure_ / float(point[2][0])
            if mole_fraction_ > 1.1:
                continue
            weight_fraction_ = co2_x * mole_fraction_ / (co2_x * mole_fraction_ + il_x * (1 - mole_fraction_))
            data_point_ = DataPoint(
                temperature_=temperature_,
                pressure_=pressure_,
                weight_fraction_=weight_fraction_,
                ionic_liquid_=ionic_liquid_)
            data_point_list_.append(data_point_)
            data_frame.data_frame.append([temperature_, pressure_, weight_fraction_])
        data_frame_list_.append(data_frame)

    for data_ in weight_fraction_list:
        ionic_liquid_name_ = None
        for component in data_[1]['components']:
            if component['name'] != 'carbon dioxide':
                ionic_liquid_name_ = component['name']
                break
        try:
            assert ionic_liquid_name_ is not None
        except AssertionError:
            print('Data structure error at henry_constant_mole_fraction_list index'
                  + str(equilibrium_pressure_list.index(data_)))
            continue
        ionic_liquid_ = name_to_il(name_smiles_list_, ionic_liquid_list_, ionic_liquid_name_)
        if ionic_liquid_ == -1:
            continue
        data_frame = DataFrame(il_=ionic_liquid_)
        for point in data_[1]['data']:
            temperature_ = float(point[0][0])
            pressure_ = float(point[1][0])
            weight_fraction_ = float(point[2][0])
            data_point_ = DataPoint(
                temperature_=temperature_,
                pressure_=pressure_,
                weight_fraction_=weight_fraction_,
                ionic_liquid_=ionic_liquid_)
            data_point_list_.append(data_point_)
            data_frame.data_frame.append([temperature_, pressure_, weight_fraction_])
        data_frame_list_.append(data_frame)

    pickle_data_point_list(data_point_list_)
    pickle_data_frame_list(data_frame_list_)
    return data_point_list_, data_frame_list_


def regress_il_property():
    data_frame_list_ = unpickle_data_frame_list()
    gp_dataset_list_ = []
    for df in data_frame_list_:
        t_list = []
        p_list = []
        p_log_list = []
        prop_list = []

        for dp in df.data_frame:
            t_list.append(dp[0])
            p_list.append(dp[1])
            p_log_list.append(dp[1] ** 2)
            prop_list.append(dp[2])

        feature_list = np.array([t_list, p_list, p_log_list]).T
        t_list = np.array(t_list)
        p_list = np.array(p_list)
        prop_list = np.array(prop_list)
        if len(t_list) > 5:
            if np.max(t_list) - np.min(t_list) >= 40:
                condition_1 = np.min(p_list - 101.) < 1000 and np.max(p_list) - np.min(p_list) > 500
                condition_2 = np.min(p_list - 101.) >= 1000 and np.max(p_list) - np.min(p_list) > 1000
                if condition_2 or condition_1:
                    model = LinearRegression()
                    model.fit(feature_list, prop_list)
                    solubility_333 = model.predict(np.array((333, 101, 101 ** 2)).reshape(1, -1))
                    solubility_383 = model.predict(np.array((383, 101, 101 ** 2)).reshape(1, -1))
                    solubility_diff = solubility_333 - solubility_383
                    print('The R is %.3f for %s . %s' % (model.score(feature_list, prop_list),
                                                         df.il.anion.smiles, df.il.cation.smiles))
                    print('The AARD is %.3f for %s . %s' % (np.mean(
                        np.abs(model.predict(feature_list)-prop_list)/prop_list),
                                                         df.il.anion.smiles, df.il.cation.smiles))
                    gp_dataset_list_.append(GPDataset(il_=df.il,
                                                      s_333=solubility_333, s_383=solubility_383,
                                                      s_diff=solubility_diff))
    # pickle_gp_dataset(gp_dataset_list_)
    return gp_dataset_list_


if __name__ == '__main__':
    # ionic_liquid_list = load_ionic_liquid_list()
    # data_point_list, data_frame_list = load_data_classified()
    gp_dataset_list = regress_il_property()
    # with open('InputData/data_point_list.data', 'rb') as f:
    #     data_point_list = pickle.load(f)
    # with open('InputData/name_smiles_list.data', 'rb') as f:
    #     name_smiles_list = pickle.load(f)
    # with open('InputData/origin_data_list.data', 'rb') as f:
    #     origin_data_list = pickle.load(f)
    pass
