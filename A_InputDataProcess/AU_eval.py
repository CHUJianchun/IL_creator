from A_InputDataProcess.AA_read_data import *
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw


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
        if np.max(t_list) - np.min(t_list) >= 10:
            if np.max(p_list) - np.min(p_list) > 500:
                model = LinearRegression()
                model.fit(feature_list, prop_list)
                solubility_333 = model.predict(np.array((333, 993, 993 ** 2)).reshape(1, -1))
                solubility_383 = model.predict(np.array((363, 102.8, 102.8 ** 2)).reshape(1, -1))
                solubility_diff = solubility_333 - solubility_383
                aard = np.mean(np.abs(model.predict(feature_list) - prop_list) / prop_list)
                # print('The R is %.3f for %s . %s' % (model.score(feature_list, prop_list), df.il.anion.smiles, df.il.cation.smiles))
                # print('The AARD is %.3f for %s . %s' % (aard, df.il.anion.smiles, df.il.cation.smiles))

                # print(data_frame_list_.index(df))
                gp_dataset_list_.append([GPDataset(il_=df.il,
                                                   s_333=solubility_333, s_383=solubility_383,
                                                   s_diff=solubility_diff), aard])

sdiff_list = []
smiles_list = []
sdiff_list = []
for x in gp_dataset_list_:
    data = x[0]
    sdiff_list.append(data.s_diff)
    smiles_list.append(data.il.anion.smiles + '.' + data.il.cation.smiles)
sdiff_list = np.array(sdiff_list)
print(sdiff_list.max())
