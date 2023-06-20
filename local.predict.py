import pandas as pd
from rdkit import Chem, DataStructs
import time, os
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

def create_feature(data):
    fp_n = 1024
    fp_cols = ['FP'+str(i) for i in range(fp_n)]
    mol_cols = ["NumAtoms", "NumBonds", "NumHeavyAtoms", "NumHeteroatoms", \
                "NumRotatableBonds", "NumAromaticRings", "NumSaturatedRings"]
    fps = []
    mols = []
    for smi in data["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mols.append([None]*len(mol_cols))
            fps.append([None]*fp_n)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits=fp_n) # 修改为ECFP
            t = np.zeros((1,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, t) 
            fps.append(t)
            num_heteroatoms = 0 if not mol.HasSubstructMatch(Chem.MolFromSmarts("[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]")) else \
                len(mol.GetSubstructMatches(Chem.MolFromSmarts("[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]")))
            mols.append([mol.GetNumAtoms(), mol.GetNumBonds(), \
                        mol.GetNumHeavyAtoms(), num_heteroatoms, \
                        AllChem.CalcNumRotatableBonds(mol), \
                        len(Chem.GetSymmSSSR(mol, True)), \
                        len(Chem.GetSymmSSSR(mol, False))])

    #print(len(fps[0]))
    #print(len(fp_cols))
    fps = pd.DataFrame(fps, columns=fp_cols, dtype=int)
    mols = pd.DataFrame(mols, columns=mol_cols)
    features = pd.concat([fps, mols], axis=1)
    return features

if __name__ == '__main__':
    start_time = time.time()

    # 设置工作路径
    os.chdir('D:/OneDrive/代码开发/16 机器学习/admet_learn/')

    # admet种类界定
    type = 'toxity'

    # 定义物种名称对应的类别标签
    class_dict = {'Sus scrofa': 'pig', 
                'Bos taurus': 'cow', 
                'Gallus gallus': 'chicken', 
                'Danio rerio': 'fish', 
                'Ovis aries': 'sheep'}

    label = 'pig'

    # 加载保存的模型进行预测
    best_model = joblib.load(f'{label}_{type}_*_best_model.pkl')

    # 利用最优模型对未知化合物进行活性分类，并计算预测准确率score
    test_data = pd.read_csv("test.csv", usecols=["compound_id", "canonical_smiles"])
    test_data["canonical_smiles"] = test_data["canonical_smiles"].apply(lambda x: Chem.CanonSmiles(x))
    test_features = create_feature(test_data)
    test_data["class"] = le.inverse_transform(best_model.predict(test_features))

    # 将结果保存在result.csv中
    test_data.to_csv(f'{label}_{type}_result.csv', columns=["compound_id", "canonical_smiles", "class"], index=False)

end_time = time.time()
total_time = end_time - start_time
print(f"Total time elapsed: {total_time:.2f} seconds")


