
class Config(object):
    
    def __init__(self, smiles_format=2, max_fp_features=2048, mode='threshold', topn_fp_features=5, threshold=0.3, morgan_radius=2):
        # define the file path
        self.LA_sdf_path = './dataset/LA'
        self.LA_smiles_file_old = './dataset/smiles_LA.csv'
        self.LA_smiles_file = './dataset/smiles_LA-revised.csv'
        self.zinc_file = './dataset/dataset_v1.csv'
        self.mixed_dataset = './dataset/dataset_v2.csv'
        self.mixed_train_dataset = './dataset/train_dataset.csv'
        self.SOM_smiles_file = './dataset/smiles_SOM.csv' # including LA and zinc smiles molecules
        self.SOM_model = './models/som.p' # used to store som model
        self.FpsSOM_model = './models/fpssom.p' # used to store fps som model
        self.SOM_fps = './models/som_fps.npy'
        self.SOM_labels = './models/som_labels.npy'
        
        # smiles format when converting sdf into smiles, 
        # 0: default, 1: remove all the isomeric structures, i.e. isomericSmiles = False, 2: no @ characters
        self.smiles_format = smiles_format
        self.max_fp_features = max_fp_features
        
        # 方案一： 设置diff阈值； 方案二：设置保存的特征数，大的更容易保存
        self.mode = mode
        self.topn_fp_features = topn_fp_features
        self.threshold = threshold
        
        # the radius in morgan fingerprints
        self.morgan_radius = morgan_radius
