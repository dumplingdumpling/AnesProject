## Setting the Environment

1. Install [RDKit](https://www.rdkit.org/docs/Install.html).
2. Install [GENTRL](https://github.com/insilicomedicine/GENTRL) model.
3. Install [MOSES](https://github.com/molecularsets/moses) from the repository

## Running Procedure
1. Run ``candi_molFingerprintsSOM.ipynb`` to train the SOM model for the Zinc and the dedicated dataset.
2. Run ``candi_pretrain.py`` to get the trained VAE model.
3. Run ``candi_sampling.py`` to generate candidate molecules from the trained model.


## Datasets:
1. [Zinc](https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv) dataset
2. Select a dataset in a targeted area.
