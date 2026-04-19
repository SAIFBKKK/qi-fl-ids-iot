@echo off
python train_hierarchical_level3_submodel_v4.py --family ddos
python train_hierarchical_level3_submodel_v4.py --family dos
python train_hierarchical_level3_submodel_v4.py --family mirai
python train_hierarchical_level3_submodel_v4.py --family recon
python train_hierarchical_level3_submodel_v4.py --family spoofing
python train_hierarchical_level3_submodel_v4.py --family web
