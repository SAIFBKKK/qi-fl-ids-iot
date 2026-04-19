@echo off
python train_hierarchical_level3_submodel.py --family ddos
python train_hierarchical_level3_submodel.py --family dos
python train_hierarchical_level3_submodel.py --family mirai
python train_hierarchical_level3_submodel.py --family recon
python train_hierarchical_level3_submodel.py --family spoofing
python train_hierarchical_level3_submodel.py --family web
