"""
run_hierarchical_pipeline_inference.py
=====================================
"""
import json, pickle
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
LABEL_COL="label"
ROOT_MODELS=Path(r"E:\dataset\processed_merged_full\hierarchical_final_v3_models")
ROOT_DATA=Path(r"E:\dataset\processed_merged_full\hierarchical_final_v3")
INPUT_CSV=Path(r"E:\dataset\processed_merged_full\minority_balancing_v3\training_ready\test.csv")
OUTPUT_DIR=Path(r"E:\dataset\processed_merged_full\hierarchical_final_inference"); OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
class MLP1(nn.Module):
    def __init__(self,input_dim,num_classes,hidden=64,dropout=0.20):
        super().__init__(); self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.BatchNorm1d(hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,num_classes))
    def forward(self,x): return self.net(x)
def load_model(model_dir:Path):
    feature_names=pickle.load(open(model_dir/"feature_names.pkl","rb")); label_map=pickle.load(open(model_dir/"label_mapping.pkl","rb")); model=MLP1(len(feature_names),len(label_map)); model.load_state_dict(torch.load(model_dir/"mlp_model_state.pt",map_location="cpu")); model.eval(); return model,feature_names,label_map
def predict_labels(model,X):
    with torch.no_grad(): return torch.argmax(model(torch.from_numpy(X.astype(np.float32))),dim=1).cpu().numpy()
def main():
    df=pd.read_csv(INPUT_CSV)
    l1_model,l1_features,l1_map=load_model(ROOT_MODELS/"level1_binary")
    X1=df[l1_features].to_numpy(dtype=np.float32); pred_l1=predict_labels(l1_model,X1)

    l2_model,l2_features,l2_map=load_model(ROOT_MODELS/"level2_family"); family_id_to_name={v:k for k,v in l2_map.items()}
    predictions=[]; predicted_family=[]
    for idx,row in df.iterrows():
        if pred_l1[idx]==0:
            predictions.append("BENIGN"); predicted_family.append("benign"); continue
        x2=row[l2_features].to_numpy(dtype=np.float32)[None,:]; fam_id=int(predict_labels(l2_model,x2)[0]); family=family_id_to_name[fam_id]; predicted_family.append(family)
        sub_data_dir=ROOT_DATA/"level3_family_submodels"/family; sub_model_dir=ROOT_MODELS/"level3_family_submodels"/family; sub_map=pickle.load(open(sub_data_dir/"label_mapping.pkl","rb"))
        if len(sub_map)==1:
            predictions.append(next(iter(sub_map.keys()))); continue
        sub_model,sub_features,sub_loaded_map=load_model(sub_model_dir); subtype_id_to_name={v:k for k,v in sub_loaded_map.items()}; x3=row[sub_features].to_numpy(dtype=np.float32)[None,:]; sub_id=int(predict_labels(sub_model,x3)[0]); predictions.append(subtype_id_to_name[sub_id])
    out=df.copy(); out["predicted_family"]=predicted_family; out["predicted_label"]=predictions; out.to_csv(OUTPUT_DIR/"hierarchical_predictions.csv",index=False)
    if LABEL_COL in df.columns:
        acc=float((out[LABEL_COL]==out["predicted_label"]).mean()); json.dump({"accuracy":acc,"n_rows":int(len(out))},open(OUTPUT_DIR/"summary.json","w"),indent=2)
    print("Saved:",OUTPUT_DIR/"hierarchical_predictions.csv",flush=True)
if __name__=="__main__": main()
