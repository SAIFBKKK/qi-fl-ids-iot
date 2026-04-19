
import json,time,copy,pickle,random,warnings
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,balanced_accuracy_score,classification_report,confusion_matrix
import torch,torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
warnings.filterwarnings("ignore")

SEED=42; LABEL_COL="label"; TARGET_COL="family_id"
DATA_DIR=Path(r"E:\dataset\processed_merged_full\hierarchical_final_v3\level2_family")
OUTPUT_DIR=Path(r"E:\dataset\processed_merged_full\hierarchical_final_v4_models\level2_family"); OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
BATCH_SIZE=2048; MAX_EPOCHS=70; LEARNING_RATE=5e-4; WEIGHT_DECAY=1e-4; EARLY_STOPPING_PATIENCE=10; SCHEDULER_PATIENCE=4; SCHEDULER_FACTOR=0.5; MIN_LR=1e-6; NUM_WORKERS=0
HIDDEN_1=128; HIDDEN_2=64; DROPOUT_1=0.30; DROPOUT_2=0.20; FOCAL_GAMMA=2.0

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def compute_metrics(y_true,y_pred):
    return {"accuracy":float(accuracy_score(y_true,y_pred)),"macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0)),"weighted_f1":float(f1_score(y_true,y_pred,average="weighted",zero_division=0)),"macro_precision":float(precision_score(y_true,y_pred,average="macro",zero_division=0)),"macro_recall":float(recall_score(y_true,y_pred,average="macro",zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred))}
class FamilyMLPV4(nn.Module):
    def __init__(self,input_dim,num_classes):
        super().__init__(); self.net=nn.Sequential(nn.Linear(input_dim,HIDDEN_1),nn.BatchNorm1d(HIDDEN_1),nn.ReLU(),nn.Dropout(DROPOUT_1),nn.Linear(HIDDEN_1,HIDDEN_2),nn.BatchNorm1d(HIDDEN_2),nn.ReLU(),nn.Dropout(DROPOUT_2),nn.Linear(HIDDEN_2,num_classes))
    def forward(self,x): return self.net(x)
class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0):
        super().__init__(); self.gamma=gamma; self.ce=nn.CrossEntropyLoss(reduction="none")
    def forward(self,logits,targets):
        ce_loss=self.ce(logits,targets); pt=torch.exp(-ce_loss); return (((1-pt)**self.gamma)*ce_loss).mean()
def load_split(path):
    df=pd.read_csv(path); feature_cols=[c for c in df.columns if c not in [LABEL_COL,"binary_label","family","family_id","subtype_id","label_id_34"]]
    X=df[feature_cols].to_numpy(dtype=np.float32); y=df[TARGET_COL].to_numpy(dtype=np.int64); return df,feature_cols,X,y
def eval_model(model,loader,criterion,device):
    model.eval(); loss_sum=0.0; n=0; yp=[]; yt=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb=xb.to(device), yb.to(device); logits=model(xb); loss=criterion(logits,yb); bs=xb.size(0); loss_sum+=loss.item()*bs; n+=bs; yp.append(torch.argmax(logits,dim=1).cpu().numpy()); yt.append(yb.cpu().numpy())
    y_true=np.concatenate(yt); y_pred=np.concatenate(yp); m=compute_metrics(y_true,y_pred); m["loss"]=float(loss_sum/max(n,1)); return m,y_true,y_pred
def save_confusion(cm,labels,csv_path,png_path,title):
    pd.DataFrame(cm,index=labels,columns=labels).to_csv(csv_path); plt.figure(figsize=(8,6)); sns.heatmap(cm,cmap="Blues"); plt.title(title); plt.tight_layout(); plt.savefig(png_path,dpi=150); plt.close()
def main():
    set_seed(SEED); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map=pickle.load(open(DATA_DIR/"label_mapping.pkl","rb")); id_to_label={v:k for k,v in label_map.items()}
    _,feature_cols,X_train,y_train=load_split(DATA_DIR/"train.csv"); _,_,X_val,y_val=load_split(DATA_DIR/"val.csv"); _,_,X_test,y_test=load_split(DATA_DIR/"test.csv")
    train_loader=DataLoader(TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train)),batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
    val_loader=DataLoader(TensorDataset(torch.from_numpy(X_val),torch.from_numpy(y_val)),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
    test_loader=DataLoader(TensorDataset(torch.from_numpy(X_test),torch.from_numpy(y_test)),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
    model=FamilyMLPV4(len(feature_cols),len(label_map)).to(device); criterion=FocalLoss(gamma=FOCAL_GAMMA)
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=SCHEDULER_FACTOR,patience=SCHEDULER_PATIENCE,min_lr=MIN_LR)
    history=[]; best_state=None; best_epoch=-1; best_f1=-1.0; patience=0; start=time.time()
    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); loss_sum=0.0; n=0; lr=optimizer.param_groups[0]["lr"]
        for xb,yb in train_loader:
            xb,yb=xb.to(device),yb.to(device); optimizer.zero_grad(set_to_none=True); logits=model(xb); loss=criterion(logits,yb); loss.backward(); optimizer.step(); bs=xb.size(0); loss_sum+=loss.item()*bs; n+=bs
        train_loss=loss_sum/max(n,1); val_m,_,_=eval_model(model,val_loader,criterion,device); scheduler.step(val_m["macro_f1"])
        history.append({"epoch":epoch,"learning_rate":float(lr),"train_loss":float(train_loss),**{f"val_{k}":float(v) for k,v in val_m.items()}})
        print(f"Epoch {epoch:03d} | lr={lr:.7f} | train_loss={train_loss:.6f} | val_macro_f1={val_m['macro_f1']:.6f}",flush=True)
        if val_m["macro_f1"]>best_f1+1e-5: best_f1=val_m["macro_f1"]; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); patience=0
        else:
            patience+=1
            if patience>=EARLY_STOPPING_PATIENCE: print("Early stopping",flush=True); break
    if best_state is not None: model.load_state_dict(best_state)
    val_m,_,_=eval_model(model,val_loader,criterion,device); test_m,yt_t,yt_p=eval_model(model,test_loader,criterion,device)
    pd.DataFrame(history).to_csv(OUTPUT_DIR/"training_history.csv",index=False); pd.DataFrame(val_m.items(),columns=["metric","value"]).to_csv(OUTPUT_DIR/"val_metrics.csv",index=False); pd.DataFrame(test_m.items(),columns=["metric","value"]).to_csv(OUTPUT_DIR/"test_metrics.csv",index=False)
    target_names=[id_to_label[i] for i in range(len(id_to_label))]; pd.DataFrame(classification_report(yt_t,yt_p,target_names=target_names,output_dict=True,zero_division=0)).T.to_csv(OUTPUT_DIR/"classification_report_test.csv")
    cm=confusion_matrix(yt_t,yt_p,labels=list(range(len(id_to_label)))); save_confusion(cm,target_names,OUTPUT_DIR/"confusion_matrix_test.csv",OUTPUT_DIR/"confusion_matrix_test.png","Level2 Family V4 Confusion Matrix")
    torch.save(model.state_dict(),OUTPUT_DIR/"mlp_model_state.pt"); pickle.dump(feature_cols,open(OUTPUT_DIR/"feature_names.pkl","wb")); pickle.dump(label_map,open(OUTPUT_DIR/"label_mapping.pkl","wb"))
    json.dump({"best_epoch":best_epoch,"best_val_macro_f1":best_f1,"n_features":len(feature_cols),"training_time_sec":time.time()-start,"architecture":"32 -> 128 -> 64 -> family_count","loss":f"focal_gamma_{FOCAL_GAMMA}"},open(OUTPUT_DIR/"run_summary.json","w"),indent=2)
    print("Saved to",OUTPUT_DIR,flush=True)
if __name__=="__main__": main()
