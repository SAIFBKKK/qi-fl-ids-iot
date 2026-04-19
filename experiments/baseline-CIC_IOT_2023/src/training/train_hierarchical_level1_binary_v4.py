
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

SEED=42; LABEL_COL="label"; TARGET_COL="binary_label"
DATA_DIR=Path(r"E:\dataset\processed_merged_full\hierarchical_final_v3\level1_binary")
OUTPUT_DIR=Path(r"E:\dataset\processed_merged_full\hierarchical_final_v4_models\level1_binary"); OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
BATCH_SIZE=2048; MAX_EPOCHS=40; LEARNING_RATE=5e-4; WEIGHT_DECAY=1e-4; EARLY_STOPPING_PATIENCE=8; SCHEDULER_PATIENCE=3; SCHEDULER_FACTOR=0.5; MIN_LR=1e-6; NUM_WORKERS=0
HIDDEN_1=128; HIDDEN_2=64; DROPOUT_1=0.25; DROPOUT_2=0.15; FOCAL_GAMMA=2.0

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true,y_pred):
    return {"accuracy":float(accuracy_score(y_true,y_pred)),"macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0)),"weighted_f1":float(f1_score(y_true,y_pred,average="weighted",zero_division=0)),"macro_precision":float(precision_score(y_true,y_pred,average="macro",zero_division=0)),"macro_recall":float(recall_score(y_true,y_pred,average="macro",zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred))}

class BinaryMLPV4(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,HIDDEN_1),nn.BatchNorm1d(HIDDEN_1),nn.ReLU(),nn.Dropout(DROPOUT_1),nn.Linear(HIDDEN_1,HIDDEN_2),nn.BatchNorm1d(HIDDEN_2),nn.ReLU(),nn.Dropout(DROPOUT_2),nn.Linear(HIDDEN_2,2))
    def forward(self,x): return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0):
        super().__init__(); self.gamma=gamma; self.ce=nn.CrossEntropyLoss(reduction="none")
    def forward(self,logits,targets):
        ce_loss=self.ce(logits,targets); pt=torch.exp(-ce_loss); return (((1-pt)**self.gamma)*ce_loss).mean()

def load_split(path):
    df=pd.read_csv(path)
    feature_cols=[c for c in df.columns if c not in [LABEL_COL,TARGET_COL,"family","family_id","subtype_id","label_id_34"]]
    X=df[feature_cols].to_numpy(dtype=np.float32); y=df[TARGET_COL].to_numpy(dtype=np.int64)
    return df,feature_cols,X,y

def balance_binary_train_df(train_df):
    benign_df=train_df[train_df[TARGET_COL]==0].copy()
    attack_df=train_df[train_df[TARGET_COL]==1].copy()
    n_target=min(len(benign_df),len(attack_df))
    benign_df=benign_df.sample(n=n_target,random_state=SEED)
    attack_df=attack_df.sample(n=n_target,random_state=SEED)
    balanced_df=pd.concat([benign_df,attack_df],ignore_index=True)
    return balanced_df.sample(frac=1.0,random_state=SEED).reset_index(drop=True)

def eval_logits(model,loader,criterion,device):
    model.eval(); loss_sum=0.0; n=0; probs_attack=[]; yt=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); logits=model(xb); loss=criterion(logits,yb); bs=xb.size(0)
            loss_sum+=loss.item()*bs; n+=bs
            prob=torch.softmax(logits,dim=1)[:,1]
            probs_attack.append(prob.cpu().numpy()); yt.append(yb.cpu().numpy())
    return float(loss_sum/max(n,1)), np.concatenate(yt), np.concatenate(probs_attack)

def threshold_search(y_true,p_attack):
    thresholds=np.arange(0.10,0.91,0.02); rows=[]; best_t=0.50; best_f1=-1.0
    for t in thresholds:
        y_pred=(p_attack>=t).astype(np.int64); m=compute_metrics(y_true,y_pred); rows.append({"threshold":float(t),**m})
        if m["macro_f1"]>best_f1: best_f1=m["macro_f1"]; best_t=float(t)
    return best_t,pd.DataFrame(rows)

def save_confusion(cm,labels,csv_path,png_path,title):
    pd.DataFrame(cm,index=labels,columns=labels).to_csv(csv_path)
    plt.figure(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues"); plt.title(title); plt.tight_layout(); plt.savefig(png_path,dpi=150); plt.close()

def main():
    set_seed(SEED); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_raw_df,feature_cols,_,_=load_split(DATA_DIR/"train.csv"); _,_,X_val,y_val=load_split(DATA_DIR/"val.csv"); _,_,X_test,y_test=load_split(DATA_DIR/"test.csv")
    train_df=balance_binary_train_df(train_raw_df)
    X_train=train_df[feature_cols].to_numpy(dtype=np.float32); y_train=train_df[TARGET_COL].to_numpy(dtype=np.int64)
    train_loader=DataLoader(TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train)),batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
    val_loader=DataLoader(TensorDataset(torch.from_numpy(X_val),torch.from_numpy(y_val)),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
    test_loader=DataLoader(TensorDataset(torch.from_numpy(X_test),torch.from_numpy(y_test)),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
    model=BinaryMLPV4(len(feature_cols)).to(device); criterion=FocalLoss(gamma=FOCAL_GAMMA)
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=SCHEDULER_FACTOR,patience=SCHEDULER_PATIENCE,min_lr=MIN_LR)
    history=[]; best_state=None; best_epoch=-1; best_f1=-1.0; patience=0; best_threshold=0.50; start=time.time()
    print("Train raw distribution:",train_raw_df[TARGET_COL].value_counts().to_dict(),flush=True)
    print("Train balanced distribution:",train_df[TARGET_COL].value_counts().to_dict(),flush=True)
    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); loss_sum=0.0; n=0; lr=optimizer.param_groups[0]["lr"]
        for xb,yb in train_loader:
            xb,yb=xb.to(device),yb.to(device); optimizer.zero_grad(set_to_none=True); logits=model(xb); loss=criterion(logits,yb); loss.backward(); optimizer.step()
            bs=xb.size(0); loss_sum+=loss.item()*bs; n+=bs
        train_loss=loss_sum/max(n,1)
        val_loss,y_val_true,p_val_attack=eval_logits(model,val_loader,criterion,device)
        current_threshold,threshold_df=threshold_search(y_val_true,p_val_attack)
        y_val_pred=(p_val_attack>=current_threshold).astype(np.int64); val_metrics=compute_metrics(y_val_true,y_val_pred)
        scheduler.step(val_metrics["macro_f1"])
        history.append({"epoch":epoch,"learning_rate":float(lr),"train_loss":float(train_loss),"val_loss":float(val_loss),"best_threshold_epoch":float(current_threshold),**{f"val_{k}":float(v) for k,v in val_metrics.items()}})
        print(f"Epoch {epoch:03d} | lr={lr:.7f} | train_loss={train_loss:.6f} | val_macro_f1={val_metrics['macro_f1']:.6f} | thr={current_threshold:.2f}",flush=True)
        if val_metrics["macro_f1"]>best_f1+1e-5:
            best_f1=val_metrics["macro_f1"]; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); best_threshold=current_threshold; threshold_df.to_csv(OUTPUT_DIR/"threshold_search_best_epoch.csv",index=False); patience=0
        else:
            patience+=1
            if patience>=EARLY_STOPPING_PATIENCE:
                print("Early stopping",flush=True); break
    if best_state is not None: model.load_state_dict(best_state)
    val_loss,yv_t,pv=eval_logits(model,val_loader,criterion,device); yv_p=(pv>=best_threshold).astype(np.int64); val_metrics=compute_metrics(yv_t,yv_p); val_metrics["loss"]=float(val_loss)
    test_loss,yt_t,pt=eval_logits(model,test_loader,criterion,device); yt_p=(pt>=best_threshold).astype(np.int64); test_metrics=compute_metrics(yt_t,yt_p); test_metrics["loss"]=float(test_loss)
    pd.DataFrame(history).to_csv(OUTPUT_DIR/"training_history.csv",index=False)
    pd.DataFrame(val_metrics.items(),columns=["metric","value"]).to_csv(OUTPUT_DIR/"val_metrics.csv",index=False)
    pd.DataFrame(test_metrics.items(),columns=["metric","value"]).to_csv(OUTPUT_DIR/"test_metrics.csv",index=False)
    pd.DataFrame(classification_report(yt_t,yt_p,target_names=["benign","attack"],output_dict=True,zero_division=0)).T.to_csv(OUTPUT_DIR/"classification_report_test.csv")
    cm=confusion_matrix(yt_t,yt_p,labels=[0,1]); save_confusion(cm,["benign","attack"],OUTPUT_DIR/"confusion_matrix_test.csv",OUTPUT_DIR/"confusion_matrix_test.png","Level1 Binary V4 Confusion Matrix")
    torch.save(model.state_dict(),OUTPUT_DIR/"mlp_model_state.pt")
    pickle.dump(feature_cols,open(OUTPUT_DIR/"feature_names.pkl","wb"))
    pickle.dump(pickle.load(open(DATA_DIR/"label_mapping.pkl","rb")),open(OUTPUT_DIR/"label_mapping.pkl","wb"))
    json.dump({"best_epoch":best_epoch,"best_val_macro_f1":best_f1,"best_threshold":best_threshold,"n_features":len(feature_cols),"training_time_sec":time.time()-start,"train_raw_distribution":train_raw_df[TARGET_COL].value_counts().to_dict(),"train_balanced_distribution":train_df[TARGET_COL].value_counts().to_dict(),"architecture":"32 -> 128 -> 64 -> 2","loss":f"focal_gamma_{FOCAL_GAMMA}"},open(OUTPUT_DIR/"run_summary.json","w"),indent=2)
    print("Saved to",OUTPUT_DIR,flush=True)

if __name__=="__main__": main()
