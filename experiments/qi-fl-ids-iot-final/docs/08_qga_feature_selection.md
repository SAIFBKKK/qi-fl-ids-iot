# P8 — QGA Feature Selection

## 1. Objectif

P8 introduit la selection de features Quantum-Inspired par QGA pour le modele L1 binaire normal vs attack.

## 2. Donnees utilisees

Le masque QGA est appris uniquement avec `train_scaled.npz` et `val_scaled.npz` L1. Le global test holdout reste reserve a l'evaluation finale.

## 3. Methode QGA

Chaque chromosome est un masque binaire de 28 positions. Une position a 1 conserve la feature, une position a 0 la supprime. Le search maintient un vecteur theta et echantillonne les probabilites `sin(theta)^2`.

## 4. Fitness

La fitness P8 est:

`0.6 * MacroF1 + 0.3 * Recall_attack - 0.1 * (features_count / 28)`

Macro-F1 reste prioritaire, attack recall protege la detection des attaques, et la penalite de taille favorise un modele plus leger pour IoT.

## 5. Artefacts QGA

Les runs QGA standalone sont ecrits sous `outputs/qga_feature_selection/runs/{run_id}/`.

## 6. FedAvg + QGA

Le masque QGA est applique aux partitions L1 P3 avant d'executer FedAvg L1. Le scenario principal est `alpha=0.5, K=3`.

Depuis P8, les entrainements FL finaux doivent utiliser un vrai runtime Flower. Le script historique `08_run_qga_fedavg_l1.py` reste disponible comme helper in-process experimental, mais il ne doit pas etre presente comme baseline FL finale.

Runtime Flower manuel:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/08_start_qga_fedavg_flower_server.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml --alpha 0.5 --clients 3 --rounds 30 --address 127.0.0.1:8083
python experiments/qi-fl-ids-iot-final/src/scripts/08_start_qga_fedavg_flower_client.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml --client-id client_1 --alpha 0.5 --clients 3 --address 127.0.0.1:8083
python experiments/qi-fl-ids-iot-final/src/scripts/08_start_qga_fedavg_flower_client.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml --client-id client_2 --alpha 0.5 --clients 3 --address 127.0.0.1:8083
python experiments/qi-fl-ids-iot-final/src/scripts/08_start_qga_fedavg_flower_client.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml --client-id client_3 --alpha 0.5 --clients 3 --address 127.0.0.1:8083
```

Smoke Flower subprocess:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/08_run_qga_fedavg_flower_smoke.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000 --address 127.0.0.1:8083
```

## 7. HeteroFL + QGA

Le masque QGA est applique aux sous-modeles weak, medium et powerful de P7 HeteroFL L1. L3 n'est pas utilise en P8.

Statut runtime: HeteroFL + QGA reste in-process experimental. Il n'est pas encore un vrai runtime Flower final, car HeteroFL utilise des largeurs de modeles differentes par tier et demande une strategie Flower specifique pour agreger des slices heterogenes. Il ne doit donc pas etre presente comme baseline Flower finale.

## 8. Commandes

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/08_verify_qga_setup.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml
python experiments/qi-fl-ids-iot-final/src/scripts/08_run_qga_feature_selection.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml --mode smoke
python experiments/qi-fl-ids-iot-final/src/scripts/08_run_qga_feature_selection.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml --mode full
python experiments/qi-fl-ids-iot-final/src/scripts/08_run_qga_fedavg_flower_smoke.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000
python experiments/qi-fl-ids-iot-final/src/scripts/08_run_qga_fedavg_l1.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml --mode full --alpha 0.5 --clients 3 --rounds 30
python experiments/qi-fl-ids-iot-final/src/scripts/08_run_qga_heterofl_l1.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml --mode full --alpha 0.5 --clients 3 --rounds 30
python experiments/qi-fl-ids-iot-final/src/scripts/08_build_qga_ablation_report.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml
```

## 9. Risques restants

- Le fast MLP est une approximation; les resultats finaux doivent etre confirmes par FedAvg/HeteroFL full.
- Un masque trop agressif peut augmenter le FPR.
- L2 + QGA est reporte a P8-b.

## 10. Conclusion P8

[Phase P8 - Code-ready apres validation smoke. Ne pas passer a P9 sans validation utilisateur.]
