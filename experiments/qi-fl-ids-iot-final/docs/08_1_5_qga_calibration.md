# P8.1.5 — QGA Calibration and Robustness Study

## 1. Objective

Test several QGA profiles, seeds, masks, and short true-Flower validation scenarios before freezing the final L1 mask.

## 2. Why One Mask Is Not Enough

QGA is stochastic, and one mask can reflect one seed and one fitness weighting. Calibration reduces the risk of selecting a brittle feature subset.

## 3. Critical QGA Parameters

- Fitness weights
- FPR penalty
- Min/max feature bounds
- Population size and generations
- Random seed

## 4. Profile Sweep

Profile sweep rows: 1

## 5. Flower 5-Round Validation

Short validation rows: 1

All ranking runs are validation-only and must not use the global test holdout.

## 6. Final Mask

No final mask selected yet.

## 7. Figures

- `C:/Users/saifb/dev/qi-fl-ids-iot/experiments/qi-fl-ids-iot-final/outputs/figures/qga/calibration/qga_profiles_macro_f1.png`
- `C:/Users/saifb/dev/qi-fl-ids-iot/experiments/qi-fl-ids-iot-final/outputs/figures/qga/calibration/qga_profiles_features_count.png`
- `C:/Users/saifb/dev/qi-fl-ids-iot/experiments/qi-fl-ids-iot-final/outputs/figures/qga/calibration/qga_profiles_fpr.png`
- `C:/Users/saifb/dev/qi-fl-ids-iot/experiments/qi-fl-ids-iot-final/outputs/figures/qga/calibration/qga_profile_fitness_boxplot.png`
- `C:/Users/saifb/dev/qi-fl-ids-iot/experiments/qi-fl-ids-iot-final/outputs/figures/qga/calibration/qga_short_flower_macro_f1_by_scenario.png`
- `C:/Users/saifb/dev/qi-fl-ids-iot/experiments/qi-fl-ids-iot-final/outputs/figures/qga/calibration/qga_short_flower_fpr_by_scenario.png`
- `C:/Users/saifb/dev/qi-fl-ids-iot/experiments/qi-fl-ids-iot-final/outputs/figures/qga/calibration/qga_engineering_score_ranking.png`
- `C:/Users/saifb/dev/qi-fl-ids-iot/experiments/qi-fl-ids-iot-final/outputs/figures/qga/calibration/qga_mask_stability_heatmap.png`

## 8. Conclusion

P8.1.5 is an engineering calibration layer. It does not launch P8-b L2, QIFA, FedTN, Docker, or dashboard work.
