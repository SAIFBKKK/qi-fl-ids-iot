# P16.10 Kali Lab Workstation Readiness

This directory prepares `lab-attacker-kali` as a future controlled lab workstation. P16.10 is a readiness and documentation phase only.

## VM Identity

- VM name: `lab-attacker-kali`
- Host-only IP: `192.168.56.103`
- Server IP: `192.168.56.1`
- Role: future CICIoT2023-inspired controlled scenario workstation inside the isolated live lab

## Files

- `kali_readiness_check.py`: safe inventory and fixed lab endpoint readiness check.
- `scenario_catalog.yaml`: descriptive catalog for the three selected future scenarios.
- `safe_scope.md`: safety boundary for this Kali workstation.
- `docs/kali_vm_readiness.md`: readiness checklist.
- `docs/kali_network_isolation.md`: host-only network isolation notes.
- `docs/selected_scenarios.md`: scenario descriptions without executable commands.

## Safety

P16.10 does not execute scenarios, does not generate traffic, does not start live capture, and does not run the scientific reference tools. Tool names are kept as references or inventory labels only.

## Safe Readiness Command

Run this inside the cloned repository on `lab-attacker-kali`:

```bash
python3 experiments/live_lab/kali/kali_readiness_check.py
```

The script writes:

- `experiments/qi-fl-ids-iot-final/outputs/reports/p16_10_kali_readiness.json`
- `experiments/qi-fl-ids-iot-final/outputs/reports/p16_10_kali_readiness.md`
