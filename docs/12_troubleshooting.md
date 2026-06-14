# Troubleshooting

## Missing Dataset Files

Full datasets are not stored in GitHub. Use the Kaggle package when it becomes available.

## Missing Model Or Scaler Files

Model checkpoints and scaler binaries were externalized. Restore the external artifacts archive only when deployment or exact result reproduction requires it.

## Docker Environment Fails

Check that `services/.env` exists locally and was created from `services/.env.example`. Replace placeholders with local values.

## MQTT Connection Fails

Check local values for `<MQTT_HOST>`, port, username, and password. Do not commit real credentials.

## CI Fails On Missing Heavy Files

CI should use tiny samples and import tests only. Do not make CI depend on full datasets or external artifacts.

## Private Data Appears In Logs

Stop and review [security_publication_checklist.md](security_publication_checklist.md). Do not publish private IPs, packet captures, credentials, or raw logs.
