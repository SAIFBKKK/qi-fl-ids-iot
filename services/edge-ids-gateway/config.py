from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gateway_id: str = Field(default="node1", alias="GATEWAY_ID")
    node_group: str = Field(default="room-a", alias="NODE_GROUP")
    mqtt_enabled: bool = Field(default=False, alias="MQTT_ENABLED")
    mqtt_client_id: str = Field(default="edge-ids-gateway-node1", alias="MQTT_CLIENT_ID")
    mqtt_keepalive: int = Field(default=30, alias="MQTT_KEEPALIVE")
    mqtt_qos: int = Field(default=1, alias="MQTT_QOS")
    mqtt_broker: str = Field(default="mosquitto", alias="MQTT_BROKER")
    mqtt_port: int = Field(default=1883, alias="MQTT_PORT")
    mqtt_username: str = Field(default="ids_user", alias="MQTT_USERNAME")
    mqtt_password: str = Field(default="", alias="MQTT_PASSWORD")
    raw_input_topic: str = Field(default="iot/raw/node1", alias="RAW_INPUT_TOPIC")
    accepted_topic: str = Field(default="iot/accepted/node1", alias="ACCEPTED_TOPIC")
    blocked_topic: str = Field(default="iot/blocked/node1", alias="BLOCKED_TOPIC")
    predictions_topic: str = Field(default="ids/predictions/node1", alias="PREDICTIONS_TOPIC")
    alerts_topic: str = Field(default="ids/alerts/node1", alias="ALERTS_TOPIC")
    status_topic: str = Field(default="ids/status/gateway/node1", alias="STATUS_TOPIC")
    model_path: str = Field(default="/artifacts/global_model.pth", alias="MODEL_PATH")
    scaler_path: str = Field(default="/artifacts/scaler.pkl", alias="SCALER_PATH")
    feature_names_path: str = Field(default="/artifacts/feature_names.pkl", alias="FEATURE_NAMES_PATH")
    label_mapping_path: str = Field(default="/artifacts/label_mapping.json", alias="LABEL_MAPPING_PATH")
    model_config_path: str = Field(default="", alias="MODEL_CONFIG_PATH")
    inference_threshold: float = Field(default=0.5, alias="INFERENCE_THRESHOLD")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )


settings = Settings()
