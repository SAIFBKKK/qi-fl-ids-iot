from __future__ import annotations


class MQTTEdgeGatewayCollector:
    """Placeholder MQTT collector for the future edge gateway service.

    TODO(P7.6): connect to MQTT, subscribe to raw topics, and publish decisions.
    """

    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False
