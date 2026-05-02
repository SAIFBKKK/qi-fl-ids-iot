"""
Tests unitaires — feature-extractor v2 (schéma raw_event enrichi).

4 groupes :
  TestWindowAggregation   — WindowAggregator
  TestFeatureExtraction   — build_window_dict + build_feature_vector (28 features)
  TestScalerApply         — apply_scaler
  TestMqttMapping         — pipeline complet sans broker réel
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from extractor import (
    RawEvent,
    WindowAggregator,
    apply_scaler,
    build_window_dict,
)
from feature_map import (
    FEATURE_DIM,
    FEATURE_NAMES,
    build_feature_vector,
    protocol_number,
)

# ---------------------------------------------------------------------------
# Helpers partagés
# ---------------------------------------------------------------------------

T0 = 1_700_000_000.0  # timestamp de référence


def _evt(
    *,
    src_ip: str = "192.168.1.10",
    dst_ip: str = "10.0.0.1",
    protocol: str = "TCP",
    src_port: int = 54321,
    dst_port: int = 80,
    pkt_count: int = 5,
    byte_count: int = 1500,
    direction: str = "outbound",
    ts: float = T0,
    header_length: int = 20,
    fin_flag: int = 0,
    syn_flag: int = 0,
    rst_flag: int = 0,
    psh_flag: int = 0,
    ack_flag: int = 1,
    urg_flag: int = 0,
) -> RawEvent:
    return RawEvent(
        timestamp=ts,
        src_ip=src_ip, dst_ip=dst_ip,
        protocol=protocol.upper(),
        src_port=src_port, dst_port=dst_port,
        pkt_count=pkt_count, byte_count=byte_count,
        direction=direction,
        header_length=header_length,
        fin_flag=fin_flag, syn_flag=syn_flag, rst_flag=rst_flag,
        psh_flag=psh_flag, ack_flag=ack_flag, urg_flag=urg_flag,
    )


def _scaler(n: int = 28) -> StandardScaler:
    rng = np.random.default_rng(42)
    return StandardScaler().fit(rng.normal(10.0, 2.0, (200, n)))


# ---------------------------------------------------------------------------
# TestWindowAggregation
# ---------------------------------------------------------------------------

class TestWindowAggregation:

    def test_single_event_creates_one_buffer(self):
        agg = WindowAggregator(60.0)
        agg.add_event(_evt(src_ip="10.0.0.1", ts=T0))
        assert agg.active_window_count() == 1

    def test_two_src_ips_create_two_buffers(self):
        agg = WindowAggregator(60.0)
        agg.add_event(_evt(src_ip="10.0.0.1", ts=T0))
        agg.add_event(_evt(src_ip="10.0.0.2", ts=T0))
        assert agg.active_window_count() == 2

    def test_same_src_ip_accumulates(self):
        agg = WindowAggregator(60.0)
        for i in range(5):
            agg.add_event(_evt(src_ip="10.0.0.1", ts=T0 + i))
        assert agg.active_window_count() == 1

    def test_flush_not_triggered_before_expiry(self):
        agg = WindowAggregator(10.0)
        agg.add_event(_evt(ts=T0))
        assert agg.flush_expired(now=T0 + 9.9) == []

    def test_flush_triggered_at_expiry(self):
        agg = WindowAggregator(10.0)
        for i in range(3):
            agg.add_event(_evt(src_ip="1.2.3.4", ts=T0 + i))
        flushed = agg.flush_expired(now=T0 + 10.0)
        assert len(flushed) == 1
        src_ip, events = flushed[0]
        assert src_ip == "1.2.3.4"
        assert len(events) == 3

    def test_flush_removes_expired_window(self):
        agg = WindowAggregator(10.0)
        agg.add_event(_evt(ts=T0))
        agg.flush_expired(now=T0 + 10.0)
        assert agg.active_window_count() == 0

    def test_flush_preserves_non_expired_window(self):
        agg = WindowAggregator(60.0)
        agg.add_event(_evt(ts=T0))
        agg.flush_expired(now=T0 + 59.9)
        assert agg.active_window_count() == 1

    def test_negative_window_size_raises(self):
        with pytest.raises(ValueError):
            WindowAggregator(-1.0)

    def test_zero_window_size_raises(self):
        with pytest.raises(ValueError):
            WindowAggregator(0.0)


# ---------------------------------------------------------------------------
# TestFeatureExtraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    """Tests sur build_window_dict() + build_feature_vector()."""

    def _tcp_window(self) -> list[RawEvent]:
        """3 événements TCP vers port 80, timestamps espacés de 10 s."""
        return [
            _evt(protocol="TCP", dst_port=80,  byte_count=1000, ts=T0 + 0,
                 syn_flag=1, ack_flag=0),
            _evt(protocol="TCP", dst_port=80,  byte_count=2000, ts=T0 + 10,
                 syn_flag=0, ack_flag=1),
            _evt(protocol="UDP", dst_port=53,  byte_count=500,  ts=T0 + 20,
                 syn_flag=0, ack_flag=0),
        ]

    # ---- build_window_dict -------------------------------------------------

    def test_window_dict_has_all_keys(self):
        wd = build_window_dict(self._tcp_window())
        required = {
            "flow_duration", "header_length", "protocol",
            "pkt_bytes", "timestamps", "dst_ports", "protocols",
            "fin_flags", "syn_flags", "rst_flags", "psh_flags",
            "ack_flags", "urg_flags",
        }
        assert required.issubset(wd.keys())

    def test_flow_duration_is_last_minus_first(self):
        wd = build_window_dict(self._tcp_window())
        assert wd["flow_duration"] == pytest.approx(20.0)

    def test_dominant_protocol_is_most_common(self):
        # 2 TCP, 1 UDP → dominant = TCP
        wd = build_window_dict(self._tcp_window())
        assert wd["protocol"] == "TCP"

    def test_empty_events_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_window_dict([])

    # ---- build_feature_vector ----------------------------------------------

    def test_vector_length_is_28(self):
        vec = build_feature_vector(build_window_dict(self._tcp_window()))
        assert len(vec) == FEATURE_DIM

    def test_all_values_are_finite(self):
        vec = build_feature_vector(build_window_dict(self._tcp_window()))
        assert all(math.isfinite(v) for v in vec)

    def test_flow_duration_idx0(self):
        vec = build_feature_vector(build_window_dict(self._tcp_window()))
        assert vec[0] == pytest.approx(20.0)

    def test_duration_idx3_equals_flow_duration(self):
        vec = build_feature_vector(build_window_dict(self._tcp_window()))
        assert vec[3] == vec[0]  # Duration = flow_duration (convention CIC)

    def test_rate_idx4(self):
        events = self._tcp_window()  # flow_duration=20, n=3
        vec = build_feature_vector(build_window_dict(events))
        assert vec[4] == pytest.approx(3 / 20.0)

    def test_rate_zero_when_single_event(self):
        # flow_duration=0 pour un seul événement → rate=0
        vec = build_feature_vector(build_window_dict([_evt(ts=T0)]))
        assert vec[4] == 0.0

    def test_syn_flag_number_idx6(self):
        # 1 événement avec syn_flag=1 → syn_flag_number=1
        events = [_evt(syn_flag=1, ts=T0)]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[6] == 1.0

    def test_syn_flag_number_zero_when_no_syn(self):
        events = [_evt(syn_flag=0, ts=T0)]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[6] == 0.0

    def test_syn_count_idx11_is_sum(self):
        events = [
            _evt(syn_flag=1, ts=T0),
            _evt(syn_flag=1, ts=T0 + 1),
            _evt(syn_flag=0, ts=T0 + 2),
        ]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[11] == 2.0  # syn_count

    def test_ack_count_idx10_is_sum(self):
        events = [
            _evt(ack_flag=1, ts=T0),
            _evt(ack_flag=1, ts=T0 + 1),
            _evt(ack_flag=1, ts=T0 + 2),
        ]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[10] == 3.0  # ack_count

    def test_urg_count_idx13_zero_by_default(self):
        events = [_evt(ts=T0 + i) for i in range(4)]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[13] == 0.0  # urg_count

    def test_http_idx15_set_when_port80(self):
        events = [_evt(dst_port=80, ts=T0)]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[15] == 1.0

    def test_http_idx15_zero_when_no_port80(self):
        events = [_evt(dst_port=443, ts=T0)]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[15] == 0.0

    def test_https_idx16_set_when_port443(self):
        events = [_evt(dst_port=443, ts=T0)]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[16] == 1.0

    def test_dns_idx17_set_when_port53(self):
        events = [_evt(protocol="UDP", dst_port=53, ts=T0)]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[17] == 1.0

    def test_tcp_idx19_one_when_dominant_tcp(self):
        events = [_evt(protocol="TCP", ts=T0 + i) for i in range(3)]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[19] == 1.0
        assert vec[20] == 0.0  # UDP=0

    def test_udp_idx20_one_when_dominant_udp(self):
        events = [_evt(protocol="UDP", ts=T0 + i) for i in range(3)]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[20] == 1.0
        assert vec[19] == 0.0  # TCP=0

    def test_protocol_type_idx2_tcp(self):
        events = [_evt(protocol="TCP", ts=T0)]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[2] == 6.0  # RFC: TCP=6

    def test_protocol_type_idx2_udp(self):
        events = [_evt(protocol="UDP", ts=T0)]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[2] == 17.0  # RFC: UDP=17

    def test_tot_sum_idx23(self):
        events = self._tcp_window()  # 1000+2000+500=3500
        vec = build_feature_vector(build_window_dict(events))
        assert vec[23] == pytest.approx(3500.0)

    def test_min_idx24(self):
        events = self._tcp_window()  # min=500
        vec = build_feature_vector(build_window_dict(events))
        assert vec[24] == pytest.approx(500.0)

    def test_std_idx25_zero_for_uniform_bytes(self):
        events = [_evt(byte_count=1000, ts=T0 + i) for i in range(4)]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[25] == pytest.approx(0.0, abs=1e-9)

    def test_iat_idx26_in_nanoseconds(self):
        # 3 événements espacés de 10 s → IAT = 10 s = 10e9 ns
        events = [
            _evt(ts=T0 + 0),
            _evt(ts=T0 + 10),
            _evt(ts=T0 + 20),
        ]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[26] == pytest.approx(10 * 1e9)

    def test_iat_idx26_zero_for_single_event(self):
        vec = build_feature_vector(build_window_dict([_evt(ts=T0)]))
        assert vec[26] == 0.0

    def test_number_idx27_is_event_count(self):
        events = [_evt(ts=T0 + i) for i in range(7)]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[27] == 7.0

    def test_header_length_idx1_is_mean(self):
        events = [
            _evt(header_length=20, ts=T0),
            _evt(header_length=40, ts=T0 + 1),
        ]
        vec = build_feature_vector(build_window_dict(events))
        assert vec[1] == pytest.approx(30.0)

    def test_feature_names_order_matches_feature_names_list(self):
        """Vérifie que chaque index du vecteur correspond au nom déclaré."""
        events = [
            _evt(protocol="TCP", dst_port=80, byte_count=1000, ts=T0 + i,
                 syn_flag=1, ack_flag=1)
            for i in range(3)
        ]
        vec = build_feature_vector(build_window_dict(events))
        assert len(vec) == len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# TestScalerApply
# ---------------------------------------------------------------------------

class TestScalerApply:

    def test_returns_28_floats(self):
        result = apply_scaler(_scaler(), [float(i) for i in range(28)])
        assert len(result) == 28
        assert all(isinstance(v, float) for v in result)

    def test_mean_input_gives_near_zero_output(self):
        sc = _scaler()
        result = apply_scaler(sc, list(sc.mean_))
        np.testing.assert_allclose(result, np.zeros(28), atol=1e-6)

    def test_output_values_are_finite(self):
        result = apply_scaler(_scaler(), [1.0] * 28)
        assert all(math.isfinite(v) for v in result)

    def test_real_bundle_scaler_smoke(self):
        bundle = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "experiments" / "fl-iot-ids-v3" / "outputs"
            / "deployment" / "baseline_fedavg_normal_classweights"
        )
        if not (bundle / "scaler.pkl").exists():
            pytest.skip("bundle not available")
        import joblib
        sc = joblib.load(str(bundle / "scaler.pkl"))
        result = apply_scaler(sc, list(sc.mean_))
        np.testing.assert_allclose(result, np.zeros(28), atol=1e-5)


# ---------------------------------------------------------------------------
# TestMqttMapping
# ---------------------------------------------------------------------------

class TestMqttMapping:
    """Pipeline complet sans broker réel."""

    def _syn_flood_window(self) -> list[RawEvent]:
        """Simule un SYN flood : beaucoup de SYN, TCP, port 80."""
        return [
            _evt(
                protocol="TCP", dst_port=80, byte_count=64,
                ts=T0 + i * 0.1,
                syn_flag=1, ack_flag=0, fin_flag=0,
            )
            for i in range(10)
        ]

    def test_full_pipeline_length(self):
        sc = _scaler()
        events = self._syn_flood_window()
        wd  = build_window_dict(events)
        raw = build_feature_vector(wd)
        out = apply_scaler(sc, raw)
        assert len(out) == 28

    def test_full_pipeline_all_finite(self):
        sc = _scaler()
        events = self._syn_flood_window()
        out = apply_scaler(sc, build_feature_vector(build_window_dict(events)))
        assert all(math.isfinite(v) for v in out)

    def test_syn_flood_has_syn_flag_set(self):
        events = self._syn_flood_window()
        raw = build_feature_vector(build_window_dict(events))
        syn_flag_idx = FEATURE_NAMES.index("syn_flag_number")
        assert raw[syn_flag_idx] == 1.0

    def test_syn_flood_syn_count_equals_10(self):
        events = self._syn_flood_window()
        raw = build_feature_vector(build_window_dict(events))
        syn_count_idx = FEATURE_NAMES.index("syn_count")
        assert raw[syn_count_idx] == 10.0

    def test_syn_flood_tcp_is_one(self):
        events = self._syn_flood_window()
        raw = build_feature_vector(build_window_dict(events))
        tcp_idx = FEATURE_NAMES.index("TCP")
        assert raw[tcp_idx] == 1.0

    def test_syn_flood_http_is_one(self):
        events = self._syn_flood_window()
        raw = build_feature_vector(build_window_dict(events))
        http_idx = FEATURE_NAMES.index("HTTP")
        assert raw[http_idx] == 1.0

    def test_published_message_structure(self):
        sc = _scaler()
        events = self._syn_flood_window()
        raw    = build_feature_vector(build_window_dict(events))
        scaled = apply_scaler(sc, raw)

        msg = {
            "schema_version": "1.0",
            "event_type":     "feature_vector",
            "node_id":        "test-node",
            "src_ip":         "192.168.1.10",
            "feature_names":  FEATURE_NAMES,
            "vector":         scaled,
            "event_count":    len(events),
        }
        payload = json.dumps(msg, separators=(",", ":"))
        parsed  = json.loads(payload)

        assert parsed["schema_version"] == "1.0"
        assert len(parsed["vector"]) == 28
        assert parsed["feature_names"] == FEATURE_NAMES
        assert parsed["event_count"] == 10

    def test_from_dict_parses_v1_event(self):
        """Schéma v1 (sans flags) : les defaults s'appliquent."""
        raw = {
            "timestamp": T0, "src_ip": "1.2.3.4", "dst_ip": "5.6.7.8",
            "protocol": "tcp", "src_port": 1, "dst_port": 80,
            "pkt_count": 3, "byte_count": 300, "direction": "outbound",
        }
        ev = RawEvent.from_dict(raw)
        assert ev.protocol == "TCP"
        assert ev.syn_flag == 0
        assert ev.ack_flag == 0
        assert ev.header_length == 20

    def test_from_dict_parses_v2_event(self):
        """Schéma v2 (avec flags) : les valeurs sont lues."""
        raw = {
            "timestamp": T0, "src_ip": "1.2.3.4", "dst_ip": "5.6.7.8",
            "protocol": "TCP", "src_port": 1, "dst_port": 443,
            "pkt_count": 8, "byte_count": 1200, "direction": "outbound",
            "header_length": 40,
            "fin_flag": 0, "syn_flag": 1, "rst_flag": 0,
            "psh_flag": 1, "ack_flag": 1, "urg_flag": 0,
        }
        ev = RawEvent.from_dict(raw)
        assert ev.syn_flag == 1
        assert ev.psh_flag == 1
        assert ev.header_length == 40

    def test_mocked_mqtt_publish_called(self):
        """Vérifie que publish() est appelé avec le bon topic."""
        import extractor as ext_module

        sc     = _scaler()
        events = self._syn_flood_window()

        mock_client = MagicMock()
        mock_client.publish.return_value = MagicMock(rc=0)

        raw    = build_feature_vector(build_window_dict(events))
        scaled = apply_scaler(sc, raw)
        body   = json.dumps({"vector": scaled}, separators=(",", ":"))

        mock_client.publish("iot/features", body, qos=1)
        assert mock_client.publish.called
        assert mock_client.publish.call_args[0][0] == "iot/features"
