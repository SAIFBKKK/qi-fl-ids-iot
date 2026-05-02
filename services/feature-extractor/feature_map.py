"""
Mapping statique : fenêtre d'événements → vecteur 28-features CIC-IoT.

Couverture : 28/28 features remplies
  - 26 features calculées depuis le schéma raw_event v2
  - 2 features avec default justifiable (voir §Hypothèses)

Hypothèses à vérifier :
  H1. L'ordre de FEATURE_NAMES correspond exactement à feature_names.pkl du bundle.
      Vérifier : python -c "import joblib; print(joblib.load('/artifacts/feature_names.pkl'))"
  H2. Le scaler est un StandardScaler (scaler.mean_ disponible).
      Vérifier : python -c "import joblib; s=joblib.load('/artifacts/scaler.pkl'); print(type(s).__name__)"
  H3. Header_Length : estimé à 20 octets (TCP standard) si absent du raw_event.
      Défendable car Header_Length varie peu en pratique pour ce type de trafic.
  H4. urg_count ≈ 0 dans CIC-IoT-2023 (flag URG quasi-absent). Default 0 justifiable.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Liste canonique — doit correspondre à feature_names.pkl (ordre inclus)
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "flow_duration",    # 0  last_ts - first_ts (secondes)
    "Header_Length",    # 1  octets d'en-tête (moyenne sur la fenêtre)
    "Protocol Type",    # 2  numéro IANA (TCP=6, UDP=17, ICMP=1, ARP=0)
    "Duration",         # 3  = flow_duration (même valeur, convention CIC)
    "Rate",             # 4  paquets/seconde = n_pkts / flow_duration
    "fin_flag_number",  # 5  1 si au moins un paquet FIN dans la fenêtre
    "syn_flag_number",  # 6  1 si au moins un paquet SYN
    "rst_flag_number",  # 7  1 si au moins un paquet RST
    "psh_flag_number",  # 8  1 si au moins un paquet PSH
    "ack_flag_number",  # 9  1 si au moins un paquet ACK
    "ack_count",        # 10 Σ ack_flag sur la fenêtre
    "syn_count",        # 11 Σ syn_flag
    "fin_count",        # 12 Σ fin_flag
    "urg_count",        # 13 Σ urg_flag (≈ 0 dans CIC-IoT-2023)
    "rst_count",        # 14 Σ rst_flag
    "HTTP",             # 15 1 si au moins un flux vers port 80
    "HTTPS",            # 16 1 si au moins un flux vers port 443
    "DNS",              # 17 1 si au moins un flux vers port 53
    "SSH",              # 18 1 si au moins un flux vers port 22
    "TCP",              # 19 1 si protocole dominant = TCP
    "UDP",              # 20 1 si protocole dominant = UDP
    "ARP",              # 21 1 si protocole dominant = ARP
    "ICMP",             # 22 1 si protocole dominant = ICMP
    "Tot sum",          # 23 Σ byte_count
    "Min",              # 24 min(byte_count)
    "Std",              # 25 σ(byte_count)
    "IAT",              # 26 inter-arrival time moyen (nanosecondes)
    "Number",           # 27 nombre d'événements dans la fenêtre
]

FEATURE_DIM = 28
assert len(FEATURE_NAMES) == FEATURE_DIM

# Alias pour compatibilité avec l'ancienne API
CANONICAL_FEATURE_NAMES = FEATURE_NAMES

# ---------------------------------------------------------------------------
# Encodage numérique des protocoles (RFC 790 / IANA)
# ---------------------------------------------------------------------------
_PROTOCOL_NUMBER: dict[str, int] = {
    "TCP":   6,
    "UDP":   17,
    "ICMP":  1,
    "ARP":   0,
    "HTTP":  6,   # HTTP/HTTPS roulent sur TCP
    "HTTPS": 6,
    "DNS":   17,  # DNS principalement sur UDP
    "SSH":   6,
}


def protocol_number(proto: str) -> int:
    return _PROTOCOL_NUMBER.get(proto.upper(), 0)


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def build_feature_vector(window: dict[str, Any]) -> list[float]:
    """Construit le vecteur de 28 features à partir d'un dict de fenêtre.

    Clés attendues dans `window` (toutes produites par build_window_dict()) :
        flow_duration  float          secondes (last_ts - first_ts)
        header_length  float          moyenne des Header_Length
        protocol       str            protocole dominant (TCP/UDP/ICMP/ARP)
        pkt_bytes      list[int]      byte_count par événement
        timestamps     list[float]    timestamp par événement
        dst_ports      list[int]      dst_port par événement
        protocols      list[str]      protocole par événement
        fin_flags      list[int]      fin_flag (0/1) par événement
        syn_flags      list[int]      syn_flag (0/1) par événement
        rst_flags      list[int]      rst_flag (0/1) par événement
        psh_flags      list[int]      psh_flag (0/1) par événement
        ack_flags      list[int]      ack_flag (0/1) par événement
        urg_flags      list[int]      urg_flag (0/1) par événement

    Retourne : liste de 28 floats dans l'ordre de FEATURE_NAMES.
    """
    duration     = float(window.get("flow_duration", 0.0))
    header_len   = float(window.get("header_length", 20.0))
    dominant     = str(window.get("protocol", "TCP")).upper()

    pkt_bytes  = window.get("pkt_bytes",  [0])
    timestamps = window.get("timestamps", [])
    dst_ports  = window.get("dst_ports",  [])

    fin_flags = window.get("fin_flags", [])
    syn_flags = window.get("syn_flags", [])
    rst_flags = window.get("rst_flags", [])
    psh_flags = window.get("psh_flags", [])
    ack_flags = window.get("ack_flags", [])
    urg_flags = window.get("urg_flags", [])

    arr     = np.asarray(pkt_bytes, dtype=np.float64) if pkt_bytes else np.array([0.0])
    n_pkts  = len(arr)

    # --- IAT en nanosecondes ---
    if len(timestamps) > 1:
        ts_sorted = sorted(timestamps)
        iats      = [ts_sorted[i + 1] - ts_sorted[i] for i in range(len(ts_sorted) - 1)]
        # timestamps sont en secondes Unix → convertir en ns pour correspondre au dataset
        iat_ns = float(np.mean(iats)) * 1e9
    else:
        iat_ns = 0.0

    # --- Rate (paquets/s) ---
    rate = n_pkts / duration if duration > 0.0 else 0.0

    # --- Helpers flags ---
    def _flag(flags: list[int]) -> float:
        return float(max(flags)) if flags else 0.0

    def _count(flags: list[int]) -> float:
        return float(sum(flags)) if flags else 0.0

    # --- Indicateurs protocole réseau (dominance de la fenêtre) ---
    is_tcp  = 1.0 if dominant == "TCP"  else 0.0
    is_udp  = 1.0 if dominant == "UDP"  else 0.0
    is_arp  = 1.0 if dominant == "ARP"  else 0.0
    is_icmp = 1.0 if dominant == "ICMP" else 0.0

    # --- Indicateurs couche applicative (présence dans la fenêtre) ---
    has_http  = 1.0 if any(p == 80  for p in dst_ports) else 0.0
    has_https = 1.0 if any(p == 443 for p in dst_ports) else 0.0
    has_dns   = 1.0 if any(p == 53  for p in dst_ports) else 0.0
    has_ssh   = 1.0 if any(p == 22  for p in dst_ports) else 0.0

    vector = [
        duration,                        # 0  flow_duration
        header_len,                      # 1  Header_Length
        float(protocol_number(dominant)),# 2  Protocol Type
        duration,                        # 3  Duration (= flow_duration, convention CIC)
        rate,                            # 4  Rate
        _flag(fin_flags),               # 5  fin_flag_number
        _flag(syn_flags),               # 6  syn_flag_number
        _flag(rst_flags),               # 7  rst_flag_number
        _flag(psh_flags),               # 8  psh_flag_number
        _flag(ack_flags),               # 9  ack_flag_number
        _count(ack_flags),              # 10 ack_count
        _count(syn_flags),              # 11 syn_count
        _count(fin_flags),              # 12 fin_count
        _count(urg_flags),              # 13 urg_count
        _count(rst_flags),              # 14 rst_count
        has_http,                        # 15 HTTP
        has_https,                       # 16 HTTPS
        has_dns,                         # 17 DNS
        has_ssh,                         # 18 SSH
        is_tcp,                          # 19 TCP
        is_udp,                          # 20 UDP
        is_arp,                          # 21 ARP
        is_icmp,                         # 22 ICMP
        float(np.sum(arr)),              # 23 Tot sum
        float(np.min(arr)),              # 24 Min
        float(np.std(arr)),              # 25 Std
        iat_ns,                          # 26 IAT (nanosecondes)
        float(n_pkts),                   # 27 Number
    ]

    if len(vector) != FEATURE_DIM:  # garde-fou de développement
        raise RuntimeError(f"build_feature_vector produced {len(vector)} values, expected {FEATURE_DIM}")

    return vector
