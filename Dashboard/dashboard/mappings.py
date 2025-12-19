# ================== PREDEFINED QUESTIONS BY ATTACK TYPE ==================

PREDEFINED_QUESTIONS = {
    "Generic": [
        "How to mitigate this kind of generic attack?",
        "What logs should I check for this attack?",
        "What are the main indicators of compromise?",
        "Which defense layers are most effective?"
    ],
    "Exploits": [
        "How to prevent exploit-based attacks?",
        "What vulnerability management practices are recommended?",
        "How to detect exploit attempts in logs?",
        "What patch management strategy should I implement?"
    ],
    "Fuzzers": [
        "How to defend against fuzzing attacks?",
        "What input validation techniques are effective?",
        "How to detect fuzzing attempts?",
        "What secure development practices help prevent fuzzing?"
    ],
    "Reconnaissance": [
        "How to limit reconnaissance activities?",
        "What monitoring techniques detect scanning?",
        "How to reduce network exposure?",
        "What are the best practices for network hardening?"
    ],
    "DoS": [
        "How to mitigate this kind of DoS?",
        "What logs should I check for this attack?",
        "How to implement DDoS protection?",
        "What rate limiting strategies are effective?"
    ],
    "Backdoors": [
        "How to detect and remove backdoors?",
        "What endpoint protection is recommended?",
        "How to monitor for backdoor activity?",
        "What access control measures prevent backdoors?"
    ],
    "Analysis": [
        "How to protect against malicious analysis?",
        "What code obfuscation techniques are effective?",
        "How to detect analysis attempts?",
        "What data protection measures are recommended?"
    ],
    "Shellcode": [
        "How to prevent shellcode execution?",
        "What memory protection techniques are effective?",
        "How to detect shellcode in network traffic?",
        "What exploit mitigation technologies should I use?"
    ],
    "Worms": [
        "How to contain worm propagation?",
        "What network segmentation strategies work best?",
        "How to detect worm activity?",
        "What patch management is critical for worms?"
    ]
}

# ================== SERVICE & PROTOCOL MAPPING ==================

SERVICE_MAPPING = {
    '0': 'HTTP', '1': 'HTTPS', '2': 'SSH', '3': 'FTP', '4': 'DNS',
    '5': 'SMTP', '6': 'POP3', '7': 'IMAP', '8': 'RDP', '9': 'TELNET',
    '10': 'MySQL', '11': 'PostgreSQL', '12': 'MongoDB', '13': 'Redis',
    '14': 'LDAP', '15': 'SMB', '16': 'NTP', '17': 'SNMP', '18': 'DHCP'
}

PROTOCOL_MAPPING = {
    '103': 'TCP', '104': 'UDP', '105': 'ICMP', '106': 'ARP',
    '107': 'HTTP', '108': 'HTTPS', '109': 'FTP', '110': 'SSH',
    '111': 'DNS', '112': 'SMTP', '113': 'DHCP', '114': 'TELNET',
    '115': 'RDP', '116': 'SMB', '117': 'SNMP', '118': 'LDAP',
    '119': 'MySQL', '120': 'ICMPv6'
}

# ======= NLP CLASS MAPPING (9 = normal,0–8 = attack types) =======

ATTACK_CLASS_MAP = {
    0: "Fuzzers",
    1: "Analysis",
    2: "Backdoors",
    3: "DoS",
    4: "Exploits",
    5: "Generic",
    6: "Reconnaissance",
    7: "Shellcode",
    8: "Worms",
    9: "unknown",
}

# ================== FEATURE LABELS ==================

FEATURE_LABELS = {
    "srcip": "Source IP Address",
    "sport": "Source Port",
    "dstip": "Destination IP Address",
    "dsport": "Destination Port",
    "proto": "Protocol",
    "state": "Connection State",
    "dur": "Duration (s)",
    "sbytes": "Bytes Sent",
    "dbytes": "Bytes Received",
    "sttl": "Source TTL",
    "dttl": "Destination TTL",
    "sloss": "Source Packet Loss",
    "dloss": "Destination Packet Loss",
    "service": "Network Service",
    "Sload": "Send Load",
    "Dload": "Receive Load",
    "Spkts": "Packets Sent",
    "Dpkts": "Packets Received",
    "swin": "Source TCP Window",
    "dwin": "Destination TCP Window",
    "stcpb": "Source TCP Sequence",
    "dtcpb": "Destination TCP Sequence",
    "smeansz": "Source Mean Packet Size",
    "dmeansz": "Destination Mean Packet Size",
    "trans_depth": "Transaction Depth",
    "res_bdy_len": "Response Body Length",
    "Sjit": "Source Jitter",
    "Djit": "Destination Jitter",
    "Stime": "Start Time",
    "Ltime": "End Time",
    "Sintpkt": "Source Inter-packet Interval",
    "Dintpkt": "Destination Inter-packet Interval",
    "tcprtt": "TCP RTT",
    "synack": "SYN-ACK Delay",
    "ackdat": "ACK-DATA Delay",
    "is_sm_ips_ports": "Similar IP/Port Events",
    "ct_state_ttl": "Connections State/TTL",
    "ct_flw_http_mthd": "HTTP Method",
    "is_ftp_login": "FTP Login",
    "ct_ftp_cmd": "FTP Commands",
    "ct_srv_src": "Connections by Service (Source)",
    "ct_srv_dst": "Connections by Service (Destination)",
    "ct_dst_ltm": "Connections to Destination",
    "ct_src_ltm": "Connections from Source",
    "ct_src_dport_ltm": "Source-Port Connections",
    "ct_dst_sport_ltm": "Destination-Port Connections",
    "ct_dst_src_ltm": "Destination-Source Connections",
    "attack_cat": "Attack Category",
    "Label": "Label (0 normal, 1 attack)",
    "text_features": "Textual Description",
    "attack_context": "Attack Context",
    "multi_class_label": "Multi-class Label (0–9)",
}