import os
import socket


def get_node_ip():
    # Prefer an explicit master addr if provided by launcher/container env.
    # This avoids relying on external network reachability (e.g. 8.8.8.8) which
    # is often blocked in restricted environments.
    env_master = (os.environ.get("MASTER_ADDR") or os.environ.get("MASTER_ADDRESS") or "").strip()
    if env_master:
        return env_master

    # Best-effort: infer the primary local IP via a UDP "connect" (no packets sent),
    # but fall back gracefully if the network is unreachable.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        pass

    # Fallback: resolve hostname, then to loopback.
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def collect_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]
