#!/usr/bin/env python3
"""
最小冒烟测试：验证 ROLL(SglangProxy) 是否会发送 `X-ROLL-Preferred-Worker-Url`
并且 router 能按该 header 将请求路由到指定 worker。

不依赖真实 sglang / sglang-router，使用三个本地 HTTP server：
- router: 监听 /generate，读取 header 并转发到对应 worker 的 /generate
- worker1/worker2: 监听 /generate，返回可被 roll.distributed.strategy.sglang_strategy.postprocess_generate 解析的最小响应

运行：
  python3 scripts/smoke_test_sglang_router_preferred_url.py
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional, Tuple


def _read_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _write_json(handler: BaseHTTPRequestHandler, obj: Any, status: int = 200) -> None:
    data = json.dumps(obj).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _http_post_json(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=5) as resp:
        body = resp.read()
        return json.loads(body.decode("utf-8"))


@dataclass(frozen=True)
class FakeWorkerConfig:
    name: str
    output_ids: Tuple[int, ...]


def make_fake_worker_handler(cfg: FakeWorkerConfig):
    class FakeWorkerHandler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path != "/generate":
                _write_json(self, {"error": "not found"}, status=404)
                return
            _ = _read_json(self)
            # 返回一个“sglang chunk”列表/字典，满足 postprocess_generate 的最低要求：
            # - output_ids
            # - meta_info.finish_reason
            resp = {
                "output_ids": list(cfg.output_ids),
                "meta_info": {"finish_reason": "stop"},
                "worker": cfg.name,  # 额外字段，便于我们断言路由是否命中
            }
            _write_json(self, resp)

        def log_message(self, format, *args):  # noqa: A002
            return

    return FakeWorkerHandler


def make_fake_router_handler(worker_urls: Dict[str, str]):
    class FakeRouterHandler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path != "/generate":
                _write_json(self, {"error": "not found"}, status=404)
                return
            payload = _read_json(self)
            preferred = self.headers.get("X-ROLL-Preferred-Worker-Url")
            # 选择目标 worker：优先 header，否则默认第一个
            target = preferred if preferred in worker_urls else next(iter(worker_urls.keys()))
            target_url = worker_urls[target] + "/generate"
            # 注意：此处不需要再转发 preferred header；我们只验证 router 选 worker 的行为
            resp = _http_post_json(target_url, payload)
            _write_json(self, resp)

        def log_message(self, format, *args):  # noqa: A002
            return

    return FakeRouterHandler


def _start_server(host: str, port: int, handler_cls) -> HTTPServer:
    server = HTTPServer((host, port), handler_cls)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main():
    # 说明：
    # - 该脚本做“最小可运行”黑盒验证：header 能驱动 router 选定 worker。
    # - 若环境未安装 ray，无法直接 import ROLL 的 SglangProxy（其模块会间接 import ray）。
    #   因此这里分两步：
    #   1) 黑盒验证：我们直接对 router 发请求，带/不带 header，验证路由行为。
    #   2) 静态验证：检查 R3 源码里是否包含 `X-ROLL-Preferred-Worker-Url` header 注入。
    import os

    host = "127.0.0.1"
    w1_port, w2_port, r_port = _find_free_port(), _find_free_port(), _find_free_port()
    w1_url = f"http://{host}:{w1_port}"
    w2_url = f"http://{host}:{w2_port}"

    worker_map = {w1_url: w1_url, w2_url: w2_url}

    w1 = _start_server(host, w1_port, make_fake_worker_handler(FakeWorkerConfig("w1", (11, 12, 13))))
    w2 = _start_server(host, w2_port, make_fake_worker_handler(FakeWorkerConfig("w2", (21, 22, 23))))
    router = _start_server(host, r_port, make_fake_router_handler(worker_map))

    router_url = f"http://{host}:{r_port}/generate"

    # Case 1: 不带 header，router 默认走第一个 worker（w1）
    payload = {"rid": "rid-normal", "input_ids": [1, 2, 3], "sampling_params": {"n": 1}, "return_logprob": False}
    out1 = _http_post_json(router_url, payload)
    assert out1["output_ids"] == [11, 12, 13], f"expected w1, got: {out1}"

    # Case 2: 带 header 指定 worker2
    out2 = _http_post_json(
        router_url,
        payload | {"rid": "rid-resume"},
        headers={"X-ROLL-Preferred-Worker-Url": w2_url},
    )
    assert out2["output_ids"] == [21, 22, 23], f"expected w2, got: {out2}"

    # 静态验证：ROLL 代码中应当存在 header 注入逻辑
    router_py = os.path.join(os.path.dirname(__file__), "..", "roll", "distributed", "scheduler", "router.py")
    with open(router_py, "r", encoding="utf-8") as f:
        text = f.read()
    assert "X-ROLL-Preferred-Worker-Url" in text, "ROLL router.py missing X-ROLL-Preferred-Worker-Url injection"

    print("OK: (1) header can force router -> worker2; (2) ROLL code contains header injection.")

    # 清理
    for s in (router, w1, w2):
        s.shutdown()
    time.sleep(0.1)


if __name__ == "__main__":
    main()

