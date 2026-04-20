import asyncio
import itertools
import math
import time
import uuid
import httpx
import weakref
from dataclasses import dataclass
from abc import abstractmethod
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import ray

from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.resume_priority import (
    RequestPriorityWeights,
    ResumeScoreWeights,
    compute_request_priority,
    compute_resume_score,
)
from roll.configs.base_config import RouterArguments
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.functionals import gather_unpadded_input_ids
from roll.utils.checkpoint_manager import download_model
from roll.utils.logging import get_logger


logger = get_logger()


@dataclass
class PendingTrajectoryRequest:
    request_id: str
    uid: int
    request_type: str
    route_meta: Dict[str, Any]
    enqueue_ts: float
    enqueue_seq: int
    base_priority: float


def extract_roll_route_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pop and return route metadata injected by runtime."""
    route_meta = payload.pop("_roll_route_meta", None)
    return route_meta if isinstance(route_meta, dict) else {}

def is_report_data_finished(data: DataProto) -> bool:
    finish_reasons = data.meta_info.get("finish_reasons", [])
    assert isinstance(finish_reasons, list), f"{finish_reasons}"
    assert all(isinstance(finish_reason, str) for finish_reason in finish_reasons), f"{finish_reasons}"
    return not any(finish_reason == "abort" for finish_reason in finish_reasons)

def raise_for_status(response: httpx.Response):
    if not response.is_success:
        try:
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(str(e))

async def wait_sglang_router_ready(router_process, url):
    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        for attempt in range(60):
            await asyncio.sleep(1)
            try:
                response = await client.get(url)
                if response.status_code in [200, 404]:
                    break
                else:
                    logger.info(f"Waiting for sglang router {url} to ready ({attempt=}) (status={response.status_code})...")
                    raise_for_status(response)
                assert router_process.is_alive()
            except httpx.ConnectError:
                logger.info(f"Waiting for sglang router {url} to start ({attempt=})...")

async def wait_sglang_router_workflow(router_url, expected):
    expected = set(expected)
    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        while True:
            await asyncio.sleep(3)
            response = await client.get(f"{router_url}/workers")
            raise_for_status(response)
            response = response.json()
            if {worker["url"] for worker in response["workers"]} == expected:
                break
            logger.info(f"Waiting for sglang router worker workflow {router_url} ready, "
                        f"{expected=}, current count={response['total']}, workers={response['workers']} ...")

class RouterManager:
    def __init__(self, actor_cluster: Cluster, router_args: RouterArguments, num_gpus_per_node: int):
        self.actor_cluster = actor_cluster
        self.workers = actor_cluster.workers

        self.strategy_name = actor_cluster.worker_config.strategy_args.strategy_name 
        self.model_path = download_model(actor_cluster.worker_config.model_args.model_name_or_path)
        self.tokenizer = default_tokenizer_provider(model_args=actor_cluster.worker_config.model_args)

        router_name = router_args.router_name
        if router_name == "PromptAffinityRouter":
            self.router_cls = PromptAffinityRouter
        elif router_name == "EnvAffinityRouter":
            self.router_cls = EnvAffinityRouter
        else:
            self.router_cls = SglangRouter
        assert self.router_cls is not SglangRouter or self.strategy_name == "sglang"
        assert (self.router_cls is SglangRouter) == (actor_cluster.worker_config.strategy_args.strategy_config.get("grpc_mode", None) is not None) # xnor
        logger.info(f"RouterManager use router {self.router_cls.__name__}")
        self.router: Router = self.router_cls(router_manager=self, workers=self.workers, model_path=self.model_path, router_args=router_args)

        self.inflight_requests = set()
        self.need_suspend = False
        self.need_shutdown = False
        self.suspend_notifier = asyncio.Event()
        self.empty_notifier = asyncio.Event()

        self.partial_gpu_manager = PartialGPUManager(actor_cluster=actor_cluster, router=self.router, num_gpus_per_node=num_gpus_per_node)

    async def initialize(self):
        await self.router.initialize()

    def router_meta(self):
        return {
            "strategy_name": self.strategy_name,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "sglang_router": self.router_cls is SglangRouter,
            "router_ip": self.router.router_ip if self.router_cls is SglangRouter else None,
            "router_port": self.router.router_port if self.router_cls is SglangRouter else None,
            "worker_urls": self.router.worker_urls if self.router_cls is SglangRouter else None,
        }

    @classmethod
    def create_client_sync(cls, self) -> "RouterClient":
        if isinstance(self, ray.actor.ActorHandle):
            meta = ray.get(self.router_meta.remote())
            proxy_cls = RayProxy
        elif isinstance(self, cls):
            meta = self.router_meta()
            proxy_cls = InprocProxy
        else:
            raise ValueError(f"self {self} is not a ray actor or RouterManager")

        proxy = proxy_cls(self)
        if meta["sglang_router"]:
            proxy = SglangProxy(proxy, meta)
        return RouterClient(proxy, meta)

    @classmethod
    async def create_client(cls, self) -> "RouterClient":
        """
        self may be a ray actor or normal class.
        """
        if isinstance(self, ray.actor.ActorHandle):
            meta = await self.router_meta.remote()
            proxy_cls = RayProxy
        elif isinstance(self, cls):
            meta = self.router_meta()
            proxy_cls = InprocProxy
        else:
            raise ValueError(f"self {self} is not a ray actor or RouterManager")

        proxy = proxy_cls(self)
        if meta["sglang_router"]:
            proxy = SglangProxy(proxy, meta)
        return RouterClient(proxy, meta)

    async def generate_request(self, payload, request_id, uid):
        return await self.router.generate_request(payload=payload, request_id=request_id, uid=uid)

    async def abort_requests(self, request_ids, uid):
        return await self.router.abort_requests(request_ids, uid)

    async def abort_all(self):
        logger.info(f"abort all requests, remaining requests: {len(self.inflight_requests)}")
        return await self.router.abort_all(list(self.inflight_requests))

    async def on_send_request(self, request_id) -> bool:
        while self.need_suspend:
            await self.suspend_notifier.wait()
        if self.need_shutdown:
            return False
        self.inflight_requests.add(request_id)
        return True

    async def on_request_routed(self, request_id):
        self.inflight_requests.remove(request_id)
        self.empty_notifier.set()

    def suspend(self):
        """
        Suspend all running requests.

        All following call of generate will be blocked until resume.
        """
        if self.need_suspend:
            return
        self.suspend_notifier.clear()
        self.need_suspend = True

    def resume(self):
        if not self.need_suspend:
            return
        self.need_suspend = False
        self.suspend_notifier.set()

    async def shutdown(self):
        self.need_shutdown = True
        await self.abort_all()
        self.resume()
        await self.wait_complete()

    async def wait_complete(self):
        """
        Wait until all running requests are finished (no matter whether suspended or not).
        """
        logger.info(f"RouterManager: wait all requests complete {self.inflight_requests=}")
        while len(self.inflight_requests) > 0:
            self.empty_notifier.clear()
            await self.empty_notifier.wait()
        logger.info(f"RouterManager: all requests completed")

    def size(self):
        return len(self.inflight_requests)

    async def shrink_workers(self, target_gpus: List[int]) -> Dict[str, Any]:
        logger.info(f"RouterManager shrink_workers {target_gpus=}")
        return await self.partial_gpu_manager.shrink_workers(target_gpus)

    async def expand_workers(self, target_gpus: List[int], skip_load: bool = False) -> Dict[str, Any]:
        logger.info(f"RouterManager expand_workers {target_gpus=}")
        return await self.partial_gpu_manager.expand_workers(target_gpus, skip_load)

    def collect_metrics(self) -> Dict[str, float]:
        return self.router.collect_metrics()

class PartialGPUManager:
    def __init__(self, actor_cluster, router, num_gpus_per_node: int):
        self.infer_cluster = actor_cluster
        self.router = router
        self.num_gpus_per_node = num_gpus_per_node

    def _get_gpus_for_dp_rank(self, dp_rank: int) -> List[int]:
        """Map DP rank to GPU IDs using cluster's device info.

        Args:
            dp_rank: Data parallel rank index (0 to dp_size-1)

        Returns:
            List of GPU IDs used by this DP rank's workers

        Example:
            # Pure DP: rank == dp_rank
            # DP rank 0 uses GPUs [0], DP rank 1 uses GPUs [1], etc.
            gpus = self._get_gpus_for_dp_rank(dp_rank=0)
            # Returns: [0]
        """
        # In agentic pipeline (pure DP): rank == dp_rank, so directly access rank2devices
        devices_info = self.infer_cluster.rank2devices[dp_rank]

        # Extract GPU IDs: gpu_id = node_rank * num_gpus_per_node + gpu_rank
        gpu_ids = []
        for device in devices_info:
            gpu_id = device["node_rank"] * self.num_gpus_per_node + device["gpu_rank"]
            gpu_ids.append(gpu_id)

        return sorted(set(gpu_ids))  # Remove duplicates and sort

    def _validate_target_gpus(self, target_gpus: List[int], mode: str) -> None:
        """Validate target_gpus input for shrink/expand operations.

        Args:
            target_gpus: List of GPU IDs to free (shrink) or restore (expand)
            mode: Operation mode ("shrink" or "expand")

        Raises:
            ValueError: If target_gpus is empty, has duplicates, or mode is invalid

        Example:
            self._validate_target_gpus([4, 5, 6, 7], mode="shrink")
            # Validates successfully

            self._validate_target_gpus([], mode="shrink")
            # Raises: ValueError("[shrink] target_gpus cannot be empty")

            self._validate_target_gpus([4, 4, 5], mode="expand")
            # Raises: ValueError("[expand] target_gpus has duplicates: [4, 4, 5]")
        """
        # VAL: VAL_NON_EMPTY
        if not target_gpus:
            raise ValueError(f"[{mode}] target_gpus cannot be empty")

        # VAL: VAL_NO_DUPLICATES
        if len(target_gpus) != len(set(target_gpus)):
            raise ValueError(f"[{mode}] target_gpus has duplicates: {target_gpus}")

        if mode not in ("shrink", "expand"):
            raise ValueError(f"Invalid mode: {mode}")

    def _validate_calculated_ranks(self, ranks: List[int], mode: str) -> None:
        """Validate calculated DP ranks against current active_dp_ranks state.

        Args:
            ranks: List of DP ranks calculated from target_gpus
            mode: Operation mode ("shrink" or "expand")

        Raises:
            ValueError: If ranks is empty, contains out-of-range values,
                       or violates state consistency (shrink: must be active,
                       expand: must be inactive)

        Example:
            # Shrink validation
            self.active_dp_ranks = {0, 1, 2, 3}
            self._validate_calculated_ranks([2, 3], mode="shrink")
            # Validates successfully (ranks 2, 3 are active)

            self._validate_calculated_ranks([4], mode="shrink")
            # Raises: ValueError("[shrink] DP rank 4 not active")

            # Expand validation
            self.active_dp_ranks = {0, 1}
            self._validate_calculated_ranks([2, 3], mode="expand")
            # Validates successfully (ranks 2, 3 are inactive)

            self._validate_calculated_ranks([0], mode="expand")
            # Raises: ValueError("[expand] DP rank 0 already active")
        """
        # VAL: VAL_NON_EMPTY
        if not ranks:
            raise ValueError(f"[{mode}] Calculated ranks list is empty")

        # VAL: VAL_INT_RANGE
        for dp_rank in ranks:
            if not (0 <= dp_rank < self.infer_cluster.world_size):
                raise ValueError(f"[{mode}] DP rank {dp_rank} out of range [0, {self.infer_cluster.world_size})")

        # AST: State consistency

        # TODO: fix this validation and move to EnvAffinityRouter
        # for dp_rank in ranks:
        #     if dp_rank not in self.active_dp_ranks:
        #         raise ValueError(f"DP rank {dp_rank} not active {mode=}")

    async def shrink_workers(self, target_gpus: List[int]) -> Dict[str, Any]:
        """Complete atomic shrink operation: validate → rebalance → offload → update routing.

        Orchestrates the full worker shrink process:
        1. Validates target_gpus input
        2. Calculates DP ranks to offload based on GPU overlap
        3. Validates calculated ranks against active state
        4. Do shrink:
           - Rebalances routing (aborts requests on shrinking workers)
           - Offloads model states from shrinking workers
        5. Returns metrics for monitoring

        Args:
            target_gpus: GPU IDs to free (e.g., [4, 5, 6, 7] to free second half of 8 GPUs)

        Returns:
            Metrics dict containing:
                - "aborted": Number of requests aborted during rebalancing
                - "remapped": Number of src_ranks remapped (cleared from routing)
                - "shrink_duration_ms": Total operation time in milliseconds
                - "offload_ranks": List of DP ranks that were offloaded

        Raises:
            ValueError: If target_gpus invalid (empty, duplicates) or
                       calculated ranks invalid (not active, out of range)
            RuntimeError: If rebalance or offload operations fail

        Example:
            # Shrink to free GPUs [4, 5, 6, 7] (second half of 8-GPU setup)
            result = await scheduler.shrink_workers([4, 5, 6, 7])
            # Returns: {"aborted": 10, "remapped": 5, "shrink_duration_ms": 2340.5, "offload_ranks": [2, 3]}

        Side Effects:
            - Updates active_dp_ranks (removes offload_ranks)
            - Aborts in-flight requests on shrinking workers
            - Clears src_rank mappings for remapped environments
            - Offloads model states from shrinking workers to CPU
        """
        start_time = time.time()

        # VAL: VAL_NON_EMPTY, VAL_NO_DUPLICATES
        self._validate_target_gpus(target_gpus, mode="shrink")
        # Calculate DP ranks to offload
        target_gpus = set(target_gpus)
        offload_ranks = [dp for dp in range(self.infer_cluster.world_size)
                         if set(self._get_gpus_for_dp_rank(dp)).intersection(target_gpus)]

        # VAL: VAL_NON_EMPTY, state consistency check
        self._validate_calculated_ranks(offload_ranks, mode="shrink")

        result = await self.router.rebalance_on_shrink(offload_ranks)

        # release the lock before blocking offload so that active dp rank can work immediately
        # Offload states from target workers
        offload_refs = self.infer_cluster.offload_states_partial(offload_ranks, blocking=False)
        await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in offload_refs])

        return {**result, "shrink_duration_ms": (time.time() - start_time) * 1000,
                "offload_ranks": offload_ranks}

    async def expand_workers(self, target_gpus: List[int], skip_load: bool = False) -> Dict[str, Any]:
        """Complete atomic expand operation: validate → load → rebalance → update routing.

        Orchestrates the full worker expand process:
        1. Validates target_gpus input
        2. Calculates DP ranks to restore based on GPU overlap
        3. Validates calculated ranks against active state (skip if skip_load=True)
        4. Do expand:
           - Loads model states on expanding workers (skip if skip_load=True)
           - Rebalances routing (proportionally redistributes requests)
        5. Returns metrics for monitoring

        Args:
            target_gpus: GPU IDs to restore (e.g., [4, 5, 6, 7] to restore second half of 8 GPUs)
            skip_load: If True, skip model loading and validation (use when model_update already loaded states).
                      This only updates active_dp_ranks to restore routing state without re-loading models.

        Returns:
            Metrics dict containing:
                - "aborted": Number of requests aborted during rebalancing (proportional redistribution)
                - "remapped": Number of src_ranks remapped (cleared from routing)
                - "expand_duration_ms": Total operation time in milliseconds
                - "load_ranks": List of DP ranks that were restored

        Raises:
            ValueError: If target_gpus invalid (empty, duplicates) or
                       calculated ranks invalid (already active, out of range)
            RuntimeError: If load or rebalance operations fail

        Example:
            # Expand to restore GPUs [4, 5, 6, 7] (second half of 8-GPU setup)
            result = await scheduler.expand_workers([4, 5, 6, 7])
            # Returns: {"aborted": 3, "remapped": 3, "expand_duration_ms": 1850.2, "load_ranks": [2, 3]}

            # After model_update already loaded states to all GPUs, just restore routing:
            result = await scheduler.expand_workers([4, 5, 6, 7], skip_load=True)

        Side Effects:
            - Updates active_dp_ranks (adds load_ranks)
            - Loads model states from CPU to expanding workers (unless skip_load=True)
            - Aborts some requests from old workers for proportional rebalancing
            - Clears src_rank mappings for rebalanced environments (will route to new workers)
        """
        start_time = time.time()

        # VAL: VAL_NON_EMPTY, VAL_NO_DUPLICATES
        self._validate_target_gpus(target_gpus, mode="expand")

        # Calculate DP ranks to restore
        target_gpus = set(target_gpus)
        load_ranks = [dp for dp in range(self.infer_cluster.world_size)
                      if set(self._get_gpus_for_dp_rank(dp)).issubset(target_gpus)]

        # VAL: VAL_NON_EMPTY, state consistency check
        # Skip validation when skip_load=True because ranks may already be "active" in cluster
        # (model states loaded by model_update) but not tracked in active_dp_ranks yet
        if not skip_load:
            self._validate_calculated_ranks(load_ranks, mode="expand")
            load_refs = self.infer_cluster.load_states_partial(load_ranks, blocking=False)
            await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in load_refs])

        result = await self.router.rebalance_on_expand(load_ranks)

        return {**result, "expand_duration_ms": (time.time() - start_time) * 1000,
                "load_ranks": load_ranks}

class RouterProxy:
    """
    Proxy to RouterManager
    """
    @abstractmethod
    async def generate_request(self, payload, request_id, uid):
        pass

    @abstractmethod
    async def on_send_request(self, request_id):
        pass

    @abstractmethod
    async def on_request_routed(self, request_id):
        pass

    def generate_request_sync(self, payload, request_id, uid):
        raise NotImplementedError

    def on_send_request_sync(self, request_id):
        raise NotImplementedError

    def on_request_routed_sync(self, request_id):
        raise NotImplementedError

class InprocProxy(RouterProxy):
    def __init__(self, router_manager: RouterManager):
        self.router_manager = router_manager

    async def generate_request(self, payload, request_id, uid):
        return await self.router_manager.generate_request(payload=payload, request_id=request_id, uid=uid)

    async def on_send_request(self, request_id):
        return await self.router_manager.on_send_request(request_id)

    async def on_request_routed(self, request_id):
        return await self.router_manager.on_request_routed(request_id)

class RayProxy(RouterProxy):
    def __init__(self, router_manager: RouterManager):
        self.router_manager = router_manager

    async def generate_request(self, payload, request_id, uid):
        return await self.router_manager.generate_request.remote(payload=payload, request_id=request_id, uid=uid)

    async def on_send_request(self, request_id):
        return await self.router_manager.on_send_request.remote(request_id)

    async def on_request_routed(self, request_id):
        return await self.router_manager.on_request_routed.remote(request_id)

    def generate_request_sync(self, payload, request_id, uid):
        return ray.get(self.router_manager.generate_request.remote(payload=payload, request_id=request_id, uid=uid))

    def on_send_request_sync(self, request_id):
        return ray.get(self.router_manager.on_send_request.remote(request_id))

    def on_request_routed_sync(self, request_id):
        return ray.get(self.router_manager.on_request_routed.remote(request_id))

class SglangProxy(RouterProxy):
    def __init__(self, proxy: RouterProxy, router_meta):
        self.proxy = proxy
        self.router_ip = router_meta["router_ip"]
        self.router_port = router_meta["router_port"]
        self.worker_urls = router_meta["worker_urls"]
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(None))
        self.client_sync = httpx.Client(timeout=httpx.Timeout(None))

    def _build_router_headers(self, payload: Dict[str, Any]) -> Dict[str, str]:
        route_meta = payload.pop("_roll_route_meta", None)
        if not isinstance(route_meta, dict):
            return {}

        headers: Dict[str, str] = {}
        request_type = route_meta.get("request_type")
        if isinstance(request_type, str) and request_type:
            headers["X-ROLL-Request-Type"] = request_type

        # Strong affinity for resume requests: hint sglang-router to route to the previous backend.
        # We encode the preferred worker by URL to avoid index/order dependency.
        if request_type == "resume":
            last_backend_id = route_meta.get("last_backend_id")
            if isinstance(last_backend_id, int) and 0 <= last_backend_id < len(self.worker_urls):
                headers["X-ROLL-Preferred-Worker-Url"] = self.worker_urls[last_backend_id]

        return headers

    async def generate_request(self, payload, request_id, uid):
        from roll.distributed.strategy.sglang_strategy import postprocess_generate
        assert "multi_modal_data" not in payload
        url = f"http://{self.router_ip}:{self.router_port}/generate"
        headers = self._build_router_headers(payload)
        response = await self.client.post(url, json=payload, headers=headers)
        raise_for_status(response)
        response = response.json()
        response = response if isinstance(response, list) else [response]
        return postprocess_generate(response)

    async def on_send_request(self, request_id):
        return await self.proxy.on_send_request(request_id)

    async def on_request_routed(self, request_id):
        return await self.proxy.on_request_routed(request_id)

    def generate_request_sync(self, payload, request_id, uid):
        from roll.distributed.strategy.sglang_strategy import postprocess_generate
        assert "multi_modal_data" not in payload
        url = f"http://{self.router_ip}:{self.router_port}/generate"
        headers = self._build_router_headers(payload)
        response = self.client_sync.post(url, json=payload, headers=headers)
        raise_for_status(response)
        response = response.json()
        response = response if isinstance(response, list) else [response]
        return postprocess_generate(response)

    def on_send_request_sync(self, request_id):
        return self.proxy.on_send_request_sync(request_id)

    def on_request_routed_sync(self, request_id):
        return self.proxy.on_request_routed_sync(request_id)

class RouterClient:
    def __init__(self, proxy, meta):
        self.proxy = proxy
        self.strategy_name = meta["strategy_name"]
        self.eos_token_id = meta["eos_token_id"]
        self.pad_token_id = meta["pad_token_id"]

    def _preprocess_generate(self, req: DataProto, request_id):
        if request_id is None:
            request_id = str(uuid.uuid4())
        payload = {"rid": str(request_id)}

        generation_config = req.meta_info.get("generation_config")
        collect_unfinished = req.meta_info.get("collect_unfinished", False)
        num_return_sequences = generation_config["num_return_sequences"]
        assert num_return_sequences == 1 or not collect_unfinished, "collect_unfinished is not supported in parallel sampling"

        max_new_tokens = req.meta_info.get("max_new_tokens", generation_config["max_new_tokens"])
        max_new_tokens = min(max_new_tokens, generation_config["max_new_tokens"])
        generation_config["max_new_tokens"] = max_new_tokens

        generation_config["eos_token_id"] = [self.eos_token_id, self.pad_token_id]
        generation_config["pad_token_id"] = self.pad_token_id

        if "multi_modal_data" in req.non_tensor_batch:
            multi_modal_data = req.non_tensor_batch["multi_modal_data"]
            assert len(multi_modal_data) == 1
            payload["multi_modal_data"] = multi_modal_data[0]
        else:
            input_ids = req.batch["input_ids"]
            assert not collect_unfinished or input_ids.size(0) == 1
            attention_mask = req.batch["attention_mask"]
            input_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
            payload["input_ids"] = input_ids[0]

        route_meta = {}
        for key in (
            "trajectory_id",
            "request_type",
            "resume_generation",
            "pause_ts",
            "pause_age_s",
            "history_len_tokens",
            "last_backend_id",
            "tool_type",
            "fairness_bucket",
        ):
            if key in req.meta_info:
                route_meta[key] = req.meta_info[key]
        if route_meta and "history_len_tokens" not in route_meta and "input_ids" in payload:
            route_meta["history_len_tokens"] = len(payload["input_ids"])
        # Runtime-only: EnvAffinityRouter / PromptAffinityRouter pop this before calling workers.
        # SGLang never sees it if routers run first; sglang_strategy also strips as a safeguard.
        if route_meta:
            payload["_roll_route_meta"] = route_meta

        match self.strategy_name:
            case "sglang":
                from roll.distributed.strategy.sglang_strategy import create_sampling_params_for_sglang
                sampling_params = create_sampling_params_for_sglang(gen_kwargs=generation_config)
                payload["sampling_params"] = sampling_params
                payload["return_logprob"] = generation_config.get("logprobs", 0) is not None
            case "vllm":
                from roll.distributed.strategy.vllm_strategy import create_sampling_params_for_vllm
                # vllm is hard coded to return logprob
                sampling_params = create_sampling_params_for_vllm(generation_config, collect_unfinished)
                payload["sampling_params"] = sampling_params
            case _:
                raise NotImplementedError(f"strategy {self.strategy_name} is not supported")
        return payload, request_id

    def _postprocess_generate(self, req, response):
        output_data = DataProto(meta_info=req.meta_info)
        output_data.meta_info["finish_reasons"] = response["finish_reasons"]
        output_data.meta_info["output_token_ids"] = response["output_token_ids"]
        output_data.meta_info["output_logprobs"] = response.get("output_logprobs", None)
        output_data.meta_info["eos_token_id"] = [self.eos_token_id, self.pad_token_id]
        output_data.meta_info["pad_token_id"] = self.pad_token_id
        if "selected_backend_id" in response:
            output_data.meta_info["selected_backend_id"] = response["selected_backend_id"]
        return output_data

    async def generate_request(self, req: DataProto, request_id, uid):
        """
        Request format is adapted for sglang generate (specificly, use rid rather than request_id),
        which can be directly used by SglangRouter.
        Request is expected to be scalar (single request).

        Response format is adapted for ROLL DataProto.
        Response is expected to be vector (expanded for parallel sample).
        """
        payload, request_id = self._preprocess_generate(req, request_id)

        if not await self.proxy.on_send_request(request_id):
            return None # shutdown
        try:
            response = await self.proxy.generate_request(payload=payload, request_id=request_id, uid=uid)
        finally:
            await self.proxy.on_request_routed(request_id)

        return self._postprocess_generate(req, response)

    def generate_request_sync(self, req: DataProto, request_id, uid):
        payload, request_id = self._preprocess_generate(req, request_id)

        if not self.proxy.on_send_request_sync(request_id):
            return None # shutdown
        try:
            response = self.proxy.generate_request_sync(payload=payload, request_id=request_id, uid=uid)
        finally:
            self.proxy.on_request_routed_sync(request_id)

        return self._postprocess_generate(req, response)

class Router:
    def __init__(self, router_manager, workers, model_path, router_args: RouterArguments):
        self.router_manager_ref = weakref.ref(router_manager)
        self.workers = workers
        self.model_path = model_path
        self.router_args = router_args

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def generate_request(self, payload, request_id, uid):
        pass

    @abstractmethod
    async def abort_requests(self, request_ids, uid):
        pass

    @abstractmethod
    async def abort_all(self, request_ids):
        pass

    async def rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        raise NotImplementedError

    async def rebalance_on_expand(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        raise NotImplementedError

    def collect_metrics(self) -> Dict[str, float]:
        return {}

class SglangRouter(Router):
    """
    Wrap of https://docs.sglang.io/advanced_features/router.html#api-surface

    This is act as a client to sglang-router, can instantiate one SglangRouterClient for every env,
    """
    async def initialize(self):
        self.router_ip = Worker.get_node_ip()
        self.router_port = Worker.get_free_port()

        self.client = httpx.AsyncClient(timeout=httpx.Timeout(None))

        self.worker_urls = await asyncio.gather(
            *[
                worker.get_url.remote()
                for worker in self.workers
            ]
        )
        self.http_mode = False if self.worker_urls[0].startswith("grpc") else True
        assert self.http_mode

        import multiprocessing
        from sglang_router.launch_router import RouterArgs, launch_router

        multiprocessing.set_start_method("spawn")

        router_config = {
            "host": self.router_ip,
            "port": self.router_port,
            "prometheus_port": Worker.get_free_port(),
            "log_level": "warn",
            "policy": "cache_aware",
            "request_timeout_secs": 1800,
            "max_concurrent_requests": -1,
            "dp_aware": False,
            "worker_urls": self.worker_urls,
        }
        extra_router_config = self.router_args.router_config
        if router_config:
            router_config.update(extra_router_config)
        router_args = RouterArgs(**router_config)
        self.router_process = multiprocessing.Process(
            target=launch_router,
            args=(router_args,),
            daemon=True
        )
        self.router_process.start()
        logger.info(f"Launch sglang-router {router_args=}")
        await wait_sglang_router_ready(self.router_process, f"http://{self.router_ip}:{self.router_port}")
        await wait_sglang_router_workflow(f"http://{self.router_ip}:{self.router_port}", self.worker_urls)

    async def generate_request(self, payload, request_id, uid):
        raise RuntimeError("SglangRouter.generate_request is not expected to be called directly, use RouterClient.")

    async def abort_requests(self, request_ids, uid):
        async def abort_request(self, url, request_id):
            response = await self.client.post(f"{url}/abort_request", json={"rid": request_id})
            raise_for_status(response)
        await asyncio.gather(
            *[
                abort_request(self, url=url, request_id=request_id)
                for request_id in request_ids for url in self.worker_urls
            ]
        )

    async def abort_all(self, request_ids):
        # Cannot use abort_all of sglang, because actor_cluster may be shared between different Routers.
        await self.abort_requests(request_ids, uid=None)

    async def abort_all_worker(self, url):
        # Can only be used when router is not shared between two scheudlers.
        response = await self.client.post(f"{url}/abort_request", json={"abort_all": True})
        raise_for_status(response)

    async def post_workers(self, urls):
        responses = await asyncio.gather(
            *[
                self.client.post(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    json={"url": url},
                )
                for url in urls
            ]
        )
        for response in responses:
            raise_for_status(response)

    async def delete_workers(self, urls):
        encoded_urls = [quote(url, safe="") for url in urls]
        responses = await asyncio.gather(
            *[self.client.delete(f"http://{self.router_ip}:{self.router_port}/workers/{url}") for url in encoded_urls]
        )
        for response in responses:
            raise_for_status(response)

    async def get_worker_loads(self, url):
        response = await self.client.get(f"{url}/get_load")
        raise_for_status(response)
        return response.json()

    async def wait_worker_complete(self, url):
        while True:
            loads = await self.get_worker_loads(url)
            if all(load["num_reqs"] == 0 and load["num_waiting_reqs"] == 0 for load in loads):
                break
            await asyncio.sleep(1)

    async def rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        shrink_urls = [self.worker_urls[dp_rank] for dp_rank in shrink_dp_ranks]

        router_manager: RouterManager = self.router_manager_ref()
        router_manager.suspend()

        await self.delete_workers(shrink_urls)
        logger.info(f"SglangRouter: delete workers on shrink {shrink_dp_ranks=} {shrink_urls=}")

        # FIXME: Do not abort and wait for all workers.
        # Because call wait_worker_complete of shrink workers may not be accurate. There may be
        # a client called on_request_routed but has not calling generate_request yet.
        # Instead, we use RouterManager.wait_complete to make sure no more requests to shrink workers.
        await asyncio.gather(*[self.abort_all_worker(url) for url in self.worker_urls])
        logger.info(f"SglangRouter: abort all requests on shrink {shrink_dp_ranks=} {shrink_urls=}")

        logger.info(f"SglangRouter: wait for running requests on shrink ")
        await router_manager.wait_complete()

        await wait_sglang_router_workflow(f"http://{self.router_ip}:{self.router_port}", {url for url in self.worker_urls if url not in shrink_urls})

        router_manager.resume()

        logger.info(f"SglangRouter: rebalance on shrink finish")

        return {"aborted": 0, "remapped": 0} # for compatibility

    async def rebalance_on_expand(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        expand_urls = [self.worker_urls[dp_rank] for dp_rank in expand_dp_ranks]

        await self.post_workers(expand_urls)
        logger.info(f"SglangRouter: post workers on expand {expand_dp_ranks=}")

        # simply abort all requests to let sglang-router to re-schedule
        await asyncio.gather(*[self.abort_all_worker(url) for url in self.worker_urls])
        logger.info(f"SglangRouter: aborted all requests on expand {expand_dp_ranks=}")

        # FIXME: assume expand all workers currently
        await wait_sglang_router_workflow(f"http://{self.router_ip}:{self.router_port}", self.worker_urls)

        return {"aborted": 0, "remapped": 0} # for compatibility

    def collect_metrics(self) -> Dict[str, float]:
        return {}

class PromptAffinityRouter(Router):
    """
    Schedule requests of the same prompt to the same worker. Choose worker using best fit
    strategy (using linear search for simplicity), blocking generate request if no worker available.

    Limit the number of running requests of each dp rank below max_running_requests.
    """
    async def initialize(self):
        self.max_running_requests = self.router_args.max_running_requests

        # key: dp_rank, value: num_inflight_requests
        self.worker_loads = {dp_rank: 0 for dp_rank in range(len(self.workers))}
        # cache-aware scheduling by uid
        self.id_to_dp_rank: Dict[int, int] = {}
        # dp_rank -> request_ids, used by abort_all
        self.dp_inflight_requests: List[int, Set[str]] = [set() for _ in self.workers]

        self.lock = asyncio.Lock()
        # used by acquire
        self.event = asyncio.Event()
        # used by reacquire
        self.worker_event = {dp_rank: asyncio.Event() for dp_rank in range(len(self.workers))}

    def __repr__(self):
        return f"worker loads: {self.worker_loads}"

    async def generate_request(self, payload, request_id, uid):
        _ = extract_roll_route_meta(payload)
        credit = payload["sampling_params"]["n"]
        dp_rank = None
        if uid not in self.id_to_dp_rank:
            # To prevent multiple generate requests for the same prompt.
            # It is safe and no performance issue to acquire lock here.
            # Because acquire is guaranteed to return as long as there has
            # one worker whose running_requests < max_running_requests no matter
            # how large credit is.
            async with self.lock:
                if uid not in self.id_to_dp_rank:
                    dp_rank = await self.acquire(credit=credit)
                    self.id_to_dp_rank[uid] = dp_rank
        if dp_rank is None:
            assert uid in self.id_to_dp_rank
            dp_rank = self.id_to_dp_rank[uid]
            assert dp_rank is not None
            await self.reacquire(dp_rank=dp_rank, credit=credit)
        try:
            self.dp_inflight_requests[dp_rank].add(request_id)
            # InferWorker.generate_request only return data with finish_reason=="abort" on abort
            # but not raise asyncio.CancelledError. This try finally block may be not necessary.
            response = await self.workers[dp_rank].generate_request.remote(payload)
            if isinstance(response, dict):
                response["selected_backend_id"] = dp_rank
            return response
            # TODO ray.cancel(ref) on asyncio.CancelledError
        finally:
            self.dp_inflight_requests[dp_rank].remove(request_id)
            self.release(dp_rank=dp_rank, credit=credit)

    async def abort_requests(self, request_ids, uid):
        assert uid is not None
        dp_rank = self.id_to_dp_rank[uid]
        await self.workers[dp_rank].abort_requests.remote(request_ids=request_ids)

    async def abort_all(self, request_ids):
        await asyncio.gather(
            *[
                self.workers[dp_rank].abort_requests.remote(list(request_ids))
                for dp_rank, request_ids in enumerate(self.dp_inflight_requests)
            ]
        )
        self.id_to_dp_rank.clear() # gc uid cache here

    async def acquire(self, credit: int) -> int:
        while True:
            # TODO add check of suspend here to stop early
            target = -1
            for dp_rank, running_requests in self.worker_loads.items():
                if running_requests >= self.max_running_requests:
                    continue
                if target == -1 or running_requests < self.worker_loads[target]:
                    target = dp_rank
            if target != -1:
                # may send more requests than max_running_requests,
                # i.e. worker_loads[target] + credit > max_running_requests
                self.worker_loads[target] += credit
                return target
            self.event.clear()
            await self.event.wait()

    async def reacquire(self, dp_rank: int, credit: int):
        assert dp_rank in self.worker_loads
        while True:
            # TODO add check of suspend here to stop early
            if self.worker_loads[dp_rank] < self.max_running_requests:
                self.worker_loads[dp_rank] += credit
                return
            self.worker_event[dp_rank].clear()
            await self.worker_event[dp_rank].wait()

    def release(self, dp_rank: int, credit: int):
        assert credit >= 0
        self.worker_loads[dp_rank] -= credit
        assert self.worker_loads[dp_rank] >= 0
        self.event.set()
        self.worker_event[dp_rank].set()

    def size(self):
        return sum(self.worker_loads.values())

    def full(self) -> bool:
        return all(running_requests >= self.max_running_requests for running_requests in self.worker_loads.values())

    def collect_metrics(self) -> Dict[str, float]:
        return {}

class EnvAffinityRouter(Router):
    """
    Schedule requests of the same (env) uid, to the same dp_rank.

    Choose dp_rank by RR for the first time.

    No rate limit now.

    Do not support partial rollout now.
    """
    async def initialize(self):
        self.src_rank2_dp_rank = {}
        self.request_id_2_src_rank: Dict[str, int] = {}  # Reverse lookup for abort
        self.running_requests: List[set[str]] = [set() for _ in range(len(self.workers))]
        self.worker_iter = itertools.cycle(range(len(self.workers)))

        # Active DP ranks for request routing
        self.active_dp_ranks: Set[int] = set(range(len(self.workers)))  # All ranks initially active
        self.routing_lock = asyncio.Lock()  # Protect routing updates
        router_config = self.router_args.router_config or {}
        self.enable_resume_priority = bool(router_config.get("enable_resume_priority", True))
        self.enable_resume_aware_routing = bool(router_config.get("enable_resume_aware_routing", False))
        self.enable_request_priority_queue = bool(router_config.get("enable_request_priority_queue", False))
        self.request_wait_aging_weight = float(router_config.get("request_wait_aging_weight", 0.1))
        self.normal_request_base_score = float(router_config.get("normal_request_base_score", 0.0))
        self.max_running_requests_per_worker = int(router_config.get("max_running_requests_per_worker", 0))
        self.resume_normal_quota = str(router_config.get("resume_normal_quota", "3:1"))
        self.normal_max_queue_wait_s = float(router_config.get("normal_max_queue_wait_s", 0.0))
        self.resume_max_queue_wait_s = float(router_config.get("resume_max_queue_wait_s", 0.0))
        self.enable_adaptive_quota = bool(router_config.get("enable_adaptive_quota", False))
        self.adaptive_quota_min_ratio = str(router_config.get("adaptive_quota_min_ratio", "1:1"))
        self.adaptive_quota_max_ratio = str(router_config.get("adaptive_quota_max_ratio", "10:1"))
        self.adaptive_quota_update_interval_s = float(router_config.get("adaptive_quota_update_interval_s", 1.0))
        self.adaptive_quota_use_affinity_signal = bool(router_config.get("adaptive_quota_use_affinity_signal", True))
        self.adaptive_quota_affinity_window = int(router_config.get("adaptive_quota_affinity_window", 128))
        self.adaptive_quota_min_feasible_rate = float(router_config.get("adaptive_quota_min_feasible_rate", 0.5))
        self.adaptive_quota_min_hit_rate = float(router_config.get("adaptive_quota_min_hit_rate", 0.5))
        self.request_score_weights = RequestPriorityWeights.from_config(
            router_config.get("request_score_weights", {})
        )
        self.resume_score_weights = ResumeScoreWeights.from_config(router_config.get("resume_score_weights", {}))
        self.force_migrate_age_s = float(router_config.get("force_migrate_age_s", 30.0))
        self.fairness_enable = bool(router_config.get("fairness_enable", False))
        self.fairness_boost_max = float(router_config.get("fairness_boost_max", 1.0))

        # Unified rollback switch: disable all resume-priority behaviors.
        if not self.enable_resume_priority:
            self.enable_resume_aware_routing = False
            self.enable_request_priority_queue = False
            self.enable_adaptive_quota = False
            self.normal_max_queue_wait_s = 0.0
            self.resume_max_queue_wait_s = 0.0
        # Pending requests split by request_type for soft-quota dispatch.
        self.pending_resume_requests: Dict[str, PendingTrajectoryRequest] = {}
        self.pending_normal_requests: Dict[str, PendingTrajectoryRequest] = {}
        self.cancelled_pending_requests: Set[str] = set()
        self.request_seq_counter = itertools.count()
        self.dispatch_condition = asyncio.Condition()
        self._quota_resume_target, self._quota_normal_target = self._parse_ratio(self.resume_normal_quota)
        self._quota_cursor = 0
        self._last_adaptive_quota_ts = 0.0
        window_size = max(1, self.adaptive_quota_affinity_window)
        self._resume_affinity_hit_window: Deque[int] = deque(maxlen=window_size)
        self._resume_affinity_feasible_window: Deque[int] = deque(maxlen=window_size)
        self._last_quota_decision_reason = "init"
        self._reset_resume_metrics()
        logger.info(
            "EnvAffinityRouter resume-aware config: "
            f"enable_resume_priority={self.enable_resume_priority}, "
            f"enabled={self.enable_resume_aware_routing}, "
            f"queue_enabled={self.enable_request_priority_queue}, "
            f"request_score_weights={self.request_score_weights}, "
            f"worker_score_weights={self.resume_score_weights}, "
            f"force_migrate_age_s={self.force_migrate_age_s}, "
            f"max_running_requests_per_worker={self.max_running_requests_per_worker}, "
            f"resume_normal_quota={self.resume_normal_quota}, "
            f"normal_max_queue_wait_s={self.normal_max_queue_wait_s}, "
            f"resume_max_queue_wait_s={self.resume_max_queue_wait_s}, "
            f"enable_adaptive_quota={self.enable_adaptive_quota}, "
            f"adaptive_quota_use_affinity_signal={self.adaptive_quota_use_affinity_signal}, "
            f"adaptive_quota_affinity_window={self.adaptive_quota_affinity_window}, "
            f"adaptive_quota_min_feasible_rate={self.adaptive_quota_min_feasible_rate}, "
            f"adaptive_quota_min_hit_rate={self.adaptive_quota_min_hit_rate}"
        )

    def _reset_resume_metrics(self):
        self.resume_total_requests = 0
        self.resume_affinity_hits = 0
        self.resume_migrations = 0
        self.resume_forced_migrations = 0
        self.resume_score_sum = 0.0
        self.resume_selected_worker_load_sum = 0.0
        self.resume_pause_age_sum = 0.0
        self.resume_queue_wait_sum = 0.0
        self.resume_wait_bucket_served: Dict[str, int] = defaultdict(int)
        self.normal_total_requests = 0
        self.normal_queue_wait_sum = 0.0
        self.queue_wait_bucket_served: Dict[str, int] = defaultdict(int)
        self.score_bucket_served: Dict[str, int] = defaultdict(int)
        self.resume_fallback_reason_count: Dict[str, int] = defaultdict(int)

    @staticmethod
    def _bucketize_queue_wait_s(queue_wait_s: float) -> str:
        s = max(0.0, float(queue_wait_s))
        if s < 0.01:
            return "lt_10ms"
        if s < 0.1:
            return "lt_100ms"
        if s < 0.5:
            return "lt_500ms"
        if s < 1.0:
            return "lt_1s"
        if s < 3.0:
            return "lt_3s"
        if s < 10.0:
            return "lt_10s"
        return "ge_10s"

    @staticmethod
    def _bucketize_score(score: float) -> str:
        v = float(score)
        if v < -5.0:
            return "lt_-5"
        if v < -1.0:
            return "lt_-1"
        if v < 0.0:
            return "lt_0"
        if v < 1.0:
            return "lt_1"
        if v < 5.0:
            return "lt_5"
        return "ge_5"

    def _affinity_feasible_proxy(self, pending: PendingTrajectoryRequest) -> int:
        """Proxy signal used for resume tie-break / metrics, not a guarantee of hit."""
        if pending.request_type != "resume":
            return 0
        route_meta = pending.route_meta
        pause_age = float(route_meta.get("pause_age_s", 0.0) or 0.0)
        if pause_age >= self.force_migrate_age_s:
            return 0
        last_backend_id = route_meta.get("last_backend_id")
        if not isinstance(last_backend_id, int):
            return 0
        if last_backend_id not in self.active_dp_ranks:
            return 0
        if not self._has_worker_capacity(last_backend_id):
            return 0
        return 1

    def _resume_fallback_reason(
        self,
        *,
        route_meta: Dict[str, Any],
        selected_dp_rank: int,
        previous_rank: Optional[int],
    ) -> str:
        """Best-effort reason for not sticking to previous backend for resume."""
        pause_age = float(route_meta.get("pause_age_s", 0.0) or 0.0)
        if pause_age >= self.force_migrate_age_s:
            return "forced_migrate"
        last_backend_id = route_meta.get("last_backend_id")
        if last_backend_id is None:
            return "no_hint"
        if not isinstance(last_backend_id, int) or last_backend_id not in self.active_dp_ranks:
            return "hint_inactive"
        if not self._has_worker_capacity(last_backend_id):
            return "hint_no_capacity"
        if previous_rank is not None and selected_dp_rank == previous_rank:
            return "hit"
        if selected_dp_rank != last_backend_id:
            # Hint existed and had capacity, but selection chose another rank (e.g., load/score).
            return "selected_other"
        return "hit"

    def _fairness_bonus(self, route_meta: Dict[str, Any]) -> float:
        if not self.fairness_enable:
            return 0.0
        bucket = route_meta.get("fairness_bucket")
        if not bucket:
            return 0.0
        served = self.resume_wait_bucket_served.get(bucket, 0)
        tracked_buckets = max(1, len(self.resume_wait_bucket_served))
        avg_served = max(1.0, self.resume_total_requests / tracked_buckets)
        deficit_ratio = max(0.0, (avg_served - served) / avg_served)
        return min(deficit_ratio * self.fairness_boost_max, self.fairness_boost_max)

    def _compute_request_base_priority(self, request_type: str, route_meta: Dict[str, Any]) -> float:
        if request_type != "resume":
            return self.normal_request_base_score
        pause_age = float(route_meta.get("pause_age_s", 0.0) or 0.0)
        history_len = float(route_meta.get("history_len_tokens", 0.0) or 0.0)
        hit_prob = 1.0 if route_meta.get("last_backend_id") is not None else 0.0
        rebuild_cost = history_len
        fairness_bonus = self._fairness_bonus(route_meta=route_meta)
        return compute_request_priority(
            pause_age_s=pause_age,
            history_len_tokens=history_len,
            hit_prob=hit_prob,
            rebuild_cost=rebuild_cost,
            fairness_bonus=fairness_bonus,
            weights=self.request_score_weights,
        )

    def _effective_request_priority(self, pending: PendingTrajectoryRequest) -> float:
        queue_wait = max(0.0, time.time() - pending.enqueue_ts)
        return pending.base_priority + self.request_wait_aging_weight * queue_wait

    @staticmethod
    def _parse_ratio(value: str) -> Tuple[int, int]:
        """Parse a quota ratio string like '3:1' -> (3, 1)."""
        raw = (value or "").strip()
        if not raw:
            return 1, 1
        parts = raw.split(":")
        if len(parts) != 2:
            return 1, 1
        try:
            a = max(0, int(parts[0]))
            b = max(0, int(parts[1]))
        except ValueError:
            return 1, 1
        if a == 0 and b == 0:
            return 1, 1
        return a, b

    def _oldest_queue_wait_s(self, pending: Dict[str, PendingTrajectoryRequest]) -> float:
        if not pending:
            return 0.0
        oldest_ts = min(req.enqueue_ts for req in pending.values())
        return max(0.0, time.time() - oldest_ts)

    @staticmethod
    def _safe_rate(window: Deque[int]) -> float:
        if not window:
            return 0.0
        return float(sum(window)) / float(len(window))

    def _affinity_window_rates(self) -> Tuple[float, float, int]:
        size = len(self._resume_affinity_hit_window)
        if size <= 0:
            return 0.0, 0.0, 0
        hit_rate = self._safe_rate(self._resume_affinity_hit_window)
        feasible_rate = self._safe_rate(self._resume_affinity_feasible_window)
        return hit_rate, feasible_rate, size

    def _maybe_update_adaptive_quota(self) -> None:
        if not self.enable_adaptive_quota:
            return
        now = time.time()
        if now - self._last_adaptive_quota_ts < self.adaptive_quota_update_interval_s:
            return
        self._last_adaptive_quota_ts = now
        min_r, min_n = self._parse_ratio(self.adaptive_quota_min_ratio)
        max_r, max_n = self._parse_ratio(self.adaptive_quota_max_ratio)
        resume_backlog = len(self.pending_resume_requests)
        normal_backlog = len(self.pending_normal_requests)
        if resume_backlog <= 0 and normal_backlog <= 0:
            return
        base_r, base_n = self._parse_ratio(self.resume_normal_quota)
        hit_rate, feasible_rate, win_size = self._affinity_window_rates()
        affinity_good = (
            (not self.adaptive_quota_use_affinity_signal)
            or (
                win_size > 0
                and feasible_rate >= self.adaptive_quota_min_feasible_rate
                and hit_rate >= self.adaptive_quota_min_hit_rate
            )
        )

        # Backlog-driven + affinity-gated ratio: only aggressively boost resume when affinity is effective.
        if normal_backlog > resume_backlog:
            self._quota_resume_target, self._quota_normal_target = min_r, max_n
            self._last_quota_decision_reason = "normal_backlog_dominant"
        elif resume_backlog > normal_backlog and affinity_good:
            self._quota_resume_target, self._quota_normal_target = max_r, min_n
            self._last_quota_decision_reason = "resume_backlog_and_affinity_good"
        elif resume_backlog > normal_backlog and not affinity_good:
            self._quota_resume_target, self._quota_normal_target = base_r, base_n
            self._last_quota_decision_reason = "resume_backlog_but_affinity_poor"
        else:
            self._quota_resume_target, self._quota_normal_target = base_r, base_n
            self._last_quota_decision_reason = "balanced_backlog"
        self._quota_cursor = 0

    def _pick_from_resume_queue(self) -> tuple[Optional[PendingTrajectoryRequest], Optional[int]]:
        if not self.pending_resume_requests:
            return None, None
        sorted_requests = sorted(
            self.pending_resume_requests.values(),
            # Tie-breaker: prefer affinity-feasible resumes (proxy), then older FIFO.
            key=lambda req: (self._effective_request_priority(req), self._affinity_feasible_proxy(req), -req.enqueue_seq),
            reverse=True,
        )
        for pending in sorted_requests:
            dp_rank = self._select_worker_for_request(pending)
            if dp_rank is not None:
                return pending, dp_rank
        return None, None

    def _pick_from_normal_queue(self) -> tuple[Optional[PendingTrajectoryRequest], Optional[int]]:
        if not self.pending_normal_requests:
            return None, None
        sorted_requests = sorted(
            self.pending_normal_requests.values(),
            key=lambda req: req.enqueue_seq,
        )
        for pending in sorted_requests:
            dp_rank = self._select_worker_for_request(pending)
            if dp_rank is not None:
                return pending, dp_rank
        return None, None

    def _pick_next_dispatchable_request(self) -> tuple[Optional[PendingTrajectoryRequest], Optional[int]]:
        if not self.pending_resume_requests and not self.pending_normal_requests:
            return None, None

        self._maybe_update_adaptive_quota()

        # Soft quota escape hatches (timeout): prevent starvation.
        if self.normal_max_queue_wait_s > 0 and self._oldest_queue_wait_s(self.pending_normal_requests) >= self.normal_max_queue_wait_s:
            selected, dp_rank = self._pick_from_normal_queue()
            if selected is not None:
                self._quota_cursor = (self._quota_cursor + 1) % max(1, self._quota_resume_target + self._quota_normal_target)
                return selected, dp_rank
        if self.resume_max_queue_wait_s > 0 and self._oldest_queue_wait_s(self.pending_resume_requests) >= self.resume_max_queue_wait_s:
            selected, dp_rank = self._pick_from_resume_queue()
            if selected is not None:
                self._quota_cursor = (self._quota_cursor + 1) % max(1, self._quota_resume_target + self._quota_normal_target)
                return selected, dp_rank

        # Skip-on-empty: never idle because of quota.
        if not self.pending_normal_requests:
            return self._pick_from_resume_queue()
        if not self.pending_resume_requests:
            return self._pick_from_normal_queue()

        # Quota cycle: [resume...][normal...]
        cycle_len = max(1, self._quota_resume_target + self._quota_normal_target)
        cursor = self._quota_cursor % cycle_len
        prefer_resume = cursor < max(0, self._quota_resume_target)

        if prefer_resume:
            selected, dp_rank = self._pick_from_resume_queue()
            if selected is None:
                selected, dp_rank = self._pick_from_normal_queue()
        else:
            selected, dp_rank = self._pick_from_normal_queue()
            if selected is None:
                selected, dp_rank = self._pick_from_resume_queue()

        if selected is not None:
            self._quota_cursor = (self._quota_cursor + 1) % cycle_len
        return selected, dp_rank

    @staticmethod
    def _build_abort_response() -> Dict[str, Any]:
        return {
            "finish_reasons": ["abort"],
            "output_token_ids": [[]],
            "output_logprobs": [[]],
        }

    def _has_worker_capacity(self, dp_rank: int) -> bool:
        if self.max_running_requests_per_worker <= 0:
            return True
        return len(self.running_requests[dp_rank]) < self.max_running_requests_per_worker

    def _select_worker_for_request(self, pending: PendingTrajectoryRequest) -> Optional[int]:
        src_rank = pending.uid
        request_type = pending.request_type
        route_meta = pending.route_meta
        if self.enable_resume_aware_routing and request_type == "resume":
            candidate_ranks = [r for r in self.active_dp_ranks if self._has_worker_capacity(r)]
            if not candidate_ranks:
                return None
            fallback_last_backend = self.src_rank2_dp_rank.get(src_rank)
            return max(
                candidate_ranks,
                key=lambda r: self._compute_resume_score(
                    r,
                    route_meta,
                    fallback_last_backend=fallback_last_backend,
                ),
            )

        # Baseline sticky routing for non-resume requests.
        mapped_rank = self.src_rank2_dp_rank.get(src_rank)
        if mapped_rank is None:
            candidate_ranks = [r for r in self.active_dp_ranks if self._has_worker_capacity(r)]
            if not candidate_ranks:
                return None
            mapped_rank = min(candidate_ranks, key=lambda r: len(self.running_requests[r]))
            self.src_rank2_dp_rank[src_rank] = mapped_rank
        elif not self._has_worker_capacity(mapped_rank):
            return None
        return mapped_rank

    def _record_resume_dispatch(
        self,
        *,
        src_rank: int,
        dp_rank: int,
        route_meta: Dict[str, Any],
        enqueue_ts: float,
        base_priority: float,
        previous_rank: Optional[int] = None,
    ):
        if previous_rank is None:
            previous_rank = self.src_rank2_dp_rank.get(src_rank)
        pause_age = float(route_meta.get("pause_age_s", 0.0) or 0.0)
        queue_wait_s = max(0.0, time.time() - enqueue_ts)
        self.resume_total_requests += 1
        self.resume_pause_age_sum += pause_age
        self.resume_selected_worker_load_sum += float(len(self.running_requests[dp_rank]))
        self.resume_queue_wait_sum += queue_wait_s
        self.resume_score_sum += base_priority + self.request_wait_aging_weight * queue_wait_s
        if previous_rank is not None and previous_rank == dp_rank:
            self.resume_affinity_hits += 1
        elif previous_rank is not None and previous_rank != dp_rank:
            self.resume_migrations += 1
            if pause_age >= self.force_migrate_age_s:
                self.resume_forced_migrations += 1
        bucket = route_meta.get("fairness_bucket")
        if bucket:
            self.resume_wait_bucket_served[bucket] += 1

        # Adaptive quota signal: rolling window of affinity feasibility & hit rate.
        # Feasible: we had a last_backend_id hint and we are not in forced-migrate window.
        last_backend_id = route_meta.get("last_backend_id")
        affinity_feasible = 1 if (pause_age < self.force_migrate_age_s and last_backend_id is not None) else 0
        affinity_hit = 1 if (previous_rank is not None and previous_rank == dp_rank) else 0
        self._resume_affinity_feasible_window.append(affinity_feasible)
        self._resume_affinity_hit_window.append(affinity_hit)

        # Request-level observability for both resume & normal.
        self.queue_wait_bucket_served[f"resume/{self._bucketize_queue_wait_s(queue_wait_s)}"] += 1
        eff_score = base_priority + self.request_wait_aging_weight * queue_wait_s
        self.score_bucket_served[f"resume/{self._bucketize_score(eff_score)}"] += 1

        reason = self._resume_fallback_reason(route_meta=route_meta, selected_dp_rank=dp_rank, previous_rank=previous_rank)
        if reason != "hit":
            self.resume_fallback_reason_count[reason] += 1

    def _record_normal_dispatch(self, *, enqueue_ts: float, base_priority: float) -> None:
        queue_wait_s = max(0.0, time.time() - enqueue_ts)
        self.normal_total_requests += 1
        self.normal_queue_wait_sum += queue_wait_s
        self.queue_wait_bucket_served[f"normal/{self._bucketize_queue_wait_s(queue_wait_s)}"] += 1
        eff_score = base_priority + self.request_wait_aging_weight * queue_wait_s
        self.score_bucket_served[f"normal/{self._bucketize_score(eff_score)}"] += 1

    def _compute_resume_score(
        self,
        dp_rank: int,
        route_meta: Dict[str, Any],
        fallback_last_backend: Optional[int] = None,
    ) -> float:
        pause_age = float(route_meta.get("pause_age_s", 0.0) or 0.0)
        history_len = float(route_meta.get("history_len_tokens", 0.0) or 0.0)
        last_backend_id = route_meta.get("last_backend_id", fallback_last_backend)
        if pause_age >= self.force_migrate_age_s:
            last_backend_id = None
        is_last_backend = 1.0 if last_backend_id == dp_rank else 0.0
        worker_load = float(len(self.running_requests[dp_rank]))
        fairness_bonus = self._fairness_bonus(route_meta=route_meta)
        return compute_resume_score(
            pause_age_s=pause_age,
            history_len_tokens=history_len,
            is_last_backend=is_last_backend,
            worker_load=worker_load,
            fairness_bonus=fairness_bonus,
            weights=self.resume_score_weights,
        )

    def _select_resume_dp_rank(self, src_rank: int, route_meta: Dict[str, Any]) -> int:
        candidate_ranks = [r for r in self.active_dp_ranks if self._has_worker_capacity(r)]
        if not candidate_ranks:
            raise RuntimeError("No active DP ranks with capacity")
        fallback_last_backend = self.src_rank2_dp_rank.get(src_rank)
        return max(
            candidate_ranks,
            key=lambda r: self._compute_resume_score(r, route_meta, fallback_last_backend=fallback_last_backend),
        )

    async def generate_request(self, payload, request_id, uid):
        route_meta = extract_roll_route_meta(payload)
        request_type = route_meta.get("request_type", "normal")
        src_rank = uid
        base_priority = self._compute_request_base_priority(request_type=request_type, route_meta=route_meta)

        if self.enable_request_priority_queue:
            pending = PendingTrajectoryRequest(
                request_id=request_id,
                uid=src_rank,
                request_type=request_type,
                route_meta=route_meta,
                enqueue_ts=time.time(),
                enqueue_seq=next(self.request_seq_counter),
                base_priority=base_priority,
            )
            async with self.dispatch_condition:
                if request_type == "resume":
                    self.pending_resume_requests[request_id] = pending
                else:
                    self.pending_normal_requests[request_id] = pending
                self.dispatch_condition.notify_all()
                try:
                    while True:
                        if request_id in self.cancelled_pending_requests:
                            self.cancelled_pending_requests.discard(request_id)
                            self.pending_resume_requests.pop(request_id, None)
                            self.pending_normal_requests.pop(request_id, None)
                            return self._build_abort_response()
                        selected, selected_dp_rank = self._pick_next_dispatchable_request()
                        if selected and selected.request_id == request_id and selected_dp_rank is not None:
                            dp_rank = selected_dp_rank
                            previous_rank = self.src_rank2_dp_rank.get(src_rank)
                            self.pending_resume_requests.pop(request_id, None)
                            self.pending_normal_requests.pop(request_id, None)
                            self.src_rank2_dp_rank[src_rank] = dp_rank
                            self.request_id_2_src_rank[request_id] = src_rank
                            self.running_requests[dp_rank].add(request_id)
                            if request_type == "resume":
                                self._record_resume_dispatch(
                                    src_rank=src_rank,
                                    dp_rank=dp_rank,
                                    route_meta=route_meta,
                                    enqueue_ts=pending.enqueue_ts,
                                    base_priority=base_priority,
                                    previous_rank=previous_rank,
                                )
                            else:
                                self._record_normal_dispatch(
                                    enqueue_ts=pending.enqueue_ts,
                                    base_priority=base_priority,
                                )
                            self.dispatch_condition.notify_all()
                            break
                        await self.dispatch_condition.wait()
                except asyncio.CancelledError:
                    self.pending_resume_requests.pop(request_id, None)
                    self.pending_normal_requests.pop(request_id, None)
                    self.cancelled_pending_requests.discard(request_id)
                    self.dispatch_condition.notify_all()
                    raise
        else:
            # Atomic routing assignment under lock to prevent TOCTOU race with shrink/expand
            async with self.routing_lock:
                if self.enable_resume_aware_routing and request_type == "resume":
                    previous_rank = self.src_rank2_dp_rank.get(src_rank)
                    dp_rank = self._select_resume_dp_rank(src_rank=src_rank, route_meta=route_meta)
                    self.src_rank2_dp_rank[src_rank] = dp_rank
                    self.request_id_2_src_rank[request_id] = src_rank
                    self.running_requests[dp_rank].add(request_id)
                    self._record_resume_dispatch(
                        src_rank=src_rank,
                        dp_rank=dp_rank,
                        route_meta=route_meta,
                        enqueue_ts=time.time(),
                        base_priority=base_priority,
                        previous_rank=previous_rank,
                    )
                else:
                    if src_rank not in self.src_rank2_dp_rank:
                        dp_rank = self._get_least_active_dp_rank()
                        self.src_rank2_dp_rank[src_rank] = dp_rank
                    dp_rank = self.src_rank2_dp_rank[src_rank]
                    self.request_id_2_src_rank[request_id] = src_rank
                    self.running_requests[dp_rank].add(request_id)

        try:
            response = await self.workers[dp_rank].generate_request.remote(payload)
            if isinstance(response, dict):
                response["selected_backend_id"] = dp_rank
            return response
        finally:
            if self.enable_request_priority_queue:
                async with self.dispatch_condition:
                    self.running_requests[dp_rank].discard(request_id)
                    # Cleanup tracking (on both success and abort paths)
                    self.request_id_2_src_rank.pop(request_id, None)
                    self.dispatch_condition.notify_all()
            else:
                self.running_requests[dp_rank].discard(request_id)
                self.request_id_2_src_rank.pop(request_id, None)

    async def abort_requests(self, request_ids, uid):
        raise NotImplementedError

    async def abort_all(self, request_ids):
        await asyncio.gather(*(
            self.workers[dp_rank].abort_requests.remote(list(self.running_requests[dp_rank]))
            for dp_rank in range(len(self.workers))
            if self.running_requests[dp_rank]
        ))
        if self.enable_request_priority_queue:
            async with self.dispatch_condition:
                for pending_request_id in itertools.chain(
                    self.pending_resume_requests.keys(), self.pending_normal_requests.keys()
                ):
                    self.cancelled_pending_requests.add(pending_request_id)
                self.pending_resume_requests.clear()
                self.pending_normal_requests.clear()
                self.dispatch_condition.notify_all()

    def _get_least_active_dp_rank(self) -> int:
        """Find DP rank with fewest assigned src_ranks (environments).

        Returns:
            DP rank with minimum src_rank count from src_rank2_dp_rank

        Raises:
            RuntimeError: If no active ranks

        Note:
            Counts unique src_ranks (environments) per worker, not in-flight requests.
            With sticky mapping, one src_rank generates multiple sequential requests.
        """
        candidate_ranks = list(self.active_dp_ranks)
        if not candidate_ranks:
            raise RuntimeError("No active DP ranks")
        # todo optimization: (yangpeng) not efficient, better to use counter for this
        # Count src_ranks per dp_rank
        src_rank_count = defaultdict(int)
        for src_rank, dp_rank in self.src_rank2_dp_rank.items():
            if dp_rank in self.active_dp_ranks:
                src_rank_count[dp_rank] += 1

        # Return dp_rank with minimum src_rank count
        return min(candidate_ranks, key=lambda r: src_rank_count[r])

    def collect_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "scheduler/router/pending_request_count": float(
                len(self.pending_resume_requests) + len(self.pending_normal_requests)
            ),
            "scheduler/router/pending_resume_request_count": float(len(self.pending_resume_requests)),
            "scheduler/router/pending_normal_request_count": float(len(self.pending_normal_requests)),
            "scheduler/router/quota_resume_target": float(self._quota_resume_target),
            "scheduler/router/quota_normal_target": float(self._quota_normal_target),
        }
        hit_rate, feasible_rate, win_size = self._affinity_window_rates()
        if win_size:
            metrics.update({
                "scheduler/router/resume_affinity_hit_rate_window": hit_rate,
                "scheduler/router/resume_affinity_feasible_rate_window": feasible_rate,
                "scheduler/router/resume_affinity_window_size": float(win_size),
            })
        for bucket, served in self.queue_wait_bucket_served.items():
            metrics[f"scheduler/router/queue_wait_bucket_served/{bucket}"] = float(served)
        for bucket, served in self.score_bucket_served.items():
            metrics[f"scheduler/router/score_bucket_served/{bucket}"] = float(served)
        for reason, cnt in self.resume_fallback_reason_count.items():
            metrics[f"scheduler/router/resume_fallback_reason/{reason}"] = float(cnt)

        if self.normal_total_requests:
            total = float(self.normal_total_requests)
            metrics["scheduler/router/normal_total_requests"] = total
            metrics["scheduler/router/normal_queue_wait_mean_s"] = self.normal_queue_wait_sum / total
        if self.resume_total_requests == 0:
            return metrics
        total = float(self.resume_total_requests)
        metrics.update({
            "scheduler/router/resume_total_requests": total,
            "scheduler/router/resume_affinity_hit_rate": self.resume_affinity_hits / total,
            "scheduler/router/resume_migration_rate": self.resume_migrations / total,
            "scheduler/router/resume_forced_migration_rate": self.resume_forced_migrations / total,
            "scheduler/router/resume_pause_age_mean_s": self.resume_pause_age_sum / total,
            "scheduler/router/resume_queue_wait_mean_s": self.resume_queue_wait_sum / total,
            "scheduler/router/resume_selected_worker_load_mean": self.resume_selected_worker_load_sum / total,
            "scheduler/router/resume_score_mean": self.resume_score_sum / total,
        })
        for bucket, served in self.resume_wait_bucket_served.items():
            metrics[f"scheduler/router/resume_bucket_served/{bucket}"] = float(served)
        self._reset_resume_metrics()
        return metrics

    def _clear_src_rank_mappings(self, src_ranks: Set[int]) -> None:
        """Clear sticky mappings to allow re-routing on retry."""
        for src_rank in src_ranks:
            self.src_rank2_dp_rank.pop(src_rank, None)

    async def rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        # Atomic operation under routing_lock
        async with self.routing_lock:
            # Rebalance (abort + update active_dp_ranks)
            return await self.rebalance_on_shrink_impl(shrink_dp_ranks)

    async def rebalance_on_shrink_impl(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        """Abort requests on shrinking workers, clear mappings for natural re-dispatch.

        Args:
            shrink_dp_ranks: DP ranks to remove from active set

        Returns:
            {"aborted": count, "remapped": count}

        Raises:
            ValueError: If shrink_dp_ranks empty/invalid/duplicates
            RuntimeError: If timeout or operation fails
        """
        # VAL: VAL_NON_EMPTY, VAL_TYPE_CHECK, VAL_INT_RANGE, VAL_NO_DUPLICATES
        if not shrink_dp_ranks:
            raise ValueError("shrink_dp_ranks cannot be empty")

        for rank in shrink_dp_ranks:
            if not isinstance(rank, int):
                raise TypeError(f"Expected int, got {type(rank)}")
            if not (0 <= rank < len(self.workers)):
                raise ValueError(f"rank {rank} out of range")

        if len(shrink_dp_ranks) != len(set(shrink_dp_ranks)):
            raise ValueError(f"Duplicates in shrink_dp_ranks")

        # P0: LOCK_TIMEOUT
        try:
            return await asyncio.wait_for(
                self._rebalance_on_shrink(shrink_dp_ranks),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("rebalance_on_shrink timed out after 30s")

    async def _rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        """Internal implementation of shrink rebalancing.

        PRE-CONDITION: routing_lock MUST be held by caller (shrink_workers).
        This method does NOT acquire the lock internally to avoid double-lock deadlock.

        Args:
            shrink_dp_ranks: DP ranks to remove from active set

        Returns:
            {"aborted": count, "remapped": count}

        Raises:
            RuntimeError: If shrink operation fails
        """
        keep_ranks = list(self.active_dp_ranks - set(shrink_dp_ranks))
        if not keep_ranks:
            raise ValueError("Cannot shrink to zero active ranks")

        old_active_ranks = self.active_dp_ranks.copy()
        self.active_dp_ranks = set(keep_ranks)

        try:
            total_aborted = 0
            abort_futures = []

            for dp_rank in shrink_dp_ranks:
                request_ids = list(self.running_requests[dp_rank])
                if not request_ids:
                    continue

                total_aborted += len(request_ids)

                abort_futures.append(
                    self.workers[dp_rank].abort_requests.remote(request_ids)
                )

            await asyncio.gather(*abort_futures)

            while True:
                remain = sum(len(self.running_requests[dp_rank]) for dp_rank in shrink_dp_ranks)
                if remain == 0:
                    break
                logger.info(f"Shrink: waiting for {len(shrink_dp_ranks)} workers {remain=} to finish abort")
                await asyncio.sleep(3)

            # Clear ALL mappings pointing to shrinking workers (not just in-flight)
            shrink_dp_ranks_set = set(shrink_dp_ranks)
            src_ranks_to_remap = set([
                src_rank for src_rank, dp_rank in self.src_rank2_dp_rank.items()
                if dp_rank in shrink_dp_ranks_set
            ])
            self._clear_src_rank_mappings(src_ranks_to_remap)

            logger.info(
                f"Shrink: aborted {total_aborted} requests, "
                f"cleared {len(src_ranks_to_remap)} mappings"
            )

            return {"aborted": total_aborted, "remapped": len(src_ranks_to_remap)}

        except Exception as e:
            self.active_dp_ranks = old_active_ranks
            raise RuntimeError(f"Shrink failed: {e}") from e

    async def rebalance_on_expand(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        # Atomic operation under routing_lock
        async with self.routing_lock:
            # Rebalance (update active_dp_ranks + conditional abort)
            return await self.rebalance_on_expand_impl(expand_dp_ranks)

    async def rebalance_on_expand_impl(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        """Add workers and rebalance via src_rank-level abort.

        Args:
            expand_dp_ranks: DP ranks to add to active set

        Returns:
            {"aborted": count, "remapped": count}

        Raises:
            ValueError: If expand_dp_ranks invalid
            RuntimeError: If timeout or operation fails
        """
        # VAL: VAL_NON_EMPTY, VAL_TYPE_CHECK, VAL_INT_RANGE, VAL_NO_DUPLICATES
        if not expand_dp_ranks:
            raise ValueError("expand_dp_ranks cannot be empty")
        for rank in expand_dp_ranks:
            if not isinstance(rank, int):
                raise TypeError(f"Expected int, got {type(rank)}")
            if not (0 <= rank < len(self.workers)):
                raise ValueError(f"rank {rank} out of range")
        if len(expand_dp_ranks) != len(set(expand_dp_ranks)):
            raise ValueError(f"Duplicates in expand_dp_ranks")

        # P0: LOCK_TIMEOUT
        try:
            return await asyncio.wait_for(
                self._rebalance_on_expand(expand_dp_ranks),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("rebalance_on_expand timed out after 30s")

    async def _rebalance_on_expand(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        """Internal implementation of expand rebalancing.

        PRE-CONDITION: routing_lock MUST be held by caller (expand_workers).
        This method does NOT acquire the lock internally to avoid double-lock deadlock.

        Algorithm: Round-robin selection across old workers
        1. Calculate proportional src_ranks to abort: src_ranks_to_keep = ceil(total * old_count / new_count)
        2. Group existing src_ranks by dp_rank (only old workers)
        3. Round-robin iterate over old workers using cycle()
        4. Select one src_rank at a time until remaining_to_abort reaches 0
        5. Abort ALL requests from selected src_ranks
        6. Clear src_rank mappings for reallocation to new workers

        Implementation Notes:
        - Uses cycle() for infinite round-robin iteration over old workers
        - Check at line 1146 (if not dp_rank in old_active_dp_ranks) is redundant
          since dp_rank_to_src_ranks already contains only old workers, but kept as defensive guard
        - Loop terminates when remaining_to_abort <= 0 or all worker lists are exhausted
        - If all workers exhausted before reaching target, loop may cycle indefinitely
          (no explicit check for empty state, but pop(0) will eventually empty all lists)

        Args:
            expand_dp_ranks: DP ranks to add to active set (already validated)

        Returns:
            {"aborted": count, "remapped": count} - count of src_ranks aborted/remapped

        Preconditions:
            - routing_lock MUST be held by caller
            - expand_dp_ranks validated (non-empty, int, in range, no duplicates)

        Postconditions:
            - active_dp_ranks updated with expand_dp_ranks
            - Selected src_ranks aborted and removed from mappings
            - Requests from aborted src_ranks reported as is_abort=True
        """
        # Calculate counts before updating active_dp_ranks
        old_dp_count = len(self.active_dp_ranks)
        old_active_dp_ranks = self.active_dp_ranks.copy()

        self.active_dp_ranks = self.active_dp_ranks | set(expand_dp_ranks)
        new_dp_count = len(self.active_dp_ranks)

        total_src_ranks = len(self.src_rank2_dp_rank)
        if total_src_ranks == 0:
            return {"aborted": 0, "remapped": 0}

        # Proportional calculation
        src_ranks_to_keep = math.ceil(int(total_src_ranks * old_dp_count / new_dp_count))
        src_ranks_to_abort = total_src_ranks - src_ranks_to_keep

        if src_ranks_to_abort <= 0:
            logger.info("Expand: no rebalancing needed (src_ranks_to_abort <= 0)")
            return {"aborted": 0, "remapped": 0}

        # Group src_ranks by dp_rank (old workers only)
        dp_rank_to_src_ranks = defaultdict(list)
        for src_rank, dp_rank in self.src_rank2_dp_rank.items():
            if dp_rank in old_active_dp_ranks:
                dp_rank_to_src_ranks[dp_rank].append(src_rank)

        # Round-robin selection: iterate over old workers and select one src_rank at a time
        # todo optimization:(yangpeng) take uneven dp load into consideration and do dynamic load balancing, not just RR
        selected_src_ranks = []
        remaining_to_abort = src_ranks_to_abort
        for dp_rank in itertools.cycle(dp_rank_to_src_ranks.keys()):
            if not dp_rank in old_active_dp_ranks:
                continue

            if remaining_to_abort <= 0:
                break

            src_ranks_on_worker = dp_rank_to_src_ranks.get(dp_rank, [])
            if not src_ranks_on_worker:
                continue
            selected_src_ranks.append(src_ranks_on_worker.pop(0))

            remaining_to_abort -= 1

        # Remove from mapping and group by dp_rank for abort
        abort_by_dp_rank = defaultdict(list)
        for src_rank in selected_src_ranks:
            dp_rank = self.src_rank2_dp_rank.pop(src_rank)

            # Find request_id(s) for this src_rank
            for request_id, sr in self.request_id_2_src_rank.items():
                if sr == src_rank:
                    abort_by_dp_rank[dp_rank].append(request_id)

        # Send batched ABORT commands
        abort_futures = []
        total_aborted = 0
        for dp_rank, request_ids in abort_by_dp_rank.items():
            if not request_ids:
                continue

            total_aborted += len(request_ids)
            abort_futures.append(
                self.workers[dp_rank].abort_requests.remote(request_ids)
            )


        await asyncio.gather(*abort_futures)

        logger.info(
            f"Expand: aborted {len(selected_src_ranks)} src_ranks, "
            f"cleared {len(selected_src_ranks)} mappings "
            f"(proportional: {old_dp_count}/{new_dp_count})"
        )

        return {"aborted": len(selected_src_ranks), "remapped": len(selected_src_ranks)}
