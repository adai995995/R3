from typing import List, Dict, Any, Optional

import time

import torch
from torch.nn.utils.rnn import pad_sequence

from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, register_llm_proxy
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.router import RouterManager, RouterClient, is_report_data_finished
from roll.utils.functionals import (
    postprocess_generate,
    concatenate_input_and_output,
)

# RouterClient._postprocess_generate copies these from worker response; must not
# be dropped when rebuilding lm_output from the original request meta_info.
_ROUTER_RESPONSE_META_KEYS = (
    "selected_backend_id",
    "selected_worker_url",
    "client_submit_ts",
    "router_handle_start_ts",
    "gateway_post_start_ts",
    "gateway_response_headers_ts",
    "gateway_body_done_ts",
    "router_return_ts",
    "direct_worker_data_path",
    "policy_route_submit_ts",
    "policy_route_submit_done_ts",
    "policy_route_return_ts",
    "policy_slim_route_request",
    "resume_dispatch_value",
    "resume_dispatch_expected_saved_tokens",
    "resume_dispatch_queue_cost_tokens",
    "resume_dispatch_memory_pressure_cost_tokens",
    "resume_dispatch_inflight",
    "resume_dispatch_inflight_ratio",
    "resume_dispatch_memory_pressure",
    "resume_dispatch_history_len_for_value",
    "resume_dispatch_matched_tokens_for_value",
    "resume_dispatch_p_hit_for_value",
    "resume_dispatch_value_source_matched",
    "resume_dispatch_value_source_p_hit",
    "resume_dispatch_value_source_prior",
    "resume_dispatch_value_min",
    "resume_admission_admitted",
    "route_model_version",
    "kv_lease_model_version",
    "kv_lease_model_version_match",
    "kv_lease_stale_version_blocked",
    "kv_hit_same_version",
    "kv_hit_stale_version_blocked",
    "engine_kv_pinned_tokens",
    "engine_kv_evicted_tokens",
    "engine_kv_evicted_pinned_tokens",
    "engine_kv_lease_hit",
    "engine_kv_lease_miss",
    "engine_kv_lease_stale_version_blocked",
    "kv_lease_state_code",
    "kv_lease_state_created",
    "kv_lease_state_active",
    "kv_lease_state_renewed",
    "kv_lease_state_expired",
    "kv_lease_state_released",
    "kv_lease_state_evicted",
    "kv_lease_version",
    "kv_lease_record_ttl_s",
    "kv_lease_record_score",
    "kv_lease_remaining_s",
    "kv_lease_backend_id",
    "policy_local_route_hint",
    "policy_local_route_hint_hit",
    "policy_local_route_hint_reason",
    "policy_local_route_hint_lease_remaining_s",
    "policy_local_route_hint_lease_score",
    "policy_local_route_hint_p_hit",
    "policy_local_route_hint_cache_age_s",
    "policy_local_route_hint_use_dispatch_value",
    "policy_local_route_hint_dispatch_value",
    "policy_local_route_hint_expected_saved_tokens",
    "policy_local_route_hint_expected_source_matched",
    "policy_local_route_hint_expected_source_p_hit",
    "policy_local_route_hint_expected_source_prior",
    "policy_local_route_hint_history_len_for_value",
    "policy_local_route_hint_matched_tokens_for_value",
    "policy_local_route_hint_p_hit_for_value",
    "policy_local_route_hint_default_p_hit",
    "policy_local_route_hint_queue_cost_tokens",
    "policy_local_route_hint_memory_pressure_cost_tokens",
    "policy_local_route_hint_inflight",
    "policy_local_route_hint_inflight_ratio",
    "policy_local_route_hint_memory_pressure",
    "policy_worker_submit_ts",
    "policy_worker_submit_done_ts",
    "policy_worker_return_ts",
    "policy_observe_submit_ts",
    "policy_observe_submit_done_ts",
    "policy_observe_return_ts",
    "policy_observe_async",
    "observe_in_critical_path",
    "observe_pending_count",
    "observe_drain_count",
    "router_route_decision_done_ts",
    "router_route_return_ts",
    "router_slim_route_request",
    "router_fast_route_path",
    "router_fast_route_reason",
    "router_observe_recv_ts",
    "engine_start_ts",
    "engine_first_token_ts",
    "engine_finish_ts",
    "worker_generator_done_ts",
    "worker_postprocess_done_ts",
    "worker_log_done_ts",
    "worker_log_skipped",
    "router_worker_response_ts",
    "router_observe_done_ts",
    "policy_ray_submit_done_ts",
    "resume_fast_path",
    "resume_fast_path_reason",
    "resume_enqueue_ts",
    "resume_dispatch_ts",
    "resume_queue_wait_s",
    "context_class_gpu_hit",
    "context_class_cpu_reload",
    "context_class_full_prefill",
    "selected_backend_affinity_hit",
    "selected_backend_migration",
    "worker_load_skew_at_dispatch",
    "selected_worker_load_at_dispatch",
    "routing_policy",
    "remaining_steps",
    "max_steps",
    "remaining_steps_ratio",
    "trajectory_value",
    "order_score",
    "dispatch_score",
    "system_dispatch_score",
    "system_delay_regret",
    "expected_prefill_saved",
    "belief_level",
    "belief_p_hit",
    "resume_lease_ttl_s",
    "resume_lease_score",
    "kv_bytes_proxy",
    "memory_pressure",
    "pending_resume_lease_ttl_s",
    "pending_resume_lease_score",
    "belief_estimated_hit_tokens",
    "belief_estimated_prefill_tokens",
    "lookup_resume_found",
    "lookup_hit_tokens",
    "lookup_cache_confidence",
    "lookup_estimated_prefill_tokens",
    "lookup_lease_remaining_s",
    "ttl_remaining_s",
    "actual_hit",
    "matched_prefix_tokens",
    "resume_prefill_tokens",
    "estimated_prefill_tokens",
    "prefill_time_ms",
    "cache_confidence",
    "context_class",
    "prefill_ratio",
    "engine_cache_confidence",
    "p_hit_measured",
    "p_hit_effective",
    "p_hit_belief",
    "saved_prefill_tokens",
    "saved_prefill_ms",
    "saved_prefill_ms_per_gb_second",
    "pinned_kv_gb_seconds",
    "avoidable_reprefill_tokens",
    "dead_pinned_kv_gb_seconds",
    "hot_resume_miss_ratio",
    "locality_mismatch_count",
    "queue_decay_loss_ms",
    "queue_decay_loss_proxy",
    "kv_lease_effective_ttl_s",
)


@register_llm_proxy("policy")
class PolicyProxy(BaseLLMProxy):
    """
    A proxy for policy model that invokes the policy model's engine (e.g. vllm/sglang) to perform generation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_client: RouterClient = RouterManager.create_client_sync(self.generate_scheduler)

    def generate(self,
                 messages: List[Dict[str, str]],
                 lm_input: DataProto,
                 generation_config: Dict[str, Any]) -> DataProto:

        lm_input.meta_info["generation_config"] = generation_config
        lm_input.meta_info["pad_to_seq_len"] = False
        src_rank = lm_input.meta_info.pop("src_rank")
        client_submit_ts = time.time()
        response_data: Optional[DataProto] = self.router_client.generate_request_sync(req=lm_input, request_id=None, uid=src_rank)
        router_return_ts = time.time()

        if response_data is None or not is_report_data_finished(response_data):
            return None

        # postprocess_generate, input_ids, attention_mask, left pad
        eos_token_id = response_data.meta_info["eos_token_id"]
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0]
        pad_token_id = response_data.meta_info["pad_token_id"]
        output_token_ids = response_data.meta_info["output_token_ids"]
        if not output_token_ids or any(len(ids) == 0 for ids in output_token_ids):
            return None
        output_tokens = [torch.tensor(token_ids) for token_ids in output_token_ids]

        output_logprobs = response_data.meta_info.get("output_logprobs", None)

        output_tensor = pad_sequence(output_tokens, batch_first=True, padding_value=pad_token_id)
        output_tensor = concatenate_input_and_output(
            input_ids=lm_input.batch["input_ids"], output_ids=output_tensor, num_return_sequences=len(output_tokens)
        )
        lm_output: DataProto = postprocess_generate(
            prompts=lm_input,
            output=output_tensor,
            num_return_sequences=len(output_tokens),
            sequence_length=output_tensor.shape[-1],
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            pad_to_seq_len=lm_input.meta_info.get("pad_to_seq_len", True),
            output_logprobs=output_logprobs,
        )
        request_repeat = lm_input.repeat(repeat_times=len(output_tokens))
        lm_output.non_tensor_batch = request_repeat.non_tensor_batch
        lm_output.meta_info = dict(request_repeat.meta_info)
        lm_output.meta_info["client_submit_ts"] = client_submit_ts
        lm_output.meta_info.setdefault("router_return_ts", router_return_ts)
        for key in _ROUTER_RESPONSE_META_KEYS:
            if key in response_data.meta_info:
                lm_output.meta_info[key] = response_data.meta_info[key]
        lm_output.meta_info.pop("generation_config", None)
        return lm_output
