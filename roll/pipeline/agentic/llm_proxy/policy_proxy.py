from typing import List, Dict, Any, Optional

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
        response_data: Optional[DataProto] = self.router_client.generate_request_sync(req=lm_input, request_id=None, uid=src_rank)

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
        for key in _ROUTER_RESPONSE_META_KEYS:
            if key in response_data.meta_info:
                lm_output.meta_info[key] = response_data.meta_info[key]
        lm_output.meta_info.pop("generation_config", None)
        return lm_output
