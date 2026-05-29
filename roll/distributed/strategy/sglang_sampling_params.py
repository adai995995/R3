"""Lightweight sglang sampling-params helper for router / env workers.

Must not import sglang_strategy or roll.third_party.sglang — EnvironmentWorker
runs on CPU and router only needs a JSON-serializable dict for HTTP payloads.
"""


def create_sampling_params_for_sglang(gen_kwargs: dict):
    return dict(
        max_new_tokens=gen_kwargs["max_new_tokens"],
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        stop_token_ids=gen_kwargs["eos_token_id"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        n=gen_kwargs["num_return_sequences"],
        stop=gen_kwargs["stop_strings"],
        no_stop_trim=gen_kwargs.get("include_stop_str_in_output", True),
    )
