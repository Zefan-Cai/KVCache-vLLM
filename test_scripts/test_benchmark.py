"""Benchmark offline inference throughput."""

import multiprocessing as mp
mp.set_start_method('spawn', force=True)
print(f"Multiprocessing start method: {mp.get_start_method()}")  # 应输出 'spawn'

import argparse
import json
import random
import time
from typing import List, Optional, Tuple
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs


async def run_vllm_async(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    quantization_param_path: Optional[str],
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    distributed_executor_backend: Optional[str],
    gpu_memory_utilization: float = 0.9,
    num_scheduler_steps: int = 1,
    use_v2_block_manager: bool = False,
    download_dir: Optional[str] = None,
    load_format: str = EngineArgs.load_format,
    disable_async_output_proc: bool = False,
    max_batch_size: Optional[int] = None,
    enable_kvc: bool = False,
    kvc_rate: float = 1.0,
    protected_window_size: int = 50,
    metric_collection_buffer_size: int = 0,
    kv_head_bias_path: str = "./kv_head_bias.npz",
    kvc_interval: int = 1,
    max_kv_per_compression: int = 5_000_000,
    new_token_limit: int = -1,
    max_cache_tokens: int = -1,
    record_decoding_metrics: bool = True,
    metric_aggregation: str = "L2-sum",
    compress_once: bool = False,
    speculative_model: Optional[str] = None,
    speculative_model_quantization: Optional[str] = None,
    spec_decoding_acceptance_method: Optional[str] = None,
    typical_acceptance_sampler_posterior_threshold: Optional[float] = None,
    typical_acceptance_sampler_posterior_alpha: Optional[float] = None,
    disable_frontend_multiprocessing: bool = False,
) -> float:
    from vllm import SamplingParams
    engine_args = AsyncEngineArgs(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        pipeline_parallel_size=2,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        quantization_param_path=quantization_param_path,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        download_dir=download_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        distributed_executor_backend=distributed_executor_backend,
        load_format=load_format,
        num_scheduler_steps=num_scheduler_steps,
        use_v2_block_manager=use_v2_block_manager,
        disable_async_output_proc=disable_async_output_proc,
        worker_use_ray=False,
        engine_use_ray=False,
        disable_log_requests=True,
        max_num_seqs=max_batch_size,
        enable_kvcompress=enable_kvc,
        target_compression_rate=kvc_rate,
        protected_window_size=protected_window_size,
        metric_collection_buffer_size=metric_collection_buffer_size,
        kv_head_bias_path=kv_head_bias_path,
        compression_interval=kvc_interval,
        max_kv_per_compression=max_kv_per_compression,
        new_token_limit=new_token_limit,
        max_cache_tokens=max_cache_tokens,
        record_decoding_metrics=record_decoding_metrics,
        metric_aggregation=metric_aggregation,
    )


    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing) as llm:

        # Add the requests to the engine.
        prompts: List[str] = []
        sampling_params: List[SamplingParams] = []
        for prompt, _, output_len in requests:
            prompts.append(prompt)
            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=0.0 if use_beam_search else 1.0,
                    top_p=1.0,
                    use_beam_search=use_beam_search,
                    ignore_eos=True,
                    max_tokens=output_len,
                ))

        generators = []
        start = time.perf_counter()
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            generator = llm.generate(prompt, sp, request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            pass
        end = time.perf_counter()
        return end - start


def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    quantization_param_path: Optional[str],
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    distributed_executor_backend: Optional[str],
    gpu_memory_utilization: float = 0.9,
    num_scheduler_steps: int = 1,
    use_v2_block_manager: bool = False,
    download_dir: Optional[str] = None,
    load_format: str = EngineArgs.load_format,
    disable_async_output_proc: bool = False,
    max_batch_size: Optional[int] = None,
    enable_kvc: bool = False,
    kvc_rate: float = 1.0,
    protected_window_size: int = 50,
    metric_collection_buffer_size: int = 0,
    kv_head_bias_path: str = "./kv_head_bias.npz",
    kvc_interval: int = 1,
    max_kv_per_compression: int = 5_000_000,
    new_token_limit: int = -1,
    max_cache_tokens: int = -1,
    record_decoding_metrics: bool = True,
    metric_aggregation: str = "L2-sum",
    compress_once: bool = False,
    # speculative decoding
    speculative_model: str = None,
    speculative_model_quantization: str = None,
    spec_decoding_acceptance_method: str = 'typical_acceptane_sampler',
    typical_acceptance_sampler_posterior_threshold: float = 0.09,
    typical_acceptance_sampler_posterior_alpha: float = 0.3,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        quantization_param_path=quantization_param_path,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        download_dir=download_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        distributed_executor_backend=distributed_executor_backend,
        load_format=load_format,
        num_scheduler_steps=num_scheduler_steps,
        use_v2_block_manager=use_v2_block_manager,
        disable_async_output_proc=disable_async_output_proc,
        max_num_seqs=max_batch_size,
        enable_kvcompress=enable_kvc,
        target_compression_rate=kvc_rate,
        protected_window_size=protected_window_size,
        metric_collection_buffer_size=metric_collection_buffer_size,
        kv_head_bias_path=kv_head_bias_path,
        compression_interval=kvc_interval,
        max_kv_per_compression=max_kv_per_compression,
        new_token_limit=new_token_limit,
        max_cache_tokens=max_cache_tokens,
        record_decoding_metrics=record_decoding_metrics,
        metric_aggregation=metric_aggregation,
        # speculative decoding
        speculative_model=speculative_model,
        speculative_model_quantization=speculative_model_quantization,
        spec_decoding_acceptance_method=spec_decoding_acceptance_method,
        typical_acceptance_sampler_posterior_threshold=typical_acceptance_sampler_posterior_threshold,
        typical_acceptance_sampler_posterior_alpha=typical_acceptance_sampler_posterior_alpha,
    )


    # Add the requests to the engine.
    prompts: List[str] = []
    sampling_params: List[SamplingParams] = []
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=0.0 if use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=use_beam_search,
                ignore_eos=True,
                max_tokens=output_len,
                max_cache_tokens=max_cache_tokens,
                target_compression_rate=kvc_rate,
                protected_window_size=protected_window_size,
                compress_once=compress_once,
            ))

    start = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    # for output in outputs:
    #     assert len(output.outputs[0].token_ids) == output_len, (
    #         f"{len(output.outputs[0].token_ids)=}, {output_len=}")
    return end - start, llm.llm_engine.scheduler[0].max_decoding_batch

def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.real_text:
        requests = [(REAL_TEXT_PROMPT[:args.input_len * CHAR_PER_TOK_APPROX],
                     args.input_len, args.output_len)
                    for _ in range(args.num_prompts)]
    elif args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(args.num_prompts)]
    else:
        requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
                                   args.output_len)

    if args.compression_rate is not None:
        block_size = 16
        args.max_cache_tokens = max(128, ((int(args.input_len / args.compression_rate) - 1) // block_size * block_size))

    run_args = [
        requests, args.model, args.tokenizer, args.quantization,
        args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
        args.trust_remote_code, args.dtype, args.max_model_len,
        args.enforce_eager, args.kv_cache_dtype,
        args.quantization_param_path, args.device,
        args.enable_prefix_caching, args.enable_chunked_prefill,
        args.max_num_batched_tokens, args.distributed_executor_backend,
        args.gpu_memory_utilization, args.num_scheduler_steps,
        args.use_v2_block_manager, args.download_dir, args.load_format,
        args.disable_async_output_proc, args.max_batch_size, args.enable_kvc,
        args.kvc_rate, args.protected_window_size, args.metric_collection_buffer_size,
        args.kv_head_bias_path, args.kvc_interval, args.max_kv_per_compression,
        args.new_token_limit, args.max_cache_tokens,
        args.record_decoding_metrics, args.metric_aggregation,
        args.compress_once,
        # speculative decoding
        args.speculative_model, args.speculative_model_quantization,
        args.spec_decoding_acceptance_method,
        args.typical_acceptance_sampler_posterior_threshold,
        args.typical_acceptance_sampler_posterior_alpha,
    ]

    if args.async_engine:
        print("use async_engine")
        run_args.append(args.disable_frontend_multiprocessing)
        elapsed_time = uvloop.run(run_vllm_async(*run_args))
        max_decoding_batch = None
    else:
        print("No use async_engine")
        elapsed_time, max_decoding_batch = run_vllm(*run_args)
        print(elapsed_time)


if __name__ == "__main__":

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    print(f"Multiprocessing start method: {mp.get_start_method()}")  # 应输出 'spawn'



    import torch
    import uvloop
    from tqdm import tqdm
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                            PreTrainedTokenizerBase)

    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client_from_engine_args)
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    from vllm.utils import FlexibleArgumentParser, merge_async_iterators
    from vllm.benchmark import BENCHMARKER





    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=19000,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        '--kv-cache-dtype',
        type=str,
        choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
        'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
        'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
    parser.add_argument(
        '--quantization-param-path',
        type=str,
        default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "openvino", "tpu", "xpu"],
        help='device type for vLLM execution, supporting CUDA, OpenVINO and '
        'CPU.')
    parser.add_argument(
        "--num-scheduler-steps",
        type=int,
        default=1,
        help="Maximum number of forward steps per scheduler call.")
    parser.add_argument("--use-v2-block-manager",
                        action='store_true',
                        help="Enable block manager v2.")
    parser.add_argument(
        "--enable-prefix-caching",
        action='store_true',
        help="Enable automatic prefix caching for vLLM backend.")
    parser.add_argument("--enable-chunked-prefill",
                        action='store_true',
                        help="enable chunked prefill for vLLM backend.")
    parser.add_argument('--max-num-batched-tokens',
                        type=int,
                        default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument(
        '--distributed-executor-backend',
        choices=['ray', 'mp'],
        default=None,
        help='Backend to use for distributed serving. When more than 1 GPU '
        'is used, will be automatically set to "ray" if installed '
        'or "mp" (multiprocessing) otherwise.')
    parser.add_argument(
        '--load-format',
        type=str,
        default=EngineArgs.load_format,
        choices=[
            'auto', 'pt', 'safetensors', 'npcache', 'dummy', 'tensorizer',
            'bitsandbytes'
        ],
        help='The format of the model weights to load.\n\n'
        '* "auto" will try to load the weights in the safetensors format '
        'and fall back to the pytorch bin format if safetensors format '
        'is not available.\n'
        '* "pt" will load the weights in the pytorch bin format.\n'
        '* "safetensors" will load the weights in the safetensors format.\n'
        '* "npcache" will load the weights in pytorch format and store '
        'a numpy cache to speed up the loading.\n'
        '* "dummy" will initialize the weights with random values, '
        'which is mainly for profiling.\n'
        '* "tensorizer" will load the weights using tensorizer from '
        'CoreWeave. See the Tensorize vLLM Model script in the Examples'
        'section for more information.\n'
        '* "bitsandbytes" will load the weights using bitsandbytes '
        'quantization.\n')
    parser.add_argument(
        "--disable-async-output-proc",
        action='store_true',
        default=False,
        help="Disable async output processor for vLLM backend.")
    parser.add_argument("--async-engine",
                        action='store_true',
                        default=False,
                        help="Use vLLM async engine rather than LLM class.")
    parser.add_argument("--disable-frontend-multiprocessing",
                        action='store_true',
                        default=False,
                        help="Disable decoupled async engine frontend.")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=256,
        help='set max batch size for vLLM backend',
    )
    parser.add_argument(
        "--benchmark-input-only",
        type=int,
        help='Only use input tokens when computing tok/sec'
    )
    parser.add_argument(
        "--benchmark-output-only",
        action="store_true",
        help='Only use output tokens when computing tok/sec'
    )
    parser.add_argument(
        "--kvc-rate",
        type=float,
        default=1.0,
        help="KV cache compression rate",
    )
    parser.add_argument(
        "--enable-kvc",
        action="store_true",
        help="Enable KV cache compression",
    )
    parser.add_argument(
        "--protected-window-size",
        type=int,
        default=50,
        help="Protected window size for KV cache compression",
    )
    parser.add_argument(
        '--metric-collection-buffer-size',
        type=int,
        default=0,
        help="Buffer length for collecting KV metric",
    )
    parser.add_argument(
        "--kv-head-bias-path",
        type=str,
        default=None,
        help="Path to KV head bias for KV cache compression",
    )
    parser.add_argument(
        "--kvc-interval",
        type=int,
        default=1_000_000,
        help="Compress KV cache every n iterations",
    )
    parser.add_argument(
        "--max-kv-per-compression",
        type=int,
        default=5_000_000,
        help="Max number of KVs per compression",
    )
    parser.add_argument(
        '--new-token-limit',
        type=int,
        default=-1,
        help='Max number of tokens that can be added before compression '
        'is forced',
    )
    parser.add_argument(
        '--max-cache-tokens',
        type=int,
        default=-1,
        help='Number of tokens to compress to',
    )
    parser.add_argument(
        '--only-prefill-metrics',
        action='store_false',
        dest='record_decoding_metrics',
        help='Disable KV metric collection during decoding',
    )
    parser.add_argument(
        '--metric-aggregation',
        choices=['L1-sum', 'L1-avg', 'L2-sum', 'L2-avg'],
        default='L2-sum',
        help='Aggregation used for KV metrics',
    )
    parser.add_argument(
        '--compress-once',
        action='store_true',
        help='Whether to compress each each sequence only '
        'once after prefill',
    )
    parser.add_argument(
        '--compression-rate',
        type=float,
        default=None,
        help='Configure max_cache_tokens as a fraction of input length',
    )
    parser.add_argument(
        '--speculative-model',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--speculative-model-quantization',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--spec-decoding-acceptance-method',
        type=str,
        choices=['typical_acceptance_sampler', 'rejection_sampler'],
        default='typical_acceptance_sampler'
    )
    parser.add_argument(
        '--typical-acceptance-sampler-posterior-threshold',
        type=float,
        default=0.09,
    )
    parser.add_argument(
        '--typical-acceptance-sampler-posterior-alpha',
        type=float,
        default=0.3,  # sqrt of posterior threshold
    )
    parser.add_argument(
        '--real-text',
        action='store_true',
    )
    parser.add_argument(
        '--latency-breakdown',
        action='store_true',
    )
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.use_beam_search:
            raise ValueError("Beam search is not supported for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")
    main(args)