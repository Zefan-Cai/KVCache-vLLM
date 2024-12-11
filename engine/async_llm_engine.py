import asyncio
import time
from functools import partial
from typing import (Any, AsyncGenerator, Callable, Dict, Iterable, List,
                    Mapping, Optional, Set, Tuple, Type, Union)

from typing_extensions import assert_never

import vllm.envs as envs
from vllm.config import (DecodingConfig, EngineConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_timeout import asyncio_timeout
from vllm.engine.llm_engine import (DecoderPromptComponents, LLMEngine,
                                    PromptComponents, SchedulerOutputState)
from vllm.engine.metrics_types import StatLoggerBase
from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.ray_utils import initialize_ray_cluster, ray
from vllm.inputs import (EncoderDecoderLLMInputs, LLMInputs, PromptInputs,
                         SingletonPromptInputs)
from vllm.inputs.parse import is_explicit_encoder_decoder_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import ExecuteModelRequest
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import print_warning_once

logger = init_logger(__name__)
ENGINE_ITERATION_TIMEOUT_S = envs.VLLM_ENGINE_ITERATION_TIMEOUT_S


class AsyncEngineDeadError(RuntimeError):
    pass


def _log_task_completion(task: asyncio.Task,
                         error_callback: Callable[[Exception], None]) -> None:
    """This function is only intended for the `engine.run_engine_loop()` task.

    In particular, that task runs a `while True` loop that can only exit if
    there is an exception.
    """

    exception = None
    try:
        return_value = task.result()
        raise AssertionError(
            f"The engine background task should never finish without an "
            f"exception. {return_value}")
    except asyncio.exceptions.CancelledError:
        # We assume that if the task is cancelled, we are gracefully shutting
        # down. This should only happen on program exit.
        logger.info("Engine is gracefully shutting down.")
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise AsyncEngineDeadError(
            "Task finished unexpectedly. This should never happen! "
            "Please open an issue on Github. See stack trace above for the "
            "actual cause.") from e


STOP_ITERATION = Exception()  # Sentinel


class AsyncStream:
    """A stream of RequestOutputs or EmbeddingRequestOutputs for a request
    that can be iterated over asynchronously via an async generator."""

    def __init__(self, request_id: str, cancel: Callable[[str], None]) -> None:
        self.request_id = request_id
        self._cancel = cancel
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput, EmbeddingRequestOutput,
                              Exception]) -> None:
        if not self._finished:
            self._queue.put_nowait(item)

    def finish(
        self,
        exception: Optional[Union[BaseException, Type[BaseException]]] = None,
    ) -> None:
        if not self._finished:
            self._finished = True
            self._queue.put_nowait(
                exception if self._is_raisable(exception) else STOP_ITERATION)

    @property
    def finished(self) -> bool:
        return self._finished

    async def generator(
        self
    ) -> AsyncGenerator[Union[RequestOutput, EmbeddingRequestOutput], None]:
        try:
            while True:
                result = await self._queue.get()
                if self._is_raisable(result):
                    if result == STOP_ITERATION:
                        return
                    raise result
                yield result
        except GeneratorExit:
            self._cancel(self.request_id)
            raise asyncio.CancelledError from None

    @staticmethod
    def _is_raisable(value: Any):
        return isinstance(value, BaseException) or \
                (isinstance(value, type) and \
                 issubclass(value, BaseException))


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._aborted_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = asyncio.Event()

    def __contains__(self, item):
        return item in self._request_streams

    def __len__(self) -> int:
        return len(self._request_streams)

    def propagate_exception(self,
                            exc: Exception,
                            request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self.abort_request(request_id, exception=exc)
        else:
            # NB: tuple() used here because self.abort_request pops the stream
            # out of self._request_streams, so we can't iterate on it directly
            for rid in tuple(self._request_streams.keys()):
                self.abort_request(rid, exception=exc)

    def process_request_output(self,
                               request_output: Union[RequestOutput,
                                                     EmbeddingRequestOutput],
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id
        finished = request_output.finished

        if finished:
            stream = self._request_streams.pop(request_id, None)
        else:
            stream = self._request_streams.get(request_id)
        # Guard against a KeyError which can occur if the request was aborted
        # while the output was generated
        if stream is not None:
            stream.put(request_output)
            if finished:
                stream.finish()

        if verbose and finished:
            logger.info("Finished request %s.", request_id)

    def process_exception(self,
                          request_id: str,
                          exception: BaseException,
                          *,
                          verbose: bool = False) -> None:
        """Propagate an exception from the engine."""
        if verbose:
            logger.info("Finished request %s.", request_id)
        self.abort_request(request_id, exception=exception)

    def add_request(self,
                    request_id: str,
                    *,
                    verbose: bool = False,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        abort_request = partial(self.abort_request, verbose=verbose)
        stream = AsyncStream(request_id, abort_request)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))

        self.new_requests_event.set()

        if verbose:
            logger.info("Added request %s.", request_id)

        return stream

    def abort_request(self,
                      request_id: str,
                      *,
                      exception: Optional[Union[BaseException,
                                                Type[BaseException]]] = None,
                      verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info("Aborted request %s.", request_id)

        self._aborted_requests.put_nowait(request_id)

        stream = self._request_streams.pop(request_id, None)
        if stream is not None:
            stream.finish(exception=exception)

    def get_new_and_aborted_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._aborted_requests.empty():
            request_id = self._aborted_requests.get_nowait()
            finished_requests.add(request_id)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            request_id = stream.request_id
            if request_id in finished_requests:
                # The request has already been aborted.
                stream.finish(asyncio.CancelledError)
                finished_requests.discard(request_id)
            else:
                self._request_streams[request_id] = stream
                new_requests.append(new_request)

        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        self.new_requests_event.clear()

    def has_new_requests(self):
        return not self._new_requests.empty()


class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def step_async(
        self, virtual_engine: int
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        # these are cached outputs from previous iterations. None if on first
        # iteration
        cached_outputs = self.cached_scheduler_outputs[virtual_engine]
        seq_group_metadata_list = cached_outputs.seq_group_metadata_list
        scheduler_outputs = cached_outputs.scheduler_outputs
        allow_async_output_proc = cached_outputs.allow_async_output_proc
        cache_moves = None

        ctx = self.scheduler_contexts[virtual_engine]

        # Clear outputs for each new scheduler iteration
        ctx.request_outputs.clear()

        if self.kvcompress_config:
            cache_moves = self.scheduler[0].schedule_kvcompress()

            if cache_moves:
                self.model_executor.execute_cache_moves(cache_moves,
                                                        self.kvcompress_state.kv_metrics)

        # skip the scheduler if there are any remaining steps in the seq groups.
        # This ensures that the scheduler is only called again when the current
        # batch has completed.
        if not self._has_remaining_steps(seq_group_metadata_list):

            # Schedule iteration
            (seq_group_metadata_list, scheduler_outputs,
             allow_async_output_proc
             ) = self.scheduler[virtual_engine].schedule()

            ctx.seq_group_metadata_list = seq_group_metadata_list
            ctx.scheduler_outputs = scheduler_outputs

            # Maybe switch from async mode to sync mode
            if not allow_async_output_proc and len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)

            if (self.scheduler_config.is_multi_step
                    and scheduler_outputs.num_lookahead_slots > 0):
                # cache the scheduler outputs for the next iteration if we have
                # lookahead slots
                self._cache_scheduler_outputs_for_multi_step(
                    virtual_engine, seq_group_metadata_list, scheduler_outputs,
                    allow_async_output_proc)

        assert seq_group_metadata_list is not None
        assert scheduler_outputs is not None

        if not scheduler_outputs.is_empty():

            finished_requests_ids = self.scheduler[
                virtual_engine].get_and_reset_finished_requests_ids()

            # Check if we have a cached last_output from the previous iteration.
            # For supporting PP this is probably the best way to pass the
            # sampled_token_ids, as a separate broadcast over all the PP stages
            # will cause one virtual engine's microbatch to block the pipeline.
            last_sampled_token_ids = \
                self._get_last_sampled_token_ids(virtual_engine)

            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                virtual_engine=virtual_engine,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                # We use ExecuteModelRequest to pass the last sampled_token_ids
                # to each of the non-last PP stages for in-place prepare_input.
                last_sampled_token_ids=last_sampled_token_ids,
                kvc_state=self.kvcompress_state)

            if allow_async_output_proc:
                execute_model_req.async_callback = self.async_callbacks[
                    virtual_engine]

            if self.kvcompress_config:
                # Temp metrics must be cleared before each forward pass to ensure correct
                # metric aggregation afterward
                self.kvcompress_state.kv_metrics.clear_temp_metrics()

            # Execute the model.
            output = await self.model_executor.execute_model_async(
                execute_model_req)

            if self.kvcompress_config:
                # Aggregate KV metrics that were collected to be used in sorting during
                # later iterations.
                self.kvcompress_state.kv_metrics.aggregate_decode()

            # we need to do this here so that last step's sampled_token_ids can
            # be passed to the next iteration for PP.
            if self.scheduler_config.is_multi_step:
                self._update_cached_scheduler_output(virtual_engine, output)
        else:
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            output = []

        # Finish the current step for all the sequence groups.
        if self.scheduler_config.is_multi_step:
            for seq_group in seq_group_metadata_list:
                seq_group.finish_step()

        # Remove finished sequences from scheduler
        if self.kvcompress_config:
            for seq_group in scheduler_outputs.scheduled_seq_groups:
                if seq_group.seq_group.is_finished():
                    seq = seq_group.seq_group.get_seqs()[0]
                    self.scheduler[virtual_engine].kvcompress_scheduler.complete_seqs([seq])

        if not self._has_remaining_steps(seq_group_metadata_list):
            # Clear the cache if we have finished all the steps
            if self.scheduler_config.is_multi_step:
                self.cached_scheduler_outputs[
                    virtual_engine] = SchedulerOutputState()

            is_async = allow_async_output_proc
            is_last_step = True
            ctx.output_queue.append(
                (output, seq_group_metadata_list, scheduler_outputs, is_async,
                 is_last_step))

            if output and allow_async_output_proc:
                assert len(
                    output
                ) == 1, "Async postprocessor expects only a single output set"
                self._advance_to_next_step(
                    output[0], seq_group_metadata_list,
                    scheduler_outputs.scheduled_seq_groups)

            if not allow_async_output_proc:
                self._process_model_outputs(ctx=ctx)

                # Log stats.
                self.do_log_stats(scheduler_outputs, output)

                # Tracing
                self.do_tracing(scheduler_outputs)

        else:
            # Multi-step case
            return ctx.request_outputs

        if not self.has_unfinished_requests():
            # Drain async postprocessor (if exists)
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            assert len(ctx.output_queue) == 0

        return ctx.request_outputs

    async def stop_remote_worker_execution_loop_async(self) -> None:
        """Stop the remote worker execution loop."""
        await self.model_executor.stop_remote_worker_execution_loop_async()

    async def _tokenize_prompt_async(
        self,
        prompt: str,
        request_id: str,
        lora_request: Optional[LoRARequest],
        **tokenization_kwargs,
    ) -> List[int]:
        """Async version of :meth:`_tokenize_prompt`."""
        tokenizer = self.get_tokenizer_group(
            missing_msg="prompts must be None if skip_tokenizer_init is True")

        return await tokenizer.encode_async(request_id=request_id,
                                            prompt=prompt,
                                            lora_request=lora_request,
                                            **tokenization_kwargs)

    async def _extract_prompt_components_async(
        self,
        inputs: SingletonPromptInputs,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
    ) -> PromptComponents:
        """Async version of :meth:`_extract_prompt_components`."""
        reference_token_ids = None
        if isinstance(inputs, str):
            prompt = inputs
            prompt_token_ids = await self._tokenize_prompt_async(
                prompt,
                request_id=request_id,
                lora_request=lora_request,
            )
            multi_modal_data = None
        elif isinstance(inputs, dict):
            if "prompt_token_ids" in inputs:
                prompt = None
                prompt_token_ids = inputs["prompt_token_ids"]
            else:
                # NOTE: This extra assignment is required to pass mypy
                prompt = parsed_prompt = inputs["prompt"]
                prompt_token_ids = await self._tokenize_prompt_async(
                    parsed_prompt,
                    request_id=request_id,
                    lora_request=lora_request,
                )

            if "reference_token_ids" in inputs:
                reference_token_ids = inputs["reference_token_ids"]
            elif "reference_completion" in inputs:
                reference_token_ids = await self._tokenize_prompt_async(
                    inputs["reference_completion"],
                    request_id=request_id,
                    lora_request=lora_request,
                    add_special_tokens=False,
                )

            multi_modal_data = inputs.get("multi_modal_data")
        else:
            assert_never(inputs)

        return prompt, prompt_token_ids, multi_modal_data, reference_token_ids

    async def _process_encoder_decoder_prompt_async(
        self,
        inputs: PromptInputs,
        request_id: str,
    ) -> EncoderDecoderLLMInputs:
        """Async version of :meth:`_process_encoder_decoder_prompt`."""
        encoder_comps: PromptComponents
        decoder_comps: DecoderPromptComponents

        if is_explicit_encoder_decoder_prompt(inputs):
            encoder_task = self._extract_prompt_components_async(
                inputs["encoder_prompt"],
                request_id=request_id,
            )

            if (decoder_input := inputs["decoder_prompt"]) is None:
                encoder_comps = await encoder_task
                decoder_comps = None, None, None
            else:
                decoder_task = self._extract_prompt_components_async(
                    decoder_input,
                    request_id=request_id,
                )

                encoder_comps, decoder_comps = await asyncio.gather(
                    encoder_task, decoder_task)
        else:
            encoder_comps = await self._extract_prompt_components_async(
                inputs,
                request_id=request_id,
            )

            decoder_comps = None, None, None

        return self._build_enc_dec_llm_inputs(encoder_comps, decoder_comps)

    async def _process_decoder_only_prompt_async(
        self,
        inputs: SingletonPromptInputs,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> LLMInputs:
        """Async version of :meth:`_process_decoder_only_prompt`."""
        prompt_comps = await self._extract_prompt_components_async(
            inputs,
            request_id=request_id,
            lora_request=lora_request,
        )

        return self._build_decoder_only_llm_inputs(
            prompt_comps,
            prompt_adapter_request=prompt_adapter_request,
        )

    async def process_model_inputs_async(
        self,
        inputs: PromptInputs,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> Union[LLMInputs, EncoderDecoderLLMInputs]:
        """Async version of :meth:`process_model_inputs`."""
        if self.is_encoder_decoder_model():
            # Encoder-decoder model requires special mapping of
            # input prompts to encoder & decoder
            model_inputs = await self._process_encoder_decoder_prompt_async(
                inputs,
                request_id=request_id,
            )
        else:
            if is_explicit_encoder_decoder_prompt(inputs):
                raise ValueError("Cannot pass encoder-decoder prompt "
                                 "to decoder-only models")

            # Decoder-only operation
            model_inputs = await self._process_decoder_only_prompt_async(
                inputs,
                request_id=request_id,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
            )

        return self.input_processor(model_inputs)

    async def add_request_async(
        self,
        request_id: str,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> None:
        """Async version of :meth:`add_request`."""
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        if arrival_time is None:
            arrival_time = time.time()

        processed_inputs = await self.process_model_inputs_async(
            inputs,
            request_id=request_id,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )

        self._add_processed_request(
            request_id=request_id,
            processed_inputs=processed_inputs,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            trace_headers=trace_headers,
        )

    async def check_health_async(self) -> None:
        if self.tokenizer:
            self.tokenizer.check_health()
        self.model_executor.check_health()


class AsyncLLMEngine:
    """An asynchronous wrapper for :class:`LLMEngine`.

    This class is used to wrap the :class:`LLMEngine` class to make it
    asynchronous. It uses asyncio to create a background loop that keeps
    processing incoming requests. The :class:`LLMEngine` is kicked by the
    generate method when there are requests in the waiting queue. The generate
    method yields the outputs from the :class:`LLMEngine` to the caller.

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args: Arguments for :class:`LLMEngine`.
        **kwargs: Arguments for :class:`LLMEngine`.
    """

    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine

    def __init__(self,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 log_requests: bool = True,
                 start_engine_loop: bool = True,
                 **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.engine = self._init_engine(*args, **kwargs)

        # This ensures quick processing of request outputs
        # so the append to asyncio queues is not delayed,
        # especially for multi-step.
        #
        # TODO: Currently, disabled for engine_use_ray, ask
        # Cody/Will/Woosuk about this case.
        self.use_process_request_outputs_callback = not self.engine_use_ray
        if self.use_process_request_outputs_callback:
            self.engine.process_request_outputs_callback = \
                self.process_request_outputs

        if self.engine_use_ray:
            print_warning_once(
                "DEPRECATED. `--engine-use-ray` is deprecated and will "
                "be removed in a future update. "
                "See https://github.com/vllm-project/vllm/issues/7045.")

            if envs.VLLM_ALLOW_ENGINE_USE_RAY:
                print_warning_once(
                    "VLLM_ALLOW_ENGINE_USE_RAY is set, force engine use Ray")
            else:
                raise ValueError("`--engine-use-ray` is deprecated. "
                                 "Set `VLLM_ALLOW_ENGINE_USE_RAY=1` to "
                                 "force use it")

        self.background_loop: Optional[asyncio.Future] = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded: Optional[asyncio.Task] = None
        self.start_engine_loop = start_engine_loop
        self._errored_with: Optional[BaseException] = None

        # Lazy initialized fields
        self._request_tracker: RequestTracker

    @classmethod
    def _get_executor_cls(
            cls, engine_config: EngineConfig) -> Type[ExecutorAsyncBase]:
        distributed_executor_backend = (
            engine_config.parallel_config.distributed_executor_backend)
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, ExecutorAsyncBase):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"ExecutorAsyncBase. Got {distributed_executor_backend}.")
            if distributed_executor_backend.uses_ray:  # type: ignore
                initialize_ray_cluster(engine_config.parallel_config)
            executor_class = distributed_executor_backend
        elif engine_config.device_config.device_type == "neuron":
            from vllm.executor.neuron_executor import NeuronExecutorAsync
            executor_class = NeuronExecutorAsync
        elif engine_config.device_config.device_type == "tpu":
            if distributed_executor_backend == "ray":
                initialize_ray_cluster(engine_config.parallel_config)
                from vllm.executor.ray_tpu_executor import RayTPUExecutorAsync
                executor_class = RayTPUExecutorAsync
            else:
                assert distributed_executor_backend is None
                from vllm.executor.tpu_executor import TPUExecutorAsync
                executor_class = TPUExecutorAsync
        elif engine_config.device_config.device_type == "cpu":
            from vllm.executor.cpu_executor import CPUExecutorAsync
            executor_class = CPUExecutorAsync
        elif engine_config.device_config.device_type == "openvino":
            assert distributed_executor_backend is None, (
                "Distributed execution is not supported with "
                "the OpenVINO backend.")
            from vllm.executor.openvino_executor import OpenVINOExecutorAsync
            executor_class = OpenVINOExecutorAsync
        elif engine_config.device_config.device_type == "xpu":
            if distributed_executor_backend is None:
                from vllm.executor.xpu_executor import XPUExecutorAsync
                executor_class = XPUExecutorAsync
            elif distributed_executor_backend == "ray":
                initialize_ray_cluster(engine_config.parallel_config)
                from vllm.executor.ray_xpu_executor import RayXPUExecutorAsync
                executor_class = RayXPUExecutorAsync
            elif distributed_executor_backend == "mp":
                initialize_ray_cluster(engine_config.parallel_config)
                from vllm.executor.multiproc_xpu_executor import (
                    MultiprocessingXPUExecutorAsync)
                executor_class = MultiprocessingXPUExecutorAsync
            else:
                raise RuntimeError(
                    "Not supported distributed execution model on XPU device.")
        elif distributed_executor_backend == "ray":
            initialize_ray_cluster(engine_config.parallel_config)
            from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync
            executor_class = RayGPUExecutorAsync
        elif distributed_executor_backend == "mp":
            from vllm.executor.multiproc_gpu_executor import (
                MultiprocessingGPUExecutorAsync)
            executor_class = MultiprocessingGPUExecutorAsync
        else:
            from vllm.executor.gpu_executor import GPUExecutorAsync
            executor_class = GPUExecutorAsync
        return executor_class

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()

        if engine_args.engine_use_ray:
            from vllm.executor import ray_utils
            ray_utils.assert_ray_available()

        executor_class = cls._get_executor_cls(engine_config)

        # Create the async LLM engine.
        engine = cls(
            executor_class.uses_ray,
            engine_args.engine_use_ray,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )
        return engine

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and self._background_loop_unshielded is not None
                and not self._background_loop_unshielded.done())

    @property
    def is_stopped(self) -> bool:
        return self.errored or (self.background_loop is not None and
                                self._background_loop_unshielded is not None
                                and self._background_loop_unshielded.done())

    @property
    def errored(self) -> bool:
        return self._errored_with is not None

    @property
    def limit_concurrency(self) -> Optional[int]:
        """Maximum number of concurrently running requests."""
        return None

    def set_errored(self, exc: Exception) -> None:
        self._errored_with = exc

    def _error_callback(self, exc: Exception) -> None:
        self.set_errored(exc)
        self._request_tracker.propagate_exception(exc)

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        if self.engine_use_ray:
            return await self.engine.get_tokenizer.remote(  # type: ignore
                lora_request)

        return await (self.engine.get_tokenizer_group().
                      get_lora_tokenizer_async(lora_request))

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.errored:
            raise AsyncEngineDeadError(
                "Background loop has errored already.") from self._errored_with
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        # Initialize the RequestTracker here so it uses the right event loop.
        self._request_tracker = RequestTracker()

        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(
            partial(_log_task_completion, error_callback=self._error_callback))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def shutdown_background_loop(self) -> None:
        """
        Shut down the background loop.

        This method needs to be called during cleanup to remove
        references to `self` and properly GC the resources held
        by the async LLM engine (e.g., the executors as well as
        their resources).
        """
        if self._background_loop_unshielded is not None:
            self._background_loop_unshielded.cancel()
            self._background_loop_unshielded = None
        self.background_loop = None

    def _init_engine(self, *args,
                     **kwargs) -> Union[_AsyncLLMEngine, "ray.ObjectRef"]:
        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        else:
            # FIXME(woosuk): This is a bit hacky. Be careful when changing the
            # order of the arguments.
            cache_config = kwargs["cache_config"]
            parallel_config = kwargs["parallel_config"]
            if (parallel_config.tensor_parallel_size == 1
                    and parallel_config.pipeline_parallel_size == 1):
                num_gpus = cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            engine_class = ray.remote(num_gpus=num_gpus)(
                self._engine_class).remote
        return engine_class(*args, **kwargs)

    async def engine_step(self, virtual_engine: int) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        new_requests, aborted_requests = (
            self._request_tracker.get_new_and_aborted_requests())

        for new_request in new_requests:
            # Add the request into the vLLM engine's waiting queue.
            # TODO: Maybe add add_request_batch to reduce Ray overhead
            try:
                if self.engine_use_ray:
                    await self.engine.add_request.remote(  # type: ignore
                        **new_request)
                else:
                    await self.engine.add_request_async(**new_request)
            except ValueError as e:
                # TODO: use a vLLM specific error for failed validation
                self._request_tracker.process_exception(
                    new_request["request_id"],
                    e,
                    verbose=self.log_requests,
                )

        if aborted_requests:
            await self._engine_abort(aborted_requests)

        if self.engine_use_ray:
            request_outputs = await self.engine.step.remote()  # type: ignore
        else:
            request_outputs = await self.engine.step_async(virtual_engine)

        # Put the outputs into the corresponding streams.
        # If used as a callback, then already invoked inside
        # LLMEngine's _process_model_outputs
        if not self.use_process_request_outputs_callback:
            all_finished = self.process_request_outputs(request_outputs)
        else:
            # For callback case, we only need to detect when all
            # requests are finished
            all_finished = all(request_output.finished
                               for request_output in request_outputs)

        return not all_finished

    def process_request_outputs(self, request_outputs) -> bool:
        # Put the outputs into the corresponding streams.
        all_finished = True
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output, verbose=self.log_requests)
            all_finished = all_finished and request_output.finished

        return all_finished

    async def _engine_abort(self, request_ids: Iterable[str]):
        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_ids)  # type: ignore
        else:
            self.engine.abort_request(request_ids)

    async def run_engine_loop(self):
        if self.engine_use_ray:
            pipeline_parallel_size = 1  # type: ignore
        else:
            pipeline_parallel_size = \
                self.engine.parallel_config.pipeline_parallel_size
        has_requests_in_progress = [False] * pipeline_parallel_size
        while True:
            if not any(has_requests_in_progress):
                logger.debug("Waiting for new requests...")
                # Stop the execute model loop in parallel workers until there
                # are more requests to process. This avoids waiting
                # indefinitely in torch.distributed ops which may otherwise
                # timeout, and unblocks the RPC thread in the workers so that
                # they can process any other queued control plane messages,
                # such as add/remove lora adapters.
                if self.engine_use_ray:
                    await (self.engine.stop_remote_worker_execution_loop.
                           remote()  # type: ignore
                           )
                else:
                    await self.engine.stop_remote_worker_execution_loop_async()
                await self._request_tracker.wait_for_new_requests()
                logger.debug("Got new requests!")
                requests_in_progress = [
                    asyncio.create_task(self.engine_step(ve))
                    for ve in range(pipeline_parallel_size)
                ]
                has_requests_in_progress = [True] * pipeline_parallel_size

            # Abort if iteration takes too long due to unrecoverable errors
            # (eg. NCCL timeouts).
            try:
                async with asyncio_timeout(ENGINE_ITERATION_TIMEOUT_S):
                    done, _ = await asyncio.wait(
                        requests_in_progress,
                        return_when=asyncio.FIRST_COMPLETED)
                    for _ in range(pipeline_parallel_size):
                        await asyncio.sleep(0)
                for task in done:
                    result = task.result()
                    virtual_engine = requests_in_progress.index(task)
                    if self.engine_use_ray:
                        has_unfinished_requests = (
                            await (self.engine.
                                   has_unfinished_requests_for_virtual_engine.
                                   remote(  # type: ignore
                                       virtual_engine)))
                    else:
                        has_unfinished_requests = (
                            self.engine.
                            has_unfinished_requests_for_virtual_engine(
                                virtual_engine))
                    if result or has_unfinished_requests:
                        requests_in_progress[virtual_engine] = (
                            asyncio.create_task(
                                self.engine_step(virtual_engine)))
                        has_requests_in_progress[virtual_engine] = True
                    else:
                        has_requests_in_progress[virtual_engine] = False
            except asyncio.TimeoutError as exc:
                logger.error(
                    "Engine iteration timed out. This should never happen!")
                self.set_errored(exc)
                raise
            await asyncio.sleep(0)

    # This method does not need to be async, but kept that way
    # for backwards compatibility.
    async def add_request(
        self,
        request_id: str,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> AsyncGenerator[Union[RequestOutput, EmbeddingRequestOutput], None]:
        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        stream = self._request_tracker.add_request(
            request_id,
            verbose=self.log_requests,
            inputs=inputs,
            params=params,
            arrival_time=arrival_time or time.time(),
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request)

        return stream.generator()

    async def generate(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            inputs: The inputs to the LLM. See
                :class:`~vllm.inputs.PromptInputs`
                for more details about the format of each input.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.
            trace_headers: OpenTelemetry trace headers.
            prompt_adapter_request: Prompt Adapter request to use
                                            for generation, if any.

        Yields:
            The output `RequestOutput` objects from the LLMEngine
            for the request.

        Details:
            - If the engine is not running, start the background loop,
              which iteratively invokes
              :meth:`~vllm.engine.async_llm_engine.AsyncLLMEngine.engine_step`
              to process the waiting requests.
            - Add the request to the engine's `RequestTracker`.
              On the next background loop, this request will be sent to
              the underlying engine.
              Also, a corresponding `AsyncStream` will be created.
            - Wait for the request outputs from `AsyncStream` and yield them.

        Example:
            >>> # Please refer to entrypoints/api_server.py for
            >>> # the complete example.
            >>>
            >>> # initialize the engine and the example input
            >>> engine = AsyncLLMEngine.from_engine_args(engine_args)
            >>> example_input = {
            >>>     "prompt": "What is LLM?",
            >>>     "stream": False, # assume the non-streaming case
            >>>     "temperature": 0.0,
            >>>     "request_id": 0,
            >>> }
            >>>
            >>> # start the generation
            >>> results_generator = engine.generate(
            >>>    example_input["prompt"],
            >>>    SamplingParams(temperature=example_input["temperature"]),
            >>>    example_input["request_id"])
            >>>
            >>> # get the results
            >>> final_output = None
            >>> async for request_output in results_generator:
            >>>     if await request.is_disconnected():
            >>>         # Abort the request if the client disconnects.
            >>>         await engine.abort(request_id)
            >>>         # Return or raise an error
            >>>         ...
            >>>     final_output = request_output
            >>>
            >>> # Process and return the final output
            >>> ...
        """
        async for output in await self.add_request(
                request_id,
                inputs,
                sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
        ):
            yield LLMEngine.validate_output(output, RequestOutput)

    async def encode(
        self,
        inputs: PromptInputs,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
    ) -> AsyncGenerator[EmbeddingRequestOutput, None]:
        """Generate outputs for a request from an embedding model.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            inputs: The inputs to the LLM. See
                :class:`~vllm.inputs.PromptInputs`
                for more details about the format of each input.
            pooling_params: The pooling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.
            trace_headers: OpenTelemetry trace headers.

        Yields:
            The output `EmbeddingRequestOutput` objects from the LLMEngine
            for the request.

        Details:
            - If the engine is not running, start the background loop,
              which iteratively invokes
              :meth:`~vllm.engine.async_llm_engine.AsyncLLMEngine.engine_step`
              to process the waiting requests.
            - Add the request to the engine's `RequestTracker`.
              On the next background loop, this request will be sent to
              the underlying engine.
              Also, a corresponding `AsyncStream` will be created.
            - Wait for the request outputs from `AsyncStream` and yield them.

        Example:
            >>> # Please refer to entrypoints/api_server.py for
            >>> # the complete example.
            >>>
            >>> # initialize the engine and the example input
            >>> engine = AsyncLLMEngine.from_engine_args(engine_args)
            >>> example_input = {
            >>>     "input": "What is LLM?",
            >>>     "request_id": 0,
            >>> }
            >>>
            >>> # start the generation
            >>> results_generator = engine.encode(
            >>>    example_input["input"],
            >>>    PoolingParams(),
            >>>    example_input["request_id"])
            >>>
            >>> # get the results
            >>> final_output = None
            >>> async for request_output in results_generator:
            >>>     if await request.is_disconnected():
            >>>         # Abort the request if the client disconnects.
            >>>         await engine.abort(request_id)
            >>>         # Return or raise an error
            >>>         ...
            >>>     final_output = request_output
            >>>
            >>> # Process and return the final output
            >>> ...
        """
        async for output in await self.add_request(
                request_id,
                inputs,
                pooling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
        ):
            yield LLMEngine.validate_output(output, EmbeddingRequestOutput)

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id,
                                            exception=asyncio.CancelledError,
                                            verbose=self.log_requests)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()  # type: ignore
        else:
            return self.engine.get_model_config()

    async def get_parallel_config(self) -> ParallelConfig:
        """Get the parallel configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_parallel_config.remote(  # type: ignore
            )
        else:
            return self.engine.get_parallel_config()

    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_decoding_config.remote(  # type: ignore
            )
        else:
            return self.engine.get_decoding_config()

    async def get_scheduler_config(self) -> SchedulerConfig:
        """Get the scheduling configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_scheduler_config.remote(  # type: ignore
            )
        else:
            return self.engine.get_scheduler_config()

    async def get_lora_config(self) -> LoRAConfig:
        """Get the lora configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_lora_config.remote(  # type: ignore
            )
        else:
            return self.engine.get_lora_config()

    async def do_log_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs] = None,
            model_output: Optional[List[SamplerOutput]] = None) -> None:
        if self.engine_use_ray:
            await self.engine.do_log_stats.remote(  # type: ignore
                scheduler_outputs, model_output)
        else:
            self.engine.do_log_stats()

    async def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        t = time.perf_counter()
        logger.debug("Starting health check...")
        if self.is_stopped:
            raise AsyncEngineDeadError("Background loop is stopped.")

        if self.engine_use_ray:
            try:
                await self.engine.check_health.remote()  # type: ignore
            except ray.exceptions.RayActorError as e:
                raise RuntimeError("Engine is dead.") from e
        else:
            await self.engine.check_health_async()
        logger.debug("Health check took %fs", time.perf_counter() - t)

    async def is_tracing_enabled(self) -> bool:
        if self.engine_use_ray:
            return await self.engine.is_tracing_enabled.remote(  # type: ignore
            )
        else:
            return self.engine.is_tracing_enabled()

    def add_logger(self, logger_name: str, logger: StatLoggerBase) -> None:
        if self.engine_use_ray:
            ray.get(
                self.engine.add_logger.remote(  # type: ignore
                    logger_name=logger_name, logger=logger))
        else:
            self.engine.add_logger(logger_name=logger_name, logger=logger)

    def remove_logger(self, logger_name: str) -> None:
        if self.engine_use_ray:
            ray.get(
                self.engine.remove_logger.remote(  # type: ignore
                    logger_name=logger_name))
        else:
            self.engine.remove_logger(logger_name=logger_name)

    async def start_profile(self) -> None:
        self.engine.model_executor._run_workers("start_profile")

    async def stop_profile(self) -> None:
        self.engine.model_executor._run_workers("stop_profile")