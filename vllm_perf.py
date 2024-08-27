from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs, AsyncEngineArgs
from vllm.engine.async_llm_engine import LLMEngine, AsyncLLMEngine
from vllm.lora.request import LoRARequest
from vllm.utils import random_uuid
from timeit import default_timer as timer

def ttft_measurer(prompt, args):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
    )
    tokenizer = llm.get_tokenizer()
    def single_request():
        sampling_params = SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=1,
            )
        prompt_token_ids = tokenizer.encode(prompt)
        llm._add_request(
                prompt=None,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                )
        start = timer()
        llm._run_engine(use_tqdm=False)
        return timer() - start
    return single_request

def tpot_measurer(prompt, args):
    engineArgs = AsyncEngineArgs(args.model)
    engineArgs.trust_remote_code = True
    engineArgs.dtype = args.dtype
    engineArgs.disable_log_stats = True
    engineArgs.disable_log_requests = True
    llm = AsyncLLMEngine.from_engine_args(engineArgs)

    async def single_request():
        sampling_params = SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=args.output_tokens,
            )
        request_id = random_uuid()
        results_generator = llm.generate(prompt, sampling_params, request_id)
        i = 0
        async for _ in results_generator:
            if i == 0:
                start = timer()
            i += 1
        return (timer() - start) / (i - 1)
    return single_request

def static_batch_measurer(prompt, args):
    llm = init_llm(args)

    def single_request():
        sampling_params = SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=args.output_tokens,
            )
        for i in range(args.batch_size):
            lora_request = None
            if args.lora_path:
                lora_id = i % args.num_loras
                lora_request = LoRARequest(f"llmperf-lora-{lora_id}", lora_id, args.lora_path)
            llm.add_request(str(i),
                            prompt,
                            sampling_params,
                            lora_request=lora_request,
            )
        done = 0
        start = timer()
        while llm.has_unfinished_requests():
            request_outputs = llm.step()
            for request_output in request_outputs:
                if request_output.finished:
                    done += 1
        assert done == args.batch_size
        total_time = timer() - start
        tokens_count = args.batch_size * args.output_tokens
        return tokens_count / total_time
    return single_request

def rate_throughput_measurer(prompt, args):
    llm = init_async_llm(args)

    async def single_request(req_num):
        sampling_params = SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=args.output_tokens,
            )
        lora_request = None
        if args.lora_path:
            lora_id = req_num % args.num_loras
            lora_request = LoRARequest(f"llmperf-lora-{lora_id}", lora_id, args.lora_path)
        request_id = random_uuid()
        results_generator = llm.generate(prompt, sampling_params, request_id, lora_request=lora_request)
        async for _ in results_generator:
            pass
        return args.output_tokens
    return single_request

def sample_rate_throughput_measurer(args):
    llm = init_async_llm(args)
    async def single_request(sample):
        sampling_params = SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=sample["output_len"],
            )
        request_id = random_uuid()
        results_generator = llm.generate(sample["prompt"], sampling_params, request_id)
        async for _ in results_generator:
            pass
        return sample["output_len"]
    return single_request

def sample_output_rate_throughput_measurer(args):
    llm = init_async_llm(args)
    async def single_request(sample):
        sampling_params = SamplingParams(
                top_k=args.top_k,
                temperature=args.temperature,
                max_tokens=4096,
            )
        request_id = random_uuid()
        results_generator = llm.generate(sample["prompt"], sampling_params, request_id)
        i = 0
        async for _ in results_generator:
            i += 1
        return i
    return single_request

def init_async_llm(args):
    engineArgs = AsyncEngineArgs(args.model)
    engineArgs.trust_remote_code = True
    engineArgs.dtype = args.dtype
    engineArgs.max_num_seqs = args.batch_size
    engineArgs.gpu_memory_utilization = args.gpu_memory_utilization
    engineArgs.disable_log_stats = True
    engineArgs.disable_log_requests = True
    engineArgs.enable_lora = args.lora_path is not None
    engineArgs.max_loras = args.num_loras
    return AsyncLLMEngine.from_engine_args(engineArgs)

def init_llm(args):
    engineArgs = EngineArgs(args.model)
    engineArgs.trust_remote_code = True
    engineArgs.dtype = args.dtype
    engineArgs.max_num_seqs = args.batch_size
    engineArgs.disable_log_stats = True
    engineArgs.disable_log_requests = True
    engineArgs.enable_lora = args.lora_path is not None
    engineArgs.max_loras = args.num_loras
    return LLMEngine.from_engine_args(engineArgs)
