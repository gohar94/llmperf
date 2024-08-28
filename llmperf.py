import argparse
import vllm_perf
import asyncio
import math
import json
import os
from timeit import default_timer as timer

def read_prompt_from_file(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

def run_test_n_times(test, n):
    total = 0
    values = []
    for i in range(n):
        value = test()
        total += value
        print(f"Iteration {i}: {value}")
        values.append(value)
    print(f"Average: {total/n}")
    return values

def run_multi_test_n_times(test, n):
    total_1 = 0
    total_2 = 0
    values_1 = []
    values_2 = []
    for i in range(n):
        value = test()
        value_1, value_2 = value
        total_1 += value_1
        total_2 += value_2
        print(f"Iteration {i}: {value_1} {value_2}")
        values_1.append(value_1)
        values_2.append(value_2)
    print(f"Average: {total_1/n} {total_2/n}")
    return values_1, values_2

async def async_run_test_n_times(test, n):
    total = 0
    values = []
    for i in range(n):
        value = await test()
        total += value
        print(f"Iteration {i}: {value}")
        values.append(value)
    print(f"Average: {total/n}")
    return values

async def send_request_periodically(request, qps, t, total):
    tasks = []
    start = timer()
    req_num = 0
    for _ in range(math.floor(total/qps)):
        for _ in range(qps):
            req_num += 1
            task = asyncio.create_task(request(req_num))
            tasks.append(task)
        await asyncio.sleep(t)
    results = await asyncio.gather(*tasks)
    total_tokens = sum(results)
    elapsed = timer() - start
    return total_tokens / elapsed

async def send_sampled_request_periodically(request, samples, qps, t, total):
    tasks = []
    start = timer()
    i = 0
    for _ in range(math.floor(total/qps)):
        for _ in range(qps):
            task = asyncio.create_task(request(samples[i]))
            tasks.append(task)
            i += 1
        await asyncio.sleep(t)
    results = await asyncio.gather(*tasks)
    total_tokens = sum(results)
    elapsed = timer() - start
    return total_tokens / elapsed

def run_ttft(args):
    prompt = read_prompt_from_file(args.prompt_file)
    measurer = None
    if args.engine == "vllm":
        measurer = vllm_perf.ttft_measurer(prompt, args)
    else:
        print(f"TTFT test not implemented for {args.engine}")
        return
    traces = run_test_n_times(measurer, args.iterations)
    write_traces("ttft_s", traces, args, args.output_file)

def run_tpot(args):
    prompt = read_prompt_from_file(args.prompt_file)
    measurer = None
    if args.engine == "vllm":
        measurer = vllm_perf.tpot_measurer(prompt, args)
    else:
        print(f"TPOT test not implemented for {args.engine}")
        return
    traces = run_test_n_times(measurer, args.iterations)
    write_traces("tpot_s", traces, args, args.output_file)

def run_tpot_ttft(args):
    prompt = read_prompt_from_file(args.prompt_file)
    measurer = None
    if args.engine == "vllm":
        measurer = vllm_perf.tpot_ttft_measurer(prompt, args)
    else:
        print(f"TPOT_TTFT test not implemented for {args.engine}")
        return
    traces = run_multi_test_n_times(measurer, args.iterations)
    tpot_traces, ttft_traces = traces
    write_traces("tpot_s", tpot_traces, args, args.tpot_output_file)
    write_traces("ttft_s", ttft_traces, args, args.ttft_output_file)

def run_static_batch(args):
    prompt = read_prompt_from_file(args.prompt_file)
    measurer = None
    if args.engine == "vllm":
        measurer = vllm_perf.static_batch_measurer(prompt, args)
    else:
        print(f"Static batch test not implemented for {args.engine}")
        return
    traces = run_test_n_times(measurer, args.iterations)
    write_traces("throughput_tps", traces, args, args.output_file)

def run_rate_throughput(args):
    prompt = read_prompt_from_file(args.prompt_file)
    measurer = None
    if args.engine == "vllm":
        measurer = vllm_perf.rate_throughput_measurer(prompt, args)
    else:
        print(f"Rate throughput test not implemented for {args.engine}")
        return
    
    async def wrapper():
        return await send_request_periodically(measurer, args.qps, args.t, args.total_requests)
    asyncio.run(async_run_test_n_times(wrapper, args.iterations))

def run_rate_sampled_throughput(args):
    with open(args.dataset, 'r') as file:
        samples = json.load(file)
    measurer = None
    if args.engine == "vllm":
        measurer = vllm_perf.sample_rate_throughput_measurer(args)
    else:
        print(f"Rate sampled throughput test not implemented for {args.engine}")
        return
    
    async def wrapper():
        return await send_sampled_request_periodically(measurer, samples, args.qps, args.t, args.total_requests)
    asyncio.run(async_run_test_n_times(wrapper, args.iterations))

def run_rate_sampled_output_throughput(args):
    with open(args.dataset, 'r') as file:
        samples = json.load(file)
    measurer = None
    if args.engine == "vllm":
        measurer = vllm_perf.sample_output_rate_throughput_measurer(args)
    else:
        print(f"Rate sampled throughput test not implemented for {args.engine}")
        return
    
    async def wrapper():
        return await send_sampled_request_periodically(measurer, samples, args.qps, args.t, args.total_requests)
    asyncio.run(async_run_test_n_times(wrapper, args.iterations))

def write_traces(trace_name, trace_values, args, output_file):
    if not output_file:
        return
    header = [
            "model",
            "test",
            "lora",
            "num_loras",
            "batch_size",
            f"{trace_name}",
    ]
    header = f"{','.join(header)}\n"
    is_write_header = not os.path.exists(output_file)
    base = [
            args.model,
            args.test,
            args.lora_path,
            args.num_loras,
            args.batch_size,
    ]
    base = [str(x) for x in base]
    base = f"{','.join(base)}"
    with open(output_file, 'a') as f:
        if is_write_header:
            f.write(header)
        for trace in trace_values:
            line = f"{base},{trace}\n"
            f.write(line)

def add_engines_parser(base_parser):
    engine_parser = base_parser.add_subparsers(title="Engine", dest="engine", required=True)
    vllm_parser = engine_parser.add_parser("vllm", help="vLLM Engine")
    vllm_parser.add_argument("--model", type=str, default="", help="The model.")
    vllm_parser.add_argument("--lora_path", type=str, default=None, help="The LoRA adapter path.")
    vllm_parser.add_argument("--num_loras", type=int, default=0, help="Number of unique LoRA adapters.")
    vllm_parser.add_argument("--dtype", type=str, default="float16", help="The dtype.")
    vllm_parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU Memory fraction")
    vllm_parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMPerf tools to measure LLM performance")

    test_parser = parser.add_subparsers(title="Test", dest="test", required=True)
    
    ttft_parser = test_parser.add_parser("ttft", help="Measure Time To First Token (TTFT)")
    ttft_parser.add_argument("--output_file", type=str, default=None, help="Path to a file to write traces into.")
    ttft_parser.add_argument("--prompt_file", type=str, help="Path to a file containing the prompt.")
    ttft_parser.add_argument("--iterations", type=int, default=10, help="The iterations parameter.")
    ttft_parser.add_argument("--output_tokens", type=int, default=128, help="Number of tokens to retrieve")
    add_engines_parser(ttft_parser)

    tpot_parser = test_parser.add_parser("tpot", help="Measure Time Per Output Token (TPOT)")
    tpot_parser.add_argument("--output_file", type=str, default=None, help="Path to a file to write traces into.")
    tpot_parser.add_argument("--prompt_file", type=str, help="Path to a file containing the prompt.")
    tpot_parser.add_argument("--iterations", type=int, default=10, help="The iterations parameter.")
    tpot_parser.add_argument("--output_tokens", type=int, default=128, help="Number of tokens to retrieve")
    add_engines_parser(tpot_parser)

    tpot_ttft_parser = test_parser.add_parser("tpot_ttft", help="Measure Time Per Output Token (TPOT) and Time To First Token (TTFT)")
    tpot_ttft_parser.add_argument("--tpot_output_file", type=str, default=None, help="Path to a file to write TPOT traces into.")
    tpot_ttft_parser.add_argument("--ttft_output_file", type=str, default=None, help="Path to a file to write TTFT traces into.")
    tpot_ttft_parser.add_argument("--prompt_file", type=str, help="Path to a file containing the prompt.")
    tpot_ttft_parser.add_argument("--iterations", type=int, default=10, help="The iterations parameter.")
    tpot_ttft_parser.add_argument("--output_tokens", type=int, default=128, help="Number of tokens to retrieve")
    add_engines_parser(tpot_ttft_parser)

    stb_parser = test_parser.add_parser("static_batch_throughput", help="Measure throughput for static batch")
    stb_parser.add_argument("--output_file", type=str, default=None, help="Path to a file to write traces into.")
    stb_parser.add_argument("--prompt_file", type=str, help="Path to a file containing the prompt.")
    stb_parser.add_argument("--iterations", type=int, default=10, help="The iterations parameter.")
    stb_parser.add_argument("--output_tokens", type=int, default=128, help="Number of tokens to retrieve")
    add_engines_parser(stb_parser)

    rth_parser = test_parser.add_parser("rate_throughput", help="Measure throughput with sending requests at constant rate")
    rth_parser.add_argument("--prompt_file", type=str, help="Path to a file containing the prompt.")
    rth_parser.add_argument("--iterations", type=int, default=1, help="The iterations parameter.")
    rth_parser.add_argument("--output_tokens", type=int, default=128, help="Number of tokens to retrieve")
    rth_parser.add_argument("--qps", type=int, default=4, help="Number of queries to send per second")
    rth_parser.add_argument("--t", type=int, default=1, help="Time frame to send the QPS amount requests")
    rth_parser.add_argument("--total_requests", type=int, default=5000, help="Number of requests to send in total")
    add_engines_parser(rth_parser)

    rst_parser = test_parser.add_parser("rate_sampled_throughput", help="Measure throughput with sending requests at constant rate")
    rst_parser.add_argument("--dataset", type=str, help="Path to a file containing the dataset.")
    rst_parser.add_argument("--iterations", type=int, default=1, help="The iterations parameter.")
    rst_parser.add_argument("--qps", type=int, default=4, help="Number of queries to send per second (Per t)")
    rst_parser.add_argument("--t", type=int, default=1, help="Time frame to send the QPS amount requests")
    rst_parser.add_argument("--total_requests", type=int, default=5000, help="Number of requests to send in total")
    add_engines_parser(rst_parser)

    rsot_parser = test_parser.add_parser("rate_sampled_output_throughput", help="Measure throughput with sending requests at constant rate")
    rsot_parser.add_argument("--dataset", type=str, help="Path to a file containing the dataset.")
    rsot_parser.add_argument("--iterations", type=int, default=1, help="The iterations parameter.")
    rsot_parser.add_argument("--qps", type=int, default=4, help="Number of queries to send per second (Per t)")
    rsot_parser.add_argument("--t", type=int, default=1, help="Time frame to send the QPS amount requests")
    rsot_parser.add_argument("--total_requests", type=int, default=5000, help="Number of requests to send in total")
    rsot_parser.add_argument("--temperature", type=float, default=1, help="Temperature in sampling phase")
    rsot_parser.add_argument("--top_k", type=int, default=15, help="Tok K in sampling phase")
    add_engines_parser(rsot_parser)
    
    args = parser.parse_args()
    print(args)


    if args.test == "ttft":
        run_ttft(args)
    elif args.test == "tpot":
        run_tpot(args)
    elif args.test == "tpot_ttft":
        run_tpot_ttft(args)
    elif args.test == "static_batch_throughput":
        run_static_batch(args)
    elif args.test == "rate_throughput":
        run_rate_throughput(args)
    elif args.test == "rate_sampled_throughput":
        run_rate_sampled_throughput(args)
    elif args.test == "rate_sampled_output_throughput":
        run_rate_sampled_output_throughput(args)
