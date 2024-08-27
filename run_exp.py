import os
import subprocess
from datetime import datetime

python = "/home/girfan/miniconda3/envs/.gpumem/bin/python"

# test = "static_batch_throughput"
test = "tpot"

model = "meta-llama/Llama-2-7b-hf"
lora = "yard1/llama-2-7b-sql-lora-test"
output_filename = "llmperf_trace"
n_iterations = 3
n_input_tokens = 2048
n_output_tokens = 256

n_loras_ = [0, 1, 2, 4, 8, 16, 32, 64, 128]
batch_size_ = [1, 2, 4, 8, 16, 32, 64, 128]

def generate_filename_timestamp():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp

output_file = f"{output_filename}_{generate_filename_timestamp()}.csv"
print("Storing traces in:", output_file)
if os.path.exists(output_file):
    os.remove(output_file)

def exec(cmd_line):
    print(cmd_line)
    try:
        result = subprocess.run(
                cmd_line,
                shell=True,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
        )
        print("Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Command failed with error:\n", e.stderr)

for n_loras in n_loras_:
    for batch_size in batch_size_:
        cmd_line = [
                f"{python}", "llmperf.py",
                f"{test}",
                "--prompt_file", f"input_examples/llama2/{n_input_tokens}_tokens",
                "--output_file", f"{output_file}",
                "--iterations", f"{n_iterations}",
                "--output_tokens", f"{n_output_tokens}",
                "vllm",
                "--batch_size", f"{batch_size}",
                "--model", f"{model}",
                "--dtype", "float16",
        ]
        if n_loras > 0:
            cmd_line.extend([
                    "--lora_path", f"{lora}",
                    "--num_loras", f"{n_loras}",
            ])
        cmd_line = ' '.join(cmd_line)
        exec(cmd_line)

print("Stored traces in:", output_file)
