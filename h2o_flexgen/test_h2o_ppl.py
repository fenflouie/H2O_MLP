import argparse
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
import os
import datetime

# === 1. Import ===
# Append local h2o_flexgen path to use the newly cloned H2O version
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from flexgen.opt_config import get_opt_config
from flexgen.flex_opt import OptLM, Policy
from flexgen.utils import ExecutionEnv
from flexgen.compression import CompressionConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hh-ratio", type=float, default=1.0, help="Heavy hitter ratio (e.g., 1.0 for no h2o)")
    parser.add_argument("--hh-all", action="store_true", help="Apply heavy hitter to all layers")
    parser.add_argument("--percent", nargs="+", type=int,
                        default=[100, 0, 100, 0, 100, 0],
                        help="Six ints separated by space. "
                             "w_gpu w_cpu cache_gpu cache_cpu act_gpu act_cpu")
    parser.add_argument("--top-k", type=int, default=8192, help="Number of neurons to keep active in MLP. Default 8192 for dense.")
    args = parser.parse_args()
    
    # 傳遞到 backend 的環境變數裡，方便做全域讀取而不需要改動整套 Policy config
    os.environ["MLP_TOP_K"] = str(args.top_k)

    # ======================================================================================
    #                                   Configuration Section
    # ======================================================================================
    # 1. Model Configuration
    model_name = "facebook/opt-1.3b" 
    
    # 2. Batch Size
    gpu_batch_size = 4
    num_gpu_batches = 1
    
    # 3. Offload Configuration 
    percent = tuple(args.percent) 
    
    # 4. Compression Configuration
    compress_weight = False
    compress_cache = False
    
    # Compression Settings
    weight_comp_config = CompressionConfig(num_bits=4, group_size=128, group_dim=0, symmetric=False)
    cache_comp_config = CompressionConfig(num_bits=4, group_size=128, group_dim=0, symmetric=False)
    
    # 5. Other Settings
    target_device = "cuda:0"
    seq_len = 512 
    gen_len = 32 # For generation benchmark
    path_to_weights = "/home/louief/opt_weights"
    # ======================================================================================
    
    # Construct log filename
    model_size = model_name.split('-')[-1]
    str_offload = f"{percent[0]}-{percent[1]}-{percent[2]}-{percent[3]}-{percent[4]}-{percent[5]}"
    str_compress = "compW" if compress_weight else "noCompW"
    str_compress += "_compC" if compress_cache else "_noCompC"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"ppl-h2o-{args.hh_ratio}-top{args.top_k}-{model_size}-bs{gpu_batch_size}-{str_offload}-{str_compress}-{timestamp}.log"

    print("="*50)
    print(f"Running H2O FlexGen Perplexity Evaluation")
    print(f"Model: {model_name}")
    print(f"Batch Size: {gpu_batch_size}")
    print(f"HH Ratio: {args.hh_ratio}")
    print(f"HH All: {args.hh_all}")
    print(f"Top-K: {args.top_k}")
    print(f"Log File: {os.path.abspath(log_filename)}")
    print("="*50)

    print("Initializing FlexGen environment...")
    env = ExecutionEnv.create(target_device)
    
    # === Policy Setup ===
    policy = Policy(gpu_batch_size, num_gpu_batches,
                    percent[0], percent[1],
                    percent[2], percent[3],
                    percent[4], percent[5],
                    overlap=True, sep_layer=True, pin_weight=True,
                    cpu_cache_compute=False, attn_sparsity=1.0,
                    compress_weight=compress_weight,
                    comp_weight_config=weight_comp_config,
                    compress_cache=compress_cache,
                    comp_cache_config=cache_comp_config,
                    hh_ratio=args.hh_ratio, hh_all=args.hh_all)

    # === Load Model ===
    print(f"Loading {model_name}...")
    opt_config = get_opt_config(model_name)
    try:
        model = OptLM(opt_config, env, path_to_weights, policy)
    except TypeError:
        model = OptLM(opt_config, env, None, policy)
        
    print("Model loaded successfully!")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Helper for dummy inputs
    def get_dummy_inputs(prompt_len, num_prompts, tokenizer):
        prompts = ["Paris is the capital city of"] * num_prompts
        input_ids = tokenizer(prompts, padding="max_length",
                            max_length=prompt_len).input_ids
        return (input_ids,)

    # === 1. Performance Benchmark (Generation) ===
    print("\n" + "="*20 + " Running Generation Benchmark " + "="*20)
    print("Warmup...")
    warmup_inputs = get_dummy_inputs(32, gpu_batch_size, tokenizer)[0]
    model.generate(warmup_inputs, max_new_tokens=1, verbose=0)
    
    print(f"Benchmarking (Prompt: {seq_len}, Gen: {gen_len})...")
    benchmark_inputs = get_dummy_inputs(seq_len, gpu_batch_size, tokenizer)[0]
    
    from flexgen.timer import timers
    from flexgen.utils import GB
    
    timers("generate").reset()
    model.generate(benchmark_inputs, max_new_tokens=gen_len, verbose=0)
    costs = timers("generate").costs
    
    prefill_latency = costs[0]
    prefill_throughput = gpu_batch_size * seq_len / prefill_latency
    decode_latency = sum(costs[1:])
    decode_throughput = gpu_batch_size * (gen_len - 1) / max(decode_latency, 1e-10)
    total_latency = prefill_latency + decode_latency
    total_throughput = (gpu_batch_size * (seq_len + gen_len - 1)) / total_latency
    
    _, gpu_peak_mem = env.gpu.mem_stats()
    
    model_bytes = opt_config.model_bytes()
    cache_bytes = opt_config.cache_bytes(gpu_batch_size, seq_len + gen_len)
    hidden_bytes = opt_config.hidden_bytes(gpu_batch_size, seq_len + gen_len)

    print(f"Prefill Latency: {prefill_latency:.3f} s")
    print(f"Decode Latency: {decode_latency:.3f} s")
    print(f"Total Throughput: {total_throughput:.3f} token/s")

    # === 2. Perplexity Evaluation ===
    print("\n" + "="*20 + " Running Perplexity Evaluation " + "="*20)
    print("Loading dataset...")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    input_ids_full = encodings.input_ids
    
    batch_input_ids = []
    for i in range(gpu_batch_size):
        start = i * seq_len
        end = start + seq_len
        if end > input_ids_full.shape[1]:
            print("Warning: Not enough data for batch size, repeating data.")
            start = 0 
            end = seq_len
        segment = input_ids_full[:, start:end].numpy() 
        batch_input_ids.append(segment[0]) 
        
    batch_input_ids = np.array(batch_input_ids) 
    
    print(f"Computing PPL on {gpu_batch_size} sequences of length {seq_len} tokens...")
    
    try:
        if hasattr(model, 'get_logits'):
            # Clear timers or anything specific if needed
            logits = model.get_logits(batch_input_ids)
        else:
            print("Error: get_logits not found in model!")
            return

        target_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(logits.device)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        ppl = torch.exp(loss)
        result_str = f"🚀 Final Result PPL: {ppl.item():.2f}"
        print(result_str)
        
        # Print the accumulated Load Balancing Auxiliary Loss from MLP
        if hasattr(env.gpu, 'mlp_aux_loss'):
            lb_str = f"⚖️  Accumulated Load Balancing Loss (MLP Top-3): {env.gpu.mlp_aux_loss:.4f}"
            print(lb_str)
            result_str += f"\n{lb_str}"
        
        print(f"Detailed results written to {os.path.abspath(log_filename)}")
        
        # Write to log file
        with open(log_filename, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Batch Size: {gpu_batch_size}\n")
            f.write(f"Top-K: {args.top_k}\n")
            f.write(f"Offload Percent: {percent}\n")
            f.write(f"Weight Compression: {compress_weight}\n")
            f.write(f"Cache Compression: {compress_cache}\n")
            f.write(f"Seq Len: {seq_len}\n")
            f.write(f"Gen Len: {gen_len}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("-" * 20 + "\n")
            f.write(f"model size: {model_bytes/GB:.3f} GB\n")
            f.write(f"cache size: {cache_bytes/GB:.3f} GB\n")
            f.write(f"peak gpu mem: {gpu_peak_mem/GB:.3f} GB\n")
            f.write(f"prefill latency: {prefill_latency:.3f} s\n")
            f.write(f"decode latency: {decode_latency:.3f} s\n")
            f.write(f"total latency: {total_latency:.3f} s\n")
            f.write(f"total throughput: {total_throughput:.3f} token/s\n")
            f.write("-" * 20 + "\n")
            f.write(f"{result_str}\n")
            
    except Exception as e:
        print(f"\nExecution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
