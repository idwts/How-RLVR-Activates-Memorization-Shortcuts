"""
Ablation Experiments: Verify the role of Functional Anchor (L18-20) and Structural Adapter (L21-22)

Experimental Design (all operations on RLVR model):
1.1 Anchor layer reset (L18-20 to base)
1.2 Adapter layer reset (L21-22 to base)
1.3 Anchor layer randomization (L18-20 to random)
1.4 Adapter layer randomization (L21-22 to random)

2.1 Keep only Anchor layer (only L18-20 from RLVR)
2.2 Keep only Adapter layer (only L21-22 from RLVR)
2.3 Keep Anchor+Adapter (L18-22 from RLVR)

Support multi-GPU parallel evaluation, four processes save results in real-time, 
merge and visualize after all complete.
"""
import json
import os
import copy
import shutil
import sys
import tempfile
import queue
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ttrl.verifier.auto_verify import auto_verify


# Configuration
CONFIG = {
    'model_paths': {
        'before':  '../Qwen2.5/Qwen2.5-Math-7B',
        'after': '../rethink_rlvr_reproduce-incorrect-qwen2.5_math_7b-lr5e-7-kl0.00-step150',
    },
    
    'allowed_experiments': ['0.2_base_model'],  # Only run base model
    
    'eval_results_dir': '../outputs/eval_outputs',
    'datasets': {
        'MATH-500': {
            'path': '../data/MATH-TTT/test.json',
            'rollouts': 1,
        },
        'LiveMathBench': {
            'path':  '../data/LiveMathBench/livemathbench_2504_v2.json',
            'rollouts': 4,
        },
        'MinervaMath': {
            'path': '../data/MinervaMath/minervamath.json',
            'rollouts':  2,
        },
    },
    'anchor_layers': [18, 19, 20],
    'adapter_layers': [21, 22],
    'max_tokens': 3072,
    'temperature': 0.0,
    'temperature_at_k': 0.6,
    'max_samples_per_dataset': 999,
    'num_gpus': 4,
    'gpu_ids': [0, 1, 2, 3],
    'output_dir': 'ablation_results',
    'partial_dir': 'ablation_results/partials',
    'model_prefixes': {
        'origin': ['Qwen2.5-Math-7B'],
        'incorrect': ['rethink_rlvr_reproduce-incorrect-qwen2.5_math_7b-lr5e-7-kl0.00-step150']
    },
}

MAX_SAMPLES_PER_DATASET = 50

BASE_MODEL_PROMPT = (
    "A conversation between User and Assistant.The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer."
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n"
    "<|user|>\n{}\n<|assistant|>\n<think>"
)


def save_json_atomic(data: Dict, path: str) -> None:
    """Save JSON atomically to avoid corruption"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def prepare_partial_dir(partial_dir: str) -> Path:
    """Prepare partial results directory"""
    path = Path(partial_dir)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def merge_partial_results(partial_dir: str) -> Dict[str, Dict]:
    """Merge partial results from all GPU workers"""
    merged:  Dict[str, Dict] = {}
    path = Path(partial_dir)
    if not path.exists():
        return merged
    for file in sorted(path.glob('gpu*_partial.json')):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        merged.update(data)
    return merged


def load_eval_results(results_dir: str, model_prefix: str, dataset_name: str) -> Optional[Dict]:
    """Load evaluation results for specific model and dataset"""
    for filename in os.listdir(results_dir):
        if filename.endswith(f'{dataset_name}.json'):
            for prefix, variants in CONFIG['model_prefixes'].items():
                if prefix == model_prefix:
                    for variant in variants:
                        if filename.startswith(variant):
                            filepath = os.path.join(results_dir, filename)
                            with open(filepath, 'r', encoding='utf-8') as f:
                                return json.load(f)
    return None


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load dataset from file"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_samples(results_dir: str, datasets: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Collect leakage and generalization samples from evaluation results"""
    leakage_samples = []
    generalization_samples = []
    
    for dataset_name, dataset_info in datasets.items():
        print(f"\nProcessing dataset: {dataset_name}")
        
        origin_results = load_eval_results(results_dir, 'origin', dataset_name)
        incorrect_results = load_eval_results(results_dir, 'incorrect', dataset_name)
        
        if origin_results is None or incorrect_results is None: 
            print(f"  Warning: Skip {dataset_name}:  missing evaluation results")
            continue
        
        dataset_data = load_dataset(dataset_info['path'])
        origin_metadata = {item['index']: item for item in origin_results.get('metadata', [])}
        incorrect_metadata = {item['index']: item for item in incorrect_results.get('metadata', [])}
        
        leak_count = 0
        gen_count = 0
        
        for idx, data_item in enumerate(dataset_data):
            if idx not in origin_metadata or idx not in incorrect_metadata: 
                continue
            
            origin_correct = bool(origin_metadata[idx].get('correct', False))
            incorrect_correct = bool(incorrect_metadata[idx].get('correct', False))
            
            sample = {
                'dataset': dataset_name,
                'index': idx,
                'prompt': data_item['prompt'],
                'answer': data_item['answer'],
                'origin_correct': origin_correct,
                'incorrect_correct': incorrect_correct,
            }
            
            if not origin_correct and incorrect_correct:
                sample['label'] = 'leakage'
                leakage_samples.append(sample)
                leak_count += 1
            elif origin_correct and incorrect_correct:
                sample['label'] = 'generalization'
                generalization_samples.append(sample)
                gen_count += 1
        
        print(f"  Leakage samples: {leak_count}, Generalization samples: {gen_count}")
    
    return leakage_samples, generalization_samples


def reset_ffn_layers(
    target_model: AutoModelForCausalLM,
    source_model: AutoModelForCausalLM,
    layer_indices: List[int]
) -> AutoModelForCausalLM: 
    """Reset FFN layers from source model to target model"""
    model = copy.deepcopy(target_model)
    for layer_idx in layer_indices: 
        if layer_idx >= len(model.model.layers):
            continue
        target_mlp = model.model.layers[layer_idx].mlp
        source_mlp = source_model.model.layers[layer_idx].mlp
        target_mlp.gate_proj.weight.data.copy_(source_mlp.gate_proj.weight.data)
        target_mlp.up_proj.weight.data.copy_(source_mlp.up_proj.weight.data)
        target_mlp.down_proj.weight.data.copy_(source_mlp.down_proj.weight.data)
    return model


def keep_only_layers(
    rlvr_model: AutoModelForCausalLM,
    base_model: AutoModelForCausalLM,
    keep_layer_indices: List[int]
) -> AutoModelForCausalLM:
    """Keep only specified layers from RLVR model, reset others to base"""
    model = copy.deepcopy(rlvr_model)
    num_layers = len(model.model.layers)
    
    for layer_idx in range(num_layers):
        if layer_idx not in keep_layer_indices:
            target_mlp = model.model.layers[layer_idx].mlp
            source_mlp = base_model.model.layers[layer_idx].mlp
            target_mlp.gate_proj.weight.data.copy_(source_mlp.gate_proj.weight.data)
            target_mlp.up_proj.weight.data.copy_(source_mlp.up_proj.weight.data)
            target_mlp.down_proj.weight.data.copy_(source_mlp.down_proj.weight.data)
    
    return model


def randomize_ffn_layers(
    model: AutoModelForCausalLM,
    layer_indices:  List[int],
    preserve_norm: bool = True
) -> AutoModelForCausalLM:
    """Randomize FFN layers while optionally preserving norm"""
    model = copy.deepcopy(model)
    
    for layer_idx in layer_indices:
        if layer_idx >= len(model.model.layers):
            continue
        
        mlp = model.model.layers[layer_idx].mlp
        
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(mlp, proj_name)
            original_weight = proj.weight.data
            original_norm = torch.norm(original_weight, p='fro')
            
            random_weight = torch.randn_like(original_weight)
            
            if preserve_norm:
                random_norm = torch.norm(random_weight, p='fro')
                random_weight = random_weight * (original_norm / random_norm)
            
            proj.weight.data.copy_(random_weight)
    
    return model


def define_experiments() -> Dict[str, Dict]:
    """Define all ablation experiments"""
    anchor_layers = CONFIG['anchor_layers']
    adapter_layers = CONFIG['adapter_layers']
    both_layers = anchor_layers + adapter_layers
    
    return {
        '0.1_rlvr_baseline': {
            'description': 'RLVR Model Baseline',
            'operation': 'none',
            'layers': None,
        },
        '0.2_base_model': {
            'description': 'Base Model (Not fine-tuned)',
            'operation': 'base_only',
            'layers': None,
        },
        '1.1_anchor_reset': {
            'description': f'Anchor layer reset (L{anchor_layers} to base)',
            'operation': 'reset',
            'layers':  anchor_layers,
        },
        '1.2_adapter_reset': {
            'description':  f'Adapter layer reset (L{adapter_layers} to base)',
            'operation': 'reset',
            'layers': adapter_layers,
        },
        '1.3_anchor_random': {
            'description': f'Anchor layer randomization (L{anchor_layers} to random)',
            'operation': 'random',
            'layers': anchor_layers,
        },
        '1.4_adapter_random':  {
            'description': f'Adapter layer randomization (L{adapter_layers} to random)',
            'operation': 'random',
            'layers': adapter_layers,
        },
        '2.1_keep_only_anchor': {
            'description': f'Keep only Anchor layer (only L{anchor_layers} from RLVR)',
            'operation': 'keep_only',
            'layers': anchor_layers,
        },
        '2.2_keep_only_adapter': {
            'description': f'Keep only Adapter layer (only L{adapter_layers} from RLVR)',
            'operation': 'keep_only',
            'layers': adapter_layers,
        },
        '2.3_keep_anchor_and_adapter': {
            'description': f'Keep Anchor+Adapter (L{both_layers} from RLVR)',
            'operation': 'keep_only',
            'layers': both_layers,
        },
    }


def create_modified_model(
    model_rlvr: AutoModelForCausalLM,
    model_base: AutoModelForCausalLM,
    operation: str,
    layers: Optional[List[int]]
) -> AutoModelForCausalLM:
    """Create modified model based on operation type"""
    if operation == 'none':
        return model_rlvr
    if operation == 'base_only': 
        return model_base
    if operation == 'reset': 
        return reset_ffn_layers(model_rlvr, model_base, layers or [])
    if operation == 'random':
        return randomize_ffn_layers(model_rlvr, layers or [])
    if operation == 'keep_only':
        return keep_only_layers(model_rlvr, model_base, layers or [])
    raise ValueError(f"Unknown operation: {operation}")


def evaluate_samples_on_gpu(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[Dict],
    device: torch.device,
    is_base_model: bool = False
) -> List[Tuple[int, str, bool]]:
    """Evaluate samples on specific GPU"""
    model.eval()
    results:  List[Tuple[int, str, bool]] = []
    
    for sample in tqdm(samples, desc=f"GPU {device}", leave=False):
        dataset_name = sample['dataset']
        dataset_info = CONFIG['datasets'][dataset_name]
        rollouts = dataset_info['rollouts']
        temperature = CONFIG['temperature'] if rollouts == 1 else CONFIG['temperature_at_k']
        
        if is_base_model:
            prompt = BASE_MODEL_PROMPT.format(sample['prompt'])
        else:
            messages = [{"role": "user", "content":  sample['prompt']}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        sample_correct = False
        
        for _ in range(rollouts):
            with torch.no_grad():
                if temperature == 0:
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=CONFIG['max_tokens'],
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                else:
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=CONFIG['max_tokens'],
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            
            generated_ids = outputs[0, input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            fake_output = type('obj', (), {'outputs': [type('obj', (), {'text': generated_text})()]})()
            result = auto_verify('math', 1, [fake_output], [sample['answer']])
            
            if result[0]:
                sample_correct = True
                break
        
        results.append((sample['index'], sample['dataset'], sample_correct))
    
    return results


def gpu_worker(
        gpu_id: int,
        experiment_queue: mp.Queue,
        model_base_path: str,
        model_rlvr_path: str,
        leakage_samples: List[Dict],
        generalization_samples:  List[Dict],
        experiments_config: Dict[str, Dict]
) -> None:
    """Worker function for each GPU process"""
    partial_path = os.path.join(CONFIG['partial_dir'], f'gpu{gpu_id}_partial.json')
    partial_results:  Dict[str, Dict] = {}
    
    if os.path.exists(partial_path):
        with open(partial_path, 'r', encoding='utf-8') as f:
            partial_results = json.load(f)
    
    try:
        device = torch.device(f'cuda:{gpu_id}')
        print(f"[GPU {gpu_id}] Initializing...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_base_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"[GPU {gpu_id}] Loading base model...")
        model_base = AutoModelForCausalLM.from_pretrained(
            model_base_path,
            device_map={'': device},
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model_base.eval()
        
        print(f"[GPU {gpu_id}] Loading RLVR model...")
        model_rlvr = AutoModelForCausalLM.from_pretrained(
            model_rlvr_path,
            device_map={'': device},
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model_rlvr.eval()
        
        print(f"[GPU {gpu_id}] Ready")
        
        while True:
            try:
                exp_name = experiment_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            if exp_name == 'DONE':
                break
            
            exp_config = experiments_config[exp_name]
            print(f"[GPU {gpu_id}] Starting experiment: {exp_name}")
            
            is_base_model_exp = (exp_config['operation'] == 'base_only')
            
            if exp_config['operation'] == 'none':
                model = model_rlvr
            elif exp_config['operation'] == 'base_only':
                model = model_base
            else:
                model = create_modified_model(
                    model_rlvr=model_rlvr,
                    model_base=model_base,
                    operation=exp_config['operation'],
                    layers=exp_config['layers']
                ).to(device)
            
            leak_results = evaluate_samples_on_gpu(model, tokenizer, leakage_samples, device, is_base_model_exp)
            gen_results = evaluate_samples_on_gpu(model, tokenizer, generalization_samples, device, is_base_model_exp)
            
            if exp_config['operation'] not in ['none', 'base_only']: 
                del model
                torch.cuda.empty_cache()
            
            leak_by_dataset = defaultdict(list)
            for idx, dataset, correct in leak_results:
                leak_by_dataset[dataset].append(correct)
            
            gen_by_dataset = defaultdict(list)
            for idx, dataset, correct in gen_results:
                gen_by_dataset[dataset].append(correct)
            
            partial_results[exp_name] = {
                'leakage':  {k: v for k, v in leak_by_dataset.items()},
                'generalization': {k: v for k, v in gen_by_dataset.items()},
            }
            
            save_json_atomic(partial_results, partial_path)
            print(f"[GPU {gpu_id}] Completed experiment: {exp_name}")
        
        print(f"[GPU {gpu_id}] Exiting")
        
    except Exception as exc:
        print(f"[GPU {gpu_id}] Error occurred: {exc}")
        traceback.print_exc()


def run_parallel_experiments(
    leakage_samples: List[Dict],
    generalization_samples: List[Dict],
    num_gpus: int = 4,
    gpu_ids: Optional[List[int]] = None
) -> Dict[str, Dict]:
    """Run experiments in parallel across multiple GPUs"""
    if gpu_ids is None:
        gpu_ids = list(range(num_gpus))
    
    partial_dir = prepare_partial_dir(CONFIG['partial_dir'])
    
    max_samples = min(CONFIG['max_samples_per_dataset'], MAX_SAMPLES_PER_DATASET)
    
    leak_by_dataset = defaultdict(list)
    gen_by_dataset = defaultdict(list)
    
    for s in leakage_samples: 
        leak_by_dataset[s['dataset']].append(s)
    for s in generalization_samples: 
        gen_by_dataset[s['dataset']].append(s)
    
    limited_leakage:  List[Dict] = []
    limited_gen: List[Dict] = []
    
    for dataset, samples in leak_by_dataset.items():
        limited_leakage.extend(samples[:max_samples])
    for dataset, samples in gen_by_dataset.items():
        limited_gen.extend(samples[:max_samples])
    
    print(f"\nLimited sample counts (per dataset <= {max_samples}):")
    print(f"  Leakage samples: {len(limited_leakage)}")
    print(f"  Generalization samples: {len(limited_gen)}")
    
    experiments = define_experiments()
    allowed = CONFIG.get('allowed_experiments')
    if allowed:
        experiments = {k: v for k, v in experiments.items() if k in allowed}
    
    experiment_names = list(experiments.keys())
    if not experiment_names:
        print("Warning: No experiments to run")
        return {}
    
    print(f"\nTotal experiments: {len(experiment_names)}")
    print(f"Using GPUs: {gpu_ids}")
    
    mp.set_start_method('spawn', force=True)
    
    experiment_queue:  mp.Queue = mp.Queue()
    for exp_name in experiment_names: 
        experiment_queue.put(exp_name)
    for _ in gpu_ids:
        experiment_queue.put('DONE')
    
    processes = []
    for gpu_id in gpu_ids: 
        p = mp.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                experiment_queue,
                CONFIG['model_paths']['before'],
                CONFIG['model_paths']['after'],
                limited_leakage,
                limited_gen,
                experiments
            )
        )
        p.start()
        processes.append(p)
    
    for proc in processes:
        proc.join()
        if proc.exitcode not in (0, None):
            print(f"Warning: Process {proc.pid} exited abnormally, exitcode={proc.exitcode}")
    
    all_results = merge_partial_results(str(partial_dir))
    
    if not all_results:
        print("Warning: Failed to merge any experiment results, please check logs.")
    else:
        print(f"Successfully merged {len(all_results)} experiment results from {partial_dir}")
    
    shutil.rmtree(partial_dir, ignore_errors=True)
    
    return all_results


def run_sequential_experiments(
        leakage_samples: List[Dict],
        generalization_samples: List[Dict],
        gpu_id: int = 0
) -> Dict[str, Dict]:
    """Run experiments sequentially on single GPU"""
    device = torch.device(f'cuda:{gpu_id}')
    print(f"\nUsing single GPU mode:  cuda:{gpu_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_paths']['before'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading base model...")
    model_base = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_paths']['before'],
        device_map={'': device},
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model_base.eval()
    
    print("Loading RLVR model...")
    model_rlvr = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_paths']['after'],
        device_map={'':  device},
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model_rlvr.eval()
    
    max_samples = min(CONFIG['max_samples_per_dataset'], MAX_SAMPLES_PER_DATASET)
    
    leak_by_dataset = defaultdict(list)
    gen_by_dataset = defaultdict(list)
    
    for s in leakage_samples:
        leak_by_dataset[s['dataset']].append(s)
    for s in generalization_samples: 
        gen_by_dataset[s['dataset']].append(s)
    
    limited_leakage: List[Dict] = []
    limited_gen: List[Dict] = []
    
    for dataset, samples in leak_by_dataset.items():
        limited_leakage.extend(samples[:max_samples])
    for dataset, samples in gen_by_dataset.items():
        limited_gen.extend(samples[:max_samples])
    
    experiments = define_experiments()
    allowed = CONFIG.get('allowed_experiments')
    if allowed:
        experiments = {k: v for k, v in experiments.items() if k in allowed}
    
    if not experiments:
        print("Warning: No experiments to run")
        return {}
    
    all_results:  Dict[str, Dict] = {}
    
    for exp_name, exp_config in experiments.items():
        print(f"\n{'=' * 60}")
        print(f"Experiment: {exp_name}")
        print(f"Description: {exp_config['description']}")
        print('=' * 60)
        
        is_base_model_exp = (exp_config['operation'] == 'base_only')
        
        if exp_config['operation'] == 'none':
            model = model_rlvr
        elif exp_config['operation'] == 'base_only': 
            model = model_base
        else:
            model = create_modified_model(
                model_rlvr=model_rlvr,
                model_base=model_base,
                operation=exp_config['operation'],
                layers=exp_config['layers']
            ).to(device)
        
        leak_results = evaluate_samples_on_gpu(model, tokenizer, limited_leakage, device, is_base_model_exp)
        gen_results = evaluate_samples_on_gpu(model, tokenizer, limited_gen, device, is_base_model_exp)
        
        leak_by_dataset_result = defaultdict(list)
        for idx, dataset, correct in leak_results:
            leak_by_dataset_result[dataset].append(correct)
        
        gen_by_dataset_result = defaultdict(list)
        for idx, dataset, correct in gen_results:
            gen_by_dataset_result[dataset].append(correct)
        
        all_results[exp_name] = {
            'leakage': {k: v for k, v in leak_by_dataset_result.items()},
            'generalization': {k:  v for k, v in gen_by_dataset_result.items()},
        }
        
        if exp_config['operation'] not in ['none', 'base_only']: 
            del model
            torch.cuda.empty_cache()
        
        leak_all = [c for results in leak_by_dataset_result.values() for c in results]
        gen_all = [c for results in gen_by_dataset_result.values() for c in results]
        print(f"  Leakage:  {np.mean(leak_all):.2%} ({sum(leak_all)}/{len(leak_all)})")
        print(f"  Generalization: {np.mean(gen_all):.2%} ({sum(gen_all)}/{len(gen_all)})")
    
    del model_base, model_rlvr
    torch.cuda.empty_cache()
    
    return all_results


def visualize_results(all_results:  Dict[str, Dict], output_dir: str) -> None:
    """Generate visualization plots from results"""
    os.makedirs(output_dir, exist_ok=True)
    
    summary_data = []
    for exp_name, exp_results in all_results.items():
        for sample_type in ['leakage', 'generalization']:
            results = exp_results.get(sample_type, {})
            all_correct = []
            for v in results.values():
                all_correct.extend(v)
            acc = np.mean(all_correct) if all_correct else 0.0
            summary_data.append({
                'experiment': exp_name,
                'sample_type': sample_type,
                'accuracy': acc,
                'correct': sum(all_correct) if all_correct else 0,
                'total': len(all_correct) if all_correct else 0,
            })
    
    df = pd.DataFrame(summary_data)
    if df.empty:
        print("Warning: No visualization data")
        return
    
    # Main comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, sample_type in enumerate(['leakage', 'generalization']):
        ax = axes[idx]
        sample_df = df[df['sample_type'] == sample_type].sort_values('experiment')
        
        colors = []
        for exp in sample_df['experiment']: 
            if 'baseline' in exp:
                colors.append('#2ecc71')
            elif 'anchor' in exp:
                colors.append('#3498db')
            elif 'adapter' in exp:
                colors.append('#e74c3c')
            else:
                colors.append('#9b59b6')
        
        bars = ax.bar(range(len(sample_df)), sample_df['accuracy'], color=colors, alpha=0.8)
        ax.set_xticks(range(len(sample_df)))
        ax.set_xticklabels(
            [e.split('_', 1)[1] if '_' in e else e for e in sample_df['experiment']],
            rotation=45, ha='right', fontsize=9
        )
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'{sample_type.capitalize()} Samples', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, acc in zip(bars, sample_df['accuracy']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'main_comparison.svg'), dpi=600, format='svg')
    plt.savefig(os.path.join(output_dir, 'main_comparison.png'), dpi=300)
    plt.close()
    
    # Heatmaps for each sample type
    for sample_type in ['leakage', 'generalization']:
        sample_df = df[df['sample_type'] == sample_type]
        plt.figure(figsize=(10, 6))
        
        pivot = sample_df.set_index('experiment')['accuracy'].to_frame().sort_index()
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2%',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Accuracy'}
        )
        
        plt.title(f'Ablation Results - {sample_type.capitalize()} Samples',
                  fontsize=14, fontweight='bold')
        plt.xlabel('')
        plt.ylabel('Experiment', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_{sample_type}.svg'), dpi=600, format='svg')
        plt.savefig(os.path.join(output_dir, f'heatmap_{sample_type}.png'), dpi=300)
        plt.close()
    
    # By-dataset breakdown
    datasets = list(CONFIG['datasets'].keys())
    for sample_type in ['leakage', 'generalization']:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5))
        if len(datasets) == 1:
            axes = [axes]
        
        for idx, dataset_name in enumerate(datasets):
            ax = axes[idx]
            exp_names = []
            accuracies = []
            
            for exp_name in sorted(all_results.keys()):
                exp_results = all_results[exp_name]
                dataset_results = exp_results.get(sample_type, {}).get(dataset_name, [])
                acc = np.mean(dataset_results) if dataset_results else 0.0
                exp_names.append(exp_name.split('_', 1)[1] if '_' in exp_name else exp_name)
                accuracies.append(acc)
            
            colors = []
            for exp in exp_names:
                if 'baseline' in exp:
                    colors.append('#2ecc71')
                elif 'anchor' in exp:
                    colors.append('#3498db')
                elif 'adapter' in exp:
                    colors.append('#e74c3c')
                else:
                    colors.append('#9b59b6')
            
            ax.bar(range(len(exp_names)), accuracies, color=colors, alpha=0.8)
            ax.set_xticks(range(len(exp_names)))
            ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax.set_title(f'{dataset_name}\n({sample_type.capitalize()})', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'by_dataset_{sample_type}.svg'), dpi=600, format='svg')
        plt.savefig(os.path.join(output_dir, f'by_dataset_{sample_type}.png'), dpi=300)
        plt.close()
    
    # Accuracy change vs baseline
    baseline_key = '0.1_rlvr_baseline'
    if baseline_key in all_results: 
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, sample_type in enumerate(['leakage', 'generalization']):
            ax = axes[idx]
            baseline_results = all_results[baseline_key].get(sample_type, {})
            baseline_all = []
            for v in baseline_results.values():
                baseline_all.extend(v)
            baseline_acc = np.mean(baseline_all) if baseline_all else 0.0
            
            changes = {}
            for exp_name in all_results.keys():
                if exp_name == baseline_key:
                    continue
                results = all_results[exp_name].get(sample_type, {})
                all_correct = []
                for v in results.values():
                    all_correct.extend(v)
                if all_correct:
                    acc = np.mean(all_correct)
                    label = exp_name.split('_', 1)[1] if '_' in exp_name else exp_name
                    changes[label] = acc - baseline_acc
            
            if changes:
                names = list(changes.keys())
                values = list(changes.values())
                colors = ['#3498db' if 'anchor' in n else '#e74c3c' if 'adapter' in n else '#9b59b6'
                          for n in names]
                
                bars = ax.barh(range(len(names)), values, color=colors, alpha=0.8)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names, fontsize=9)
                ax.set_xlabel('Accuracy Change (vs Baseline)', fontsize=11, fontweight='bold')
                ax.set_title(f'{sample_type.capitalize()} Samples', fontsize=12, fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax.grid(axis='x', alpha=0.3)
                
                for bar, val in zip(bars, values):
                    x_pos = bar.get_width() + 0.01 if val >= 0 else bar.get_width() - 0.01
                    ha = 'left' if val >= 0 else 'right'
                    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                            f'{val: +.1%}', ha=ha, va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_change.svg'), dpi=600, format='svg')
        plt.savefig(os.path.join(output_dir, 'accuracy_change.png'), dpi=300)
        plt.close()
    
    print(f"\nVisualization results saved to {output_dir}")


def save_results_to_json(
    all_results: Dict[str, Dict],
    leakage_samples: List[Dict],
    generalization_samples:  List[Dict],
    output_dir: str
) -> None:
    """Save results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    experiments = define_experiments()
    summary:  Dict[str, Dict] = {}
    
    for exp_name, exp_results in all_results.items():
        summary[exp_name] = {
            'description': experiments.get(exp_name, {}).get('description', exp_name),
        }
        
        for sample_type in ['leakage', 'generalization']:
            results = exp_results.get(sample_type, {})
            by_dataset = {}
            all_correct = []
            
            for dataset_name, correctness in results.items():
                if correctness:
                    by_dataset[dataset_name] = {
                        'accuracy': float(np.mean(correctness)),
                        'correct':  int(sum(correctness)),
                        'total': len(correctness)
                    }
                    all_correct.extend(correctness)
            
            summary[exp_name][sample_type] = {
                'overall':  {
                    'accuracy': float(np.mean(all_correct)) if all_correct else 0.0,
                    'correct':  int(sum(all_correct)) if all_correct else 0,
                    'total': len(all_correct)
                },
                'by_dataset': by_dataset
            }
    
    output_data = {
        'config': {
            'anchor_layers': CONFIG['anchor_layers'],
            'adapter_layers': CONFIG['adapter_layers'],
            'datasets': list(CONFIG['datasets'].keys()),
            'num_gpus': CONFIG['num_gpus'],
        },
        'sample_counts': {
            'leakage': len(leakage_samples),
            'generalization': len(generalization_samples),
        },
        'results': summary
    }
    
    output_file = os.path.join(output_dir, 'ablation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")


def print_summary(all_results: Dict[str, Dict]) -> None:
    """Print experiment summary table"""
    experiments = define_experiments()
    
    print("\n" + "=" * 80)
    print("Experiment Summary")
    print("=" * 80)
    
    headers = ["Experiment", "Description", "Leakage", "Generalization"]
    rows = []
    
    for exp_name in sorted(all_results.keys()):
        exp_results = all_results[exp_name]
        desc = experiments.get(exp_name, {}).get('description', '')[: 30]
        
        leak_all = []
        gen_all = []
        for v in exp_results.get('leakage', {}).values():
            leak_all.extend(v)
        for v in exp_results.get('generalization', {}).values():
            gen_all.extend(v)
        
        leak_acc = f"{np.mean(leak_all):.1%}" if leak_all else "N/A"
        gen_acc = f"{np.mean(gen_all):.1%}" if gen_all else "N/A"
        
        rows.append([exp_name, desc, leak_acc, gen_acc])
    
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    header_str = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    
    print(header_str)
    print("-" * len(header_str))
    
    for row in rows:
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
    
    print("=" * 80)


def main() -> None:
    """Main entry point"""
    print("=" * 60)
    print("Ablation Experiments v3:  Anchor (L18-20) & Adapter (L21-22)")
    print("=" * 60)
    print(f"Anchor layers: {CONFIG['anchor_layers']}")
    print(f"Adapter layers: {CONFIG['adapter_layers']}")
    print(f"GPU configuration: {CONFIG['gpu_ids']}")
    print(f"Max samples per dataset: {CONFIG['max_samples_per_dataset']}")
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    print("\nStep 1: Collecting samples...")
    leakage_samples, generalization_samples = collect_samples(
        CONFIG['eval_results_dir'],
        CONFIG['datasets']
    )
    
    print(f"\nTotal:")
    print(f"  Leakage samples (wrong to right): {len(leakage_samples)}")
    print(f"  Generalization samples (always right): {len(generalization_samples)}")
    
    if not leakage_samples and not generalization_samples:
        print("\nNo valid samples found, exiting")
        return
    
    print("\nStep 2: Running ablation experiments...")
    
    if len(CONFIG['gpu_ids']) == 1:
        all_results = run_sequential_experiments(
            leakage_samples,
            generalization_samples,
            gpu_id=CONFIG['gpu_ids'][0]
        )
    else:
        all_results = run_parallel_experiments(
            leakage_samples,
            generalization_samples,
            num_gpus=CONFIG['num_gpus'],
            gpu_ids=CONFIG['gpu_ids']
        )
    
    print("\nStep 3: Saving results...")
    save_results_to_json(all_results, leakage_samples, generalization_samples, CONFIG['output_dir'])
    
    print("\nStep 4: Generating visualizations...")
    visualize_results(all_results, CONFIG['output_dir'])
    
    print_summary(all_results)
    
    print("\nComplete!")


if __name__ == "__main__":
    main()