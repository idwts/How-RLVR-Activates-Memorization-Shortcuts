"""
MLP Neuron Steering Experiment - Enhanced Evaluation with auto_verify
Goal: Identify task-relevant neurons and quantitatively measure steering effects
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.multiprocessing as mp
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ttrl.verifier.auto_verify import auto_verify


# Configuration
CONFIG = {
    "model_path": "../rethink_rlvr_reproduce-incorrect-qwen2.5_math_7b-lr5e-7-kl0.00-step150",
    "tokenizer_path":  "../Qwen2.5/Qwen2.5-Math-7B",
    "correct_answers_file": "correct_answers_data.json",
    "dataset_dir": "../data",
    
    "anchor_layers": [18, 19, 20],
    "control_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27],
    "top_k_neurons": 50,
    "top_tokens": 10,
    
    "amplify_factors": [1.5, 2.0, 3.0],
    "suppress_factors": [0.5, 0.2, 0.0],
    
    "num_rollouts": 2,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_new_tokens": 512,
    
    "target_datasets": ["MATH-500", "MinervaMath"],
    "samples_per_dataset": 999,
    
    "output_dir": "steering_full_results"
}

BENCHMARKS = {
    "MATH-500": {
        "path": "../data/MATH-TTT/test.json",
        "rollouts": 1,
    },
    "LiveMathBench": {
        "path":  "../data/LiveMathBench/livemathbench_2504_v2.json",
        "rollouts": 4,
    },
    "MinervaMath":  {
        "path": "../data/MinervaMath/minervamath.json",
        "rollouts":  2,
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(CONFIG["output_dir"], exist_ok=True)


# Answer Verification

def verify_outputs(outputs: List[str], ground_truths: List[str], task:  str = "math") -> List[bool]:
    """
    Verify correctness of model outputs using auto_verify
    
    Args:
        outputs: List of generated texts
        ground_truths: List of corresponding correct answers
        task: Task type (default "math")
    
    Returns:
        List of booleans indicating correctness
    """
    class DummyOutput:
        def __init__(self, text):
            self.text = text
    
    class DummyCompletionOutput:
        def __init__(self, text):
            self.outputs = [DummyOutput(text)]
    
    wrapped_outputs = [DummyCompletionOutput(text) for text in outputs]
    correctness = auto_verify(task, 1, wrapped_outputs, ground_truths)
    
    return correctness


# Data Loading

def load_correct_answers(file_path: str) -> List[Dict]:
    """Load correct answers from file"""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Correct answers file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get("correct_answers", [])


def load_dataset_questions(dataset_name: str, dataset_dir: str) -> List[Dict]:
    """Load questions from dataset"""
    if dataset_name in BENCHMARKS:
        dataset_path = Path(BENCHMARKS[dataset_name]["path"])
    else:
        dataset_path = Path(dataset_dir) / f"{dataset_name}.jsonl"
        if not dataset_path.exists():
            dataset_path = Path(dataset_dir) / f"{dataset_name}.json"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")

    questions = []
    if dataset_path.suffix == ".jsonl":
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f: 
                questions.append(json.loads(line.strip()))
    else:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                questions = data
            elif isinstance(data, dict) and "data" in data: 
                questions = data["data"]
            else:
                raise ValueError(f"Unknown dataset format: {dataset_path}")

    return questions


def prepare_test_cases(
    correct_answers_file: str,
    dataset_dir: str,
    target_datasets: List[str],
    samples_per_dataset: int
) -> List[Dict]:
    """Prepare test cases from correct answers"""
    correct_answers = load_correct_answers(correct_answers_file)
    
    dataset_samples = defaultdict(list)
    for item in correct_answers:
        dataset = item.get("dataset")
        if dataset in target_datasets:
            dataset_samples[dataset].append(item)
    
    test_cases = []
    
    for dataset in target_datasets:
        if dataset not in dataset_samples:
            print(f"Warning: No correct answers found for dataset {dataset}")
            continue
        
        try:
            all_questions = load_dataset_questions(dataset, dataset_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        
        selected_samples = dataset_samples[dataset][: samples_per_dataset]
        
        for sample in selected_samples:
            question_idx = sample.get("question_index")
            
            if question_idx < len(all_questions):
                question_data = all_questions[question_idx]
                
                prompt = question_data.get("problem") or question_data.get("question") or question_data.get("prompt")
                answer = question_data.get("solution") or question_data.get("answer")
                
                if prompt and answer:
                    test_cases.append({
                        "prompt": prompt,
                        "answer": answer,
                        "dataset": dataset,
                        "question_index": question_idx
                    })
            else:
                print(f"Warning: Question index {question_idx} out of range for {dataset}")
    
    print(f"\nLoaded {len(test_cases)} test cases from {len(target_datasets)} datasets")
    for dataset in target_datasets:
        count = sum(1 for t in test_cases if t['dataset'] == dataset)
        print(f"  - {dataset}: {count} samples")
    
    return test_cases


# Neuron Analysis

def capture_mlp_activations(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    layer_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Capture MLP activations"""
    activations = {}
    mlp = model.model.layers[layer_idx].mlp
    
    def hook_fn(module, input):
        activations['keys'] = input[0].detach()
    
    handle = mlp.down_proj.register_forward_pre_hook(hook_fn)
    
    with torch.no_grad():
        model(input_ids)
    
    handle.remove()
    
    keys = activations['keys']
    if keys.size(0) > 1:
        keys_last = keys[: , -1, : ].mean(dim=0)
    else:
        keys_last = keys[0, -1, :]
    
    W_down = mlp.down_proj.weight.detach()
    
    return keys_last, W_down.T


def project_neurons_to_vocab(
    W_down: torch.Tensor,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    top_n: int = 10
) -> List[Dict]:
    """Project neurons to vocabulary space"""
    ln = model.model.norm
    lm_head = model.lm_head.weight.detach()
    
    neuron_info = []
    
    with torch.cuda.amp.autocast(enabled=(W_down.dtype == torch.float16)):
        for neuron_idx in range(W_down.shape[0]):
            neuron_vec = W_down[neuron_idx, :]
            
            normed_vec = ln(neuron_vec.unsqueeze(0)).squeeze(0)
            logits = torch.matmul(lm_head, normed_vec)
            probs = torch.softmax(logits, dim=-1)
            
            top_probs, top_indices = torch.topk(probs, k=top_n)
            top_tokens = [tokenizer.decode([tid]) for tid in top_indices.cpu().tolist()]
            
            neuron_info.append({
                'neuron_idx': neuron_idx,
                'top_tokens': top_tokens,
                'top_probs': top_probs.cpu().tolist()
            })
    
    return neuron_info


def identify_task_relevant_neurons(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    target_answer: str,
    layer_idx: int,
    top_k: int = 50
) -> Dict:
    """Identify task-relevant neurons"""
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(DEVICE)
    
    keys, W_down = capture_mlp_activations(model, input_ids, layer_idx)
    neuron_vocab = project_neurons_to_vocab(W_down, model, tokenizer)
    
    answer_tokens = set(tokenizer.encode(str(target_answer), add_special_tokens=False))
    
    scores = []
    for i, info in enumerate(neuron_vocab):
        key_magnitude = keys[i].abs().item()
        
        top_token_ids = []
        for t in info['top_tokens']: 
            encoded = tokenizer.encode(t, add_special_tokens=False)
            if encoded:
                top_token_ids.append(encoded[0])
        
        overlap = len(set(top_token_ids) & answer_tokens) / max(len(answer_tokens), 1)
        
        score = key_magnitude * (1 + 10 * overlap)
        scores.append({
            'neuron_idx':  i,
            'key_value': key_magnitude,
            'answer_overlap': overlap,
            'score':  score,
            'top_tokens': info['top_tokens']
        })
    
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        'layer_idx': layer_idx,
        'top_neurons': scores[:top_k],
        'all_keys': keys.cpu().numpy()
    }


# Steering with Multiple Rollouts

class NeuronSteeringHook:
    """Neuron steering hook"""
    def __init__(self, neuron_indices: List[int], scale_factor: float):
        self.neuron_indices = neuron_indices
        self.scale_factor = scale_factor
    
    def __call__(self, module, input):
        x = input[0].clone()
        x[: , : , self.neuron_indices] *= self.scale_factor
        return (x,)


def steer_generation_with_rollouts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer_idx: int,
    neuron_indices: List[int],
    scale_factor: float,
    num_rollouts: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 512
) -> List[str]:
    """Execute multiple rollouts and return all generation results"""
    hook = NeuronSteeringHook(neuron_indices, scale_factor)
    handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(hook)
    
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).input_ids.to(model.device)
    
    generations = []
    
    with torch.no_grad():
        for _ in range(num_rollouts):
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generations.append(generated_text)
    
    handle.remove()
    
    return generations


# Enhanced Evaluation Metrics

def compute_metrics(
    baseline_outputs: List[str],
    steered_outputs: List[str],
    ground_truth: str
) -> Dict:
    """
    Compute comprehensive evaluation metrics (using auto_verify)
    """
    gt_list = [ground_truth] * len(baseline_outputs)
    
    # 1.Verify accuracy using auto_verify
    baseline_correct = verify_outputs(baseline_outputs, gt_list)
    steered_correct = verify_outputs(steered_outputs, gt_list)
    
    baseline_acc = np.mean(baseline_correct)
    steered_acc = np.mean(steered_correct)
    
    # 2.Calculate output consistency (consistency across multiple rollouts)
    baseline_unique = len(set(baseline_outputs))
    steered_unique = len(set(steered_outputs))
    
    baseline_consistency = 1 - (baseline_unique - 1) / max(len(baseline_outputs), 1)
    steered_consistency = 1 - (steered_unique - 1) / max(len(steered_outputs), 1)
    
    # 3.Calculate answer change rate (how many differ between baseline vs steered)
    answer_changed = sum(b != s for b, s in zip(baseline_outputs, steered_outputs)) / len(baseline_outputs)
    
    return {
        'baseline_accuracy': baseline_acc,
        'steered_accuracy': steered_acc,
        'accuracy_delta': steered_acc - baseline_acc,
        'baseline_consistency': baseline_consistency,
        'steered_consistency': steered_consistency,
        'answer_changed_rate': answer_changed,
        'num_rollouts': len(baseline_outputs)
    }


# Main Experiment Loop

def run_steering_experiment(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_cases: List[Dict],
    neuron_analysis: Dict,
    layer_idx: int,
    amplify_factors: List[float],
    suppress_factors: List[float],
    num_rollouts: int,
    temperature: float,
    top_p: float
) -> pd.DataFrame:
    """Run complete steering experiment (enhanced version)"""
    results = []
    
    task_neurons = [n['neuron_idx'] for n in neuron_analysis['top_neurons'][:10]]
    
    for case in tqdm(test_cases, desc=f"Steering L{layer_idx}"):
        prompt = case['prompt']
        ground_truth = case['answer']
        dataset = case['dataset']
        question_idx = case['question_index']
        
        # 1.Baseline (no intervention)
        baseline_outputs = steer_generation_with_rollouts(
            model, tokenizer, prompt, layer_idx, [], 1.0,
            num_rollouts, temperature, top_p
        )
        
        # 2.Intervene on key neurons
        for factor in amplify_factors + suppress_factors:
            steered_outputs = steer_generation_with_rollouts(
                model, tokenizer, prompt, layer_idx, task_neurons, factor,
                num_rollouts, temperature, top_p
            )
            
            metrics = compute_metrics(baseline_outputs, steered_outputs, ground_truth)
            
            results.append({
                'dataset': dataset,
                'question_index': question_idx,
                'layer':  layer_idx,
                'intervention':  'task_neurons',
                'scale_factor': factor,
                **metrics,
                'baseline_sample':  baseline_outputs[0][:100] + "...",
                'steered_sample': steered_outputs[0][:100] + "..."
            })
        
        # 3.Random control
        random_neurons = np.random.choice(
            range(neuron_analysis['all_keys'].shape[0]),
            size=10,
            replace=False
        ).tolist()
        
        for factor in [0.0, 2.0]: 
            random_steered = steer_generation_with_rollouts(
                model, tokenizer, prompt, layer_idx, random_neurons, factor,
                num_rollouts, temperature, top_p
            )
            
            metrics = compute_metrics(baseline_outputs, random_steered, ground_truth)
            
            results.append({
                'dataset': dataset,
                'question_index':  question_idx,
                'layer': layer_idx,
                'intervention': 'random_control',
                'scale_factor':  factor,
                **metrics,
                'baseline_sample': baseline_outputs[0][:100] + "...",
                'steered_sample': random_steered[0][:100] + "..."
            })
    
    return pd.DataFrame(results)


# Visualization

def visualize_steering_results(results_df: pd.DataFrame, output_dir: str):
    """Generate accuracy heatmap visualization"""
    task_df = results_df[results_df['intervention'] == 'task_neurons']
    datasets = task_df['dataset'].unique().tolist()
    dataset_str = ", ".join(datasets) if len(datasets) <= 3 else f"{len(datasets)} datasets"
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create pivot table - accuracy heatmap
    pivot = task_df.pivot_table(
        index='layer',
        columns='scale_factor',
        values='steered_accuracy',
        aggfunc='mean'
    )
    pivot = pivot.sort_index(axis=1)
    pivot_pct = pivot * 100
    
    # Draw heatmap
    sns.heatmap(
        pivot_pct,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=50,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Accuracy (%)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={'fontsize':  10}
    )
    
    ax.set_title(
        f'Steering Effect:  Accuracy (%) - Leakage Datasets Results\n({dataset_str})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Scale Factor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer', fontsize=12, fontweight='bold')
    
    # Add reference line (separate suppression and amplification)
    if 1.0 in pivot.columns:
        baseline_col_idx = list(pivot.columns).index(1.0)
        ax.axvline(x=baseline_col_idx + 1, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        
        # Add region annotations
        n_cols = len(pivot.columns)
        if baseline_col_idx > 0:
            ax.text(
                baseline_col_idx / 2, -1.5,
                'Suppression', ha='center', fontsize=11,
                color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.3)
            )
        if baseline_col_idx < n_cols - 1:
            ax.text(
                baseline_col_idx + 1 + (n_cols - baseline_col_idx - 1) / 2, -1.5,
                'Amplification', ha='center', fontsize=11,
                color='darkgreen', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3)
            )
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/steering_accuracy_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/steering_accuracy_heatmap.svg", format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy heatmap to {output_dir}/steering_accuracy_heatmap.png")


# Multi-GPU Worker

def gpu_worker(
    gpu_id: int,
    assigned_layers: List[int],
    test_cases: List[Dict],
    amplify_factors: List[float],
    suppress_factors: List[float],
    config: Dict,
    shared_results:  mp.Queue,
    is_main_gpu: bool = False,
    preloaded_model: Optional[AutoModelForCausalLM] = None,
    preloaded_tokenizer: Optional[AutoTokenizer] = None
):
    """Each GPU processes assigned layers"""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    if is_main_gpu:
        print(f"[GPU {gpu_id}] Using preloaded model (Main GPU)")
        model = preloaded_model
        tokenizer = preloaded_tokenizer
    else:
        print(f"[GPU {gpu_id}] Loading model on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_path"],
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            device_map={"": device},
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.eval()
        print(f"[GPU {gpu_id}] Model loaded.")
    
    print(f"[GPU {gpu_id}] Processing {len(assigned_layers)} layers...")
    
    # 1.Neuron Analysis phase
    layer_analysis = {}
    sample_case = test_cases[0]
    
    for layer_idx in assigned_layers:
        analysis = identify_task_relevant_neurons(
            model, tokenizer,
            sample_case['prompt'],
            sample_case['answer'],
            layer_idx,
            top_k=config["top_k_neurons"]
        )
        layer_analysis[layer_idx] = analysis
    
    shared_results.put(("neuron_analysis", gpu_id, layer_analysis))
    print(f"[GPU {gpu_id}] Neuron analysis complete. Starting steering experiments...")
    
    # 2.Steering Experiment phase
    steering_results = []
    
    for layer_idx in tqdm(assigned_layers, desc=f"GPU {gpu_id} Steering"):
        task_neurons = [n['neuron_idx'] for n in layer_analysis[layer_idx]['top_neurons'][:10]]
        
        for case in test_cases:
            prompt = case['prompt']
            ground_truth = case['answer']
            dataset = case['dataset']
            question_idx = case['question_index']
            
            # Baseline
            baseline_outputs = steer_generation_with_rollouts(
                model, tokenizer, prompt, layer_idx, [], 1.0,
                config["num_rollouts"], config["temperature"], config["top_p"]
            )
            
            # Task neurons steering
            for factor in amplify_factors + suppress_factors:
                steered_outputs = steer_generation_with_rollouts(
                    model, tokenizer, prompt, layer_idx, task_neurons, factor,
                    config["num_rollouts"], config["temperature"], config["top_p"]
                )
                
                metrics = compute_metrics(baseline_outputs, steered_outputs, ground_truth)
                
                steering_results.append({
                    'dataset': dataset,
                    'question_index': question_idx,
                    'layer': layer_idx,
                    'intervention':  'task_neurons',
                    'scale_factor': factor,
                    **metrics
                })
            
            # Random control
            random_neurons = np.random.choice(
                range(layer_analysis[layer_idx]['all_keys'].shape[0]),
                size=10,
                replace=False
            ).tolist()
            
            for factor in [0.0, 2.0]: 
                random_steered = steer_generation_with_rollouts(
                    model, tokenizer, prompt, layer_idx, random_neurons, factor,
                    config["num_rollouts"], config["temperature"], config["top_p"]
                )
                
                metrics = compute_metrics(baseline_outputs, random_steered, ground_truth)
                
                steering_results.append({
                    'dataset':  dataset,
                    'question_index': question_idx,
                    'layer': layer_idx,
                    'intervention': 'random_control',
                    'scale_factor': factor,
                    **metrics
                })
    
    shared_results.put(("steering_results", gpu_id, steering_results))
    print(f"[GPU {gpu_id}] All tasks completed!")


# Main Pipeline

def main():
    mp.set_start_method('spawn', force=True)
    
    print("=" * 70)
    print("MLP Neuron Steering Experiment (Multi-GPU Parallel)")
    print("=" * 70)
    
    # 1.Load model on main GPU
    print("\n[1/4] Loading model on main GPU...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_path"],
        device_map={"": "cuda:0"},
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded on GPU 0")
    
    # 2.Prepare test cases
    print("\n[2/4] Loading test cases...")
    test_cases = prepare_test_cases(
        CONFIG["correct_answers_file"],
        CONFIG["dataset_dir"],
        CONFIG["target_datasets"],
        CONFIG["samples_per_dataset"]
    )
    
    if not test_cases:
        print("Error: No test cases loaded.")
        return
    
    # 3.Multi-GPU parallel steering
    print(f"\n[3/4] Running multi-GPU steering experiments...")
    
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")
    
    all_layers = CONFIG["anchor_layers"] + CONFIG["control_layers"]
    
    # Assign layers to different GPUs
    layers_per_gpu = len(all_layers) // num_gpus
    gpu_assignments = []
    for i in range(num_gpus):
        start_idx = i * layers_per_gpu
        if i == num_gpus - 1:
            end_idx = len(all_layers)
        else:
            end_idx = (i + 1) * layers_per_gpu
        gpu_assignments.append(all_layers[start_idx:end_idx])
    
    print("\nGPU assignments:")
    for i, layers in enumerate(gpu_assignments):
        print(f"  GPU {i}:  Layers {layers}")
    
    # Create shared queue
    result_queue = mp.Queue()
    
    # Start worker processes
    processes = []
    for gpu_id, assigned_layers in enumerate(gpu_assignments):
        if len(assigned_layers) == 0:
            continue
        
        p = mp.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                assigned_layers,
                test_cases,
                CONFIG["amplify_factors"],
                CONFIG["suppress_factors"],
                CONFIG,
                result_queue,
                gpu_id == 0,
                model if gpu_id == 0 else None,
                tokenizer if gpu_id == 0 else None
            )
        )
        p.start()
        processes.append(p)
    
    # Collect results
    neuron_analysis_parts = {}
    steering_results_parts = []
    
    expected_messages = len(processes) * 2
    received_count = 0
    
    while received_count < expected_messages: 
        msg_type, gpu_id, data = result_queue.get()
        
        if msg_type == "neuron_analysis":
            neuron_analysis_parts.update(data)
            print(f"Received neuron analysis from GPU {gpu_id}")
        elif msg_type == "steering_results":
            steering_results_parts.extend(data)
            print(f"Received steering results from GPU {gpu_id}")
        
        received_count += 1
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("\nAll GPU workers completed!")
    
    # Merge results
    print("\nMerging and saving results...")
    
    # Save neuron analysis
    with open(f"{CONFIG['output_dir']}/neuron_analysis.json", "w") as f:
        json.dump(
            {k: {'top_neurons': v['top_neurons'][:20]} for k, v in neuron_analysis_parts.items()},
            f, indent=2
        )
    print(f"Neuron analysis saved to {CONFIG['output_dir']}/neuron_analysis.json")
    
    # Save steering results
    combined_results = pd.DataFrame(steering_results_parts)
    combined_results.to_csv(f"{CONFIG['output_dir']}/steering_results_enhanced.csv", index=False)
    print(f"Steering results saved to {CONFIG['output_dir']}/steering_results_enhanced.csv")
    
    # 4.Visualization
    print("\n[4/4] Generating visualization...")
    visualize_steering_results(combined_results, CONFIG["output_dir"])
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    task_data = combined_results[combined_results['intervention'] == 'task_neurons']
    
    for layer_idx in sorted(task_data['layer'].unique()):
        layer_data = task_data[task_data['layer'] == layer_idx]
        
        print(f"\nLayer {layer_idx}:")
        print(f"  Baseline Accuracy:       {layer_data['baseline_accuracy'].mean()*100:.1f}%")
        print(f"  Steered Accuracy:       {layer_data['steered_accuracy'].mean()*100:.1f}%")
        print(f"  Accuracy Delta:         {layer_data['accuracy_delta'].mean()*100:+.1f}%")
        print(f"  Answer Changed Rate:    {layer_data['answer_changed_rate'].mean()*100:.1f}%")
        print(f"  Output Consistency:     {layer_data['steered_consistency'].mean():.3f}")
    
    print("\n" + "=" * 70)
    print(f"All results saved to {CONFIG['output_dir']}/")
    print("=" * 70)


if __name__ == "__main__":
    main()