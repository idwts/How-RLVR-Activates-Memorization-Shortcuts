"""
Per-Sample Steering Trajectory Analysis
Goal: Visualize how steering affects individual samples across layers
      Compare baseline (alpha=1.0) vs suppression (alpha=0.5) vs amplification (alpha=3.0)
      
Modified:  Steering on L18, L19, L20 simultaneously
"""

import os
import sys
import json
import warnings
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ttrl.verifier.auto_verify import auto_verify


# Configuration
CONFIG = {
    "model_path": "../rethink_rlvr_reproduce-incorrect-qwen2.5_math_7b-lr5e-7-kl0.00-step150",
    "tokenizer_path": "../Qwen2.5/Qwen2.5-Math-7B",
    "correct_answers_file": "correct_answers_data.json",
    "dataset_dir": "../data",
    
    "target_dataset": "MATH-500",
    "target_question_indices": [213],
    
    "steering_layers": [18, 19, 20],
    "steering_factors": {
        "baseline": 1.0,
        "suppression": 0.2,
        "amplification": 3.0
    },
    
    "top_k_neurons": 50,
    "top_k_tokens": 10,
    "temperature": 0,
    "max_new_tokens": 2048,
    
    "output_dir": "per_sample_steering_trajectory_multi_layer",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu"
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

BENCHMARKS = {
    "MATH-500": {"path": "../data/MATH-TTT/test.json"}
}


# Data Loading

def load_correct_answers(file_path: str) -> List[Dict]:
    """Load correct answers from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("correct_answers", [])


def load_dataset_questions(dataset_name: str) -> List[Dict]:
    """Load dataset questions"""
    dataset_path = Path(BENCHMARKS[dataset_name]["path"])
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        if dataset_path.suffix == ".jsonl":
            questions = [json.loads(line) for line in f]
        else:
            data = json.load(f)
            questions = data if isinstance(data, list) else data.get("data", [])
    
    return questions


def get_target_sample(dataset:  str, question_idx: int) -> Optional[Dict]:
    """Get target sample"""
    correct_answers = load_correct_answers(CONFIG["correct_answers_file"])
    all_questions = load_dataset_questions(dataset)
    
    for item in correct_answers:
        if item.get("dataset") == dataset and item.get("question_index") == question_idx:
            if question_idx < len(all_questions):
                question_data = all_questions[question_idx]
                prompt = (question_data.get("problem") or 
                         question_data.get("question") or 
                         question_data.get("prompt"))
                answer = (question_data.get("solution") or 
                         question_data.get("answer"))
                
                if prompt and answer:
                    return {
                        "prompt": prompt,
                        "answer": answer,
                        "dataset": dataset,
                        "question_index": question_idx
                    }
    return None


# Answer Position Finding

def find_answer_token_position(
    generated_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    prompt_length: int
) -> Optional[Tuple[int, str, int]]:
    """Find position of first answer token after boxed"""
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    generated_text = tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=False)
    
    print(f"      Generated text preview: {generated_text[:300]}...")
    
    boxed_patterns = [
        r'\\boxed\{',
        r'\\boxed\s*\{',
        r'boxed\{',
    ]
    
    boxed_match = None
    answer_content = None
    
    for pattern in boxed_patterns:
        boxed_match = re.search(pattern, full_text)
        if boxed_match:
            break
    
    if not boxed_match:
        print(f"      Warning: No boxed found in generated text")
        answer_tag = '<answer>'
        if answer_tag in full_text:
            answer_start = full_text.find(answer_tag) + len(answer_tag)
            after_answer = full_text[answer_start: ].lstrip()
            answer_content = after_answer[: 50]
            print(f"      Using answer tag fallback: {answer_content}...")
        else:
            print(f"      Using last token as fallback")
            return None, None, None
    else:
        boxed_start = boxed_match.end()
        after_boxed = full_text[boxed_start:]
        
        brace_count = 1
        answer_end = 0
        for i, char in enumerate(after_boxed):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    answer_end = i
                    break
        
        if answer_end == 0:
            answer_content = after_boxed.strip()
        else:
            answer_content = after_boxed[:answer_end].strip()
        
        print(f"      Found answer in boxed:  '{answer_content}'")
    
    if not answer_content:
        print(f"      Warning: Empty answer content")
        return None, None, None
    
    answer_preview = answer_content[:20]
    answer_tokens = tokenizer(answer_content, add_special_tokens=False)['input_ids']
    if not answer_tokens:
        print(f"      Warning: Answer tokenization failed")
        return None, None, None
    
    full_token_list = generated_ids[0].cpu().tolist()
    first_answer_token_id = answer_tokens[0]
    
    for i in range(prompt_length, len(full_token_list)):
        if full_token_list[i] == first_answer_token_id: 
            match_len = min(3, len(answer_tokens))
            if full_token_list[i:i+match_len] == answer_tokens[:match_len]:
                print(f"      Found answer token at position {i}")
                print(f"      Answer token: '{tokenizer.decode([first_answer_token_id])}'")
                print(f"      Answer preview: '{answer_preview}'")
                return i, answer_content, first_answer_token_id
    
    print(f"      Warning: Could not precisely locate answer token")
    fallback_pos = len(full_token_list) - min(5, len(answer_tokens))
    print(f"      Using fallback position: {fallback_pos}")
    return fallback_pos, answer_content, first_answer_token_id


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
        keys_last = keys[: , -1, :].mean(dim=0)
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
    
    with torch.amp.autocast('cuda', enabled=(W_down.dtype == torch.float16)):
        for neuron_idx in range(W_down.shape[0]):
            neuron_vec = W_down[neuron_idx, :]
            
            normed_vec = ln(neuron_vec.unsqueeze(0)).squeeze(0)
            logits = torch.matmul(lm_head, normed_vec)
            probs = torch.softmax(logits, dim=-1)
            
            top_probs, top_indices = torch.topk(probs, k=top_n)
            top_tokens = [tokenizer.decode([tid]) for tid in top_indices.cpu().tolist()]
            
            neuron_info.append({
                'neuron_idx':  neuron_idx,
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
    top_k:  int = 10,
    device: str = "cuda"
) -> List[int]:
    """Identify task-relevant neurons, return neuron indices"""
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(device)
    
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
        scores.append({'neuron_idx': i, 'score': score})
    
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    return [s['neuron_idx'] for s in scores[:top_k]]


# Multi-Layer Steering Hook

class NeuronSteeringHook: 
    """Single-layer neuron steering hook"""
    def __init__(self, neuron_indices: List[int], scale_factor: float):
        self.neuron_indices = neuron_indices
        self.scale_factor = scale_factor
    
    def __call__(self, module, input):
        x = input[0].clone()
        x[: , : , self.neuron_indices] *= self.scale_factor
        return (x,)


# Layer-wise Logit Lens Analysis

def get_layerwise_predictions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    steering_layers: List[int],
    layer_neuron_map: Dict[int, List[int]],
    scale_factor: float,
    top_k: int = 10,
    device: str = "cuda"
) -> Dict:
    """
    Get predictions for each layer's MLP output (for answer position token)
    
    Modified:  Simultaneous steering on multiple layers
    """
    # Step 1: First use steering to generate complete text and find answer position
    handles = []
    
    # Register steering hook for each layer
    if scale_factor != 1.0:
        for layer_idx in steering_layers: 
            if layer_idx in layer_neuron_map and layer_neuron_map[layer_idx]:
                hook = NeuronSteeringHook(layer_neuron_map[layer_idx], scale_factor)
                handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(hook)
                handles.append(handle)
    
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).input_ids.to(device)
    
    prompt_length = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=CONFIG["max_new_tokens"],
            do_sample=(CONFIG["temperature"] > 0),
            temperature=CONFIG["temperature"] if CONFIG["temperature"] > 0 else None,
            top_p=0.95 if CONFIG["temperature"] > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=False
        )
    
    generated_ids = outputs.sequences
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Remove all hooks
    for handle in handles: 
        handle.remove()
    handles = []
    
    # Step 2: Find answer position in generated text
    answer_position, answer_content, answer_token_id = find_answer_token_position(
        generated_ids, tokenizer, prompt_length
    )
    
    if answer_position is None:
        print(f"      Warning: Failed to find answer, using last token as fallback")
        answer_position = generated_ids.shape[1] - 1
        answer_content = "N/A"
        answer_token_id = generated_ids[0, answer_position].item()
    
    # Verify:  Confirm this position's token is what we're looking for
    actual_token_at_position = generated_ids[0, answer_position].item()
    actual_token_str = tokenizer.decode([actual_token_at_position])
    print(f"      Analyzing position {answer_position}: token='{actual_token_str}' (ID={actual_token_at_position})")
    
    # Get sequence up to answer position (not including answer token)
    context_ids = generated_ids[: , :answer_position]
    print(f"      Context sequence length: {context_ids.shape[1]} (analyzing prediction at position {answer_position})")
    
    # Step 3: Forward pass on context sequence, collect hidden states
    # Re-register steering hooks
    if scale_factor != 1.0:
        for layer_idx in steering_layers:
            if layer_idx in layer_neuron_map and layer_neuron_map[layer_idx]: 
                hook = NeuronSteeringHook(layer_neuron_map[layer_idx], scale_factor)
                handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(hook)
                handles.append(handle)
    
    with torch.no_grad():
        full_outputs = model(
            context_ids,
            output_hidden_states=True,
            return_dict=True
        )
    
    # Remove all hooks
    for handle in handles:
        handle.remove()
    
    # Step 4: Collect predictions at last position (predicting answer token) for each layer
    layer_predictions = []
    n_layers = model.config.num_hidden_layers
    
    ln = model.model.norm
    lm_head = model.lm_head
    
    last_pos = context_ids.shape[1] - 1
    
    for layer_idx in range(n_layers):
        hidden_state = full_outputs.hidden_states[layer_idx + 1]
        hidden_at_pos = hidden_state[0, last_pos, :]
        
        normed = ln(hidden_at_pos)
        logits = lm_head(normed)
        probs = torch.softmax(logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
        
        top_k_tokens = []
        for prob, idx in zip(top_k_probs.detach().cpu().numpy(), top_k_indices.detach().cpu().numpy()):
            token_str = tokenizer.decode([idx])
            top_k_tokens.append({
                'token': token_str,
                'token_id': int(idx),
                'probability': float(prob)
            })
        
        layer_predictions.append({
            'layer':  layer_idx,
            'top_k_tokens': top_k_tokens,
            'top_1_token':  top_k_tokens[0]['token'],
            'top_1_prob': top_k_tokens[0]['probability']
        })
    
    return {
        'generated_text': generated_text,
        'answer_position': answer_position,
        'answer_content': answer_content,
        'answer_token_id': answer_token_id,
        'answer_token_str': actual_token_str,
        'layer_predictions': layer_predictions,
        'steering_layers': steering_layers,
        'scale_factor': scale_factor
    }


# Visualization

def visualize_steering_trajectory(
    baseline_results: Dict,
    suppression_results: Dict,
    amplification_results: Dict,
    sample_info: Dict,
    output_dir: str
):
    """Visualize steering trajectory"""
    dataset = sample_info['dataset']
    question_idx = sample_info['question_index']
    ground_truth = sample_info['answer']
    
    sample_dir = os.path.join(output_dir, f"{dataset}_q{question_idx}")
    os.makedirs(sample_dir, exist_ok=True)
    
    n_layers = len(baseline_results['layer_predictions'])
    
    steering_layers_str = ", ".join([f"L{l}" for l in baseline_results['steering_layers']])
    
    # Modified: Order as suppression -> baseline -> amplification
    conditions = [
        ("Suppression (alpha=0.2)", suppression_results, 'coral'),
        ("Baseline (alpha=1.0)", baseline_results, 'steelblue'),
        ("Amplification (alpha=3.0)", amplification_results, 'green')
    ]
    
    # Figure 1: Logit Lens - Top-1 Token Probability Across Layers
    fig, axes = plt.subplots(1, 3, figsize=(24, max(10, n_layers * 0.4)))
    
    for ax, (title, results, color) in zip(axes, conditions):
        layers = [p['layer'] for p in results['layer_predictions']]
        tokens = [p['top_1_token'] for p in results['layer_predictions']]
        probs = [p['top_1_prob'] for p in results['layer_predictions']]
        
        answer_token = results['answer_token_str']
        colors_bar = ['darkgreen' if token == answer_token else color for token in tokens]
        
        bars = ax.barh(layers, probs, color=colors_bar, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(layers)
        ax.set_yticklabels([f'L{l}' for l in layers], fontsize=10)
        ax.set_xlabel('Top-1 Probability', fontsize=24, fontweight='bold')
        ax.set_ylabel('Layer', fontsize=24, fontweight='bold')
        ax.set_title(title, fontsize=30, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        ax.set_xlim([0, 1])
        
        for i, (layer, token, prob) in enumerate(zip(layers, tokens, probs)):
            token_display = token[: 10] + '...' if len(token) > 10 else token
            ax.text(prob + 0.02, i, f'{token_display}', va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle(
        f'Steering Trajectory:  Layer-wise Predictions for Answer Token\n'
        f'Steering on {steering_layers_str} | {dataset} Q{question_idx} | Ground Truth: "{ground_truth}"',
        fontsize=26, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, '1_logit_lens_comparison.svg'),
                format='svg', dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  Saved Logit Lens comparison")
    
    # Figure 2: Top-10 Token Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(24, max(10, n_layers * 0.5)))
    
    for ax, (title, results, _) in zip(axes, conditions):
        probs_matrix = []
        tokens_matrix = []
        
        for layer_pred in results['layer_predictions']: 
            layer_probs = [t['probability'] for t in layer_pred['top_k_tokens']]
            layer_tokens = [t['token'] for t in layer_pred['top_k_tokens']]
            probs_matrix.append(layer_probs)
            tokens_matrix.append(layer_tokens)
        
        probs_array = np.array(probs_matrix)
        
        im = ax.imshow(probs_array, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f'L{i}' for i in range(n_layers)], fontsize=9)
        ax.set_xticks(range(CONFIG["top_k_tokens"]))
        ax.set_xticklabels([f'Top-{i+1}' for i in range(CONFIG["top_k_tokens"])], fontsize=10)
        
        for i in range(n_layers):
            for j in range(CONFIG["top_k_tokens"]):
                token_name = tokens_matrix[i][j]
                token_display = token_name[:6] + '..' if len(token_name) > 6 else token_name
                prob_value = probs_array[i, j]
                
                text_content = f'{token_display}\n{prob_value:.2f}'
                ax.text(j, i, text_content, ha="center", va="center",
                       color="black", fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Top-K Position', fontsize=24, fontweight='bold')
        ax.set_ylabel('Layer', fontsize=24, fontweight='bold')
        ax.set_title(title, fontsize=30, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', fontsize=24, fontweight='bold')
    
    plt.suptitle(
        f'Top-10 Token Distributions for Answer Token Position\n'
        f'Steering on {steering_layers_str} | {dataset} Q{question_idx}',
        fontsize=26, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, '2_top10_heatmap_comparison.svg'),
                format='svg', dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  Saved Top-10 heatmap comparison")
    
    # Figure 3: Final Output Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    conditions_short = ['Suppression', 'Baseline', 'Amplification']
    final_tokens = [
        suppression_results['layer_predictions'][-1]['top_1_token'],
        baseline_results['layer_predictions'][-1]['top_1_token'],
        amplification_results['layer_predictions'][-1]['top_1_token']
    ]
    final_probs = [
        suppression_results['layer_predictions'][-1]['top_1_prob'],
        baseline_results['layer_predictions'][-1]['top_1_prob'],
        amplification_results['layer_predictions'][-1]['top_1_prob']
    ]
    
    colors = ['coral', 'steelblue', 'green']
    bars = ax.bar(conditions_short, final_probs, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Final Probability', fontsize=14, fontweight='bold')
    ax.set_title(f'Final Layer Prediction for Answer Token\nSteering on {steering_layers_str} | {dataset} Q{question_idx}',
                fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    for bar, token, prob in zip(bars, final_tokens, final_probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{token}\n({prob:.3f})',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Reference')
    ax.text(0.5, 0.55,
            f'Ground Truth Answer: "{ground_truth}"\n'
            f'Answer Token: "{baseline_results["answer_token_str"]}"',
            ha='left', fontsize=11,
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, '3_final_output_comparison.svg'),
                format='svg', dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  Saved final output comparison")
    
    return sample_dir


# Answer Verification

def verify_single_answer(response:  str, ground_truth: str) -> bool:
    """Verify single answer"""
    class DummyOutput:
        def __init__(self, text):
            self.text = text
    
    class DummyCompletionOutput:
        def __init__(self, text):
            self.outputs = [DummyOutput(text)]
    
    wrapped = [DummyCompletionOutput(response)]
    correctness = auto_verify("math", 1, wrapped, [ground_truth])
    return correctness[0]


# Main Analysis

def analyze_single_sample(
    sample:  Dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str
) -> Dict:
    """Analyze steering trajectory for single sample"""
    print(f"\n{'='*60}")
    print(f"Analyzing {sample['dataset']} Q{sample['question_index']}")
    print(f"Ground Truth: {sample['answer']}")
    print(f"{'='*60}")
    
    # 1.Identify task-relevant neurons for each layer
    print(f"\n[1/4] Identifying task-relevant neurons for L{CONFIG['steering_layers']}...")
    layer_neuron_map = {}
    
    for layer_idx in CONFIG['steering_layers']:
        neuron_indices = identify_task_relevant_neurons(
            model, tokenizer,
            sample['prompt'],
            sample['answer'],
            layer_idx,
            top_k=CONFIG['top_k_neurons'],
            device=device
        )
        layer_neuron_map[layer_idx] = neuron_indices
        print(f"  L{layer_idx}: top-{len(neuron_indices)} neurons:  {neuron_indices[: 5]}...")
    
    # 2.Analyze three steering conditions
    results = {}
    
    for condition_name, scale_factor in CONFIG['steering_factors'].items():
        print(f"\n[2/4] Analyzing {condition_name} (alpha={scale_factor})...")
        
        results[condition_name] = get_layerwise_predictions(
            model, tokenizer,
            sample['prompt'],
            CONFIG['steering_layers'],
            layer_neuron_map,
            scale_factor,
            top_k=CONFIG['top_k_tokens'],
            device=device
        )
        
        answer_token = results[condition_name]['answer_token_str']
        final_token = results[condition_name]['layer_predictions'][-1]['top_1_token']
        final_prob = results[condition_name]['layer_predictions'][-1]['top_1_prob']
        
        is_correct = verify_single_answer(
            results[condition_name]['generated_text'],
            sample['answer']
        )
        
        results[condition_name]['is_correct'] = is_correct
        
        print(f"  Answer Token: '{answer_token}'")
        print(f"  Final Layer Prediction: '{final_token}' (prob={final_prob:.4f})")
        print(f"  Full Answer Correct: {is_correct}")
    
    # 3.Visualization
    print(f"\n[3/4] Generating visualizations...")
    output_dir = visualize_steering_trajectory(
        results['baseline'],
        results['suppression'],
        results['amplification'],
        sample,
        CONFIG['output_dir']
    )
    
    # 4.Save results
    print(f"\n[4/4] Saving results...")
    
    summary = {
        'sample_info': sample,
        'layer_neuron_map': {str(k): v for k, v in layer_neuron_map.items()},
        'steering_layers':  CONFIG['steering_layers'],
        'conditions': {}
    }
    
    for condition_name in CONFIG['steering_factors'].keys():
        summary['conditions'][condition_name] = {
            'scale_factor': results[condition_name]['scale_factor'],
            'answer_position': results[condition_name]['answer_position'],
            'answer_token': results[condition_name]['answer_token_str'],
            'answer_token_id': results[condition_name]['answer_token_id'],
            'answer_content': results[condition_name]['answer_content'],
            'final_layer_prediction': results[condition_name]['layer_predictions'][-1]['top_1_token'],
            'final_layer_probability': results[condition_name]['layer_predictions'][-1]['top_1_prob'],
            'is_correct': results[condition_name]['is_correct'],
            'generated_text_preview': results[condition_name]['generated_text'][:200] + '...'
        }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved summary to {summary_file}")
    
    return {
        'sample':  sample,
        'results': results,
        'output_dir': output_dir
    }


def main():
    print("=" * 70)
    print("Per-Sample Steering Trajectory Analysis")
    print(f"Steering on Layers: {CONFIG['steering_layers']}")
    print("Analyzing Answer Token Predictions")
    print("=" * 70)
    
    # 1.Load model
    print("\n[Setup] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_path"], trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_path"],
        device_map={"": CONFIG["device"]},
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded")
    
    # 2.Load target samples
    print("\n[Setup] Loading target samples...")
    samples = []
    for question_idx in CONFIG["target_question_indices"]:
        sample = get_target_sample(CONFIG["target_dataset"], question_idx)
        if sample: 
            samples.append(sample)
            print(f"  Loaded Q{question_idx}")
        else:
            print(f"  Failed to load Q{question_idx}")
    
    if not samples:
        print("Error: No samples loaded!")
        return
    
    # 3.Analyze each sample
    all_results = []
    
    for sample in samples:
        try:
            result = analyze_single_sample(sample, model, tokenizer, CONFIG["device"])
            all_results.append(result)
            
            torch.cuda.empty_cache()
            
        except Exception as e: 
            print(f"\nError analyzing {sample['dataset']} Q{sample['question_index']}: {e}")
            import traceback
            traceback.print_exc()
    
    # 4.Generate overall report
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for result in all_results:
        sample = result['sample']
        print(f"\n{sample['dataset']} Q{sample['question_index']}:")
        print(f"  Ground Truth: {sample['answer']}")
        
        for condition_name in CONFIG['steering_factors'].keys():
            condition_result = result['results'][condition_name]
            answer_token = condition_result['answer_token_str']
            final_token = condition_result['layer_predictions'][-1]['top_1_token']
            is_correct = condition_result['is_correct']
            
            print(f"  {condition_name.capitalize():15s}:  Answer Token='{answer_token}' | "
                  f"L27 predicts '{final_token}' | {'Correct' if is_correct else 'Wrong'}")
    
    print(f"\n{'='*70}")
    print(f"All results saved to {CONFIG['output_dir']}/")
    print(f"Steering applied on layers: {CONFIG['steering_layers']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()