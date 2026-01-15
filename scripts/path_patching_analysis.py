"""
Path Patching / Causal Tracing Analysis

Experimental Design:
1.Run before_rlvr model (produces incorrect output) and save activations
2.Run after_rlvr model (produces correct output) and save activations
3.Replace activations layer by layer from before model with after model activations
4.Observe which layer replacement causes output to become correct
"""

import json
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ttrl.verifier.auto_verify import auto_verify


# Configuration parameters
CONFIG = {
    'correct_answers_file': 'correct_answers_data.json',
    
    'model_checkpoints': {
        'before':  {
            'name': 'before_rlvr',
            'path': '/data/liwenxi/rlvr/code/Meta-Llama-3.1-8B',
        },
        'after': {
            'name': 'after_rlvr',
            'path': '/data/liwenxi/rlvr/code/outputs/DeepScaleR_mv_labeled_llama3.1_8b_instruct_incorrect-meta-llama/Llama-3.1-8B/1218/DeepScaleR_mv_labeled_llama3.1_8b_instruct_incorrect-RLVR-math/ckpt/_actor/step150_fp32',
        }
    },
    
    'output_dir': 'llama_path_patching_analysis',
    'output_file': 'patching_results.json',
    
    'max_new_tokens': 512,
    'temperature': 0.6,
    'num_rollouts': 4,
    
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',

    'max_per_dataset':  999,
    'target_datasets': ['MATH-500', 'MinervaMath', 'LiveMathBench'],
}


class ActivationCache:
    """Store activations for each layer of the model"""
    def __init__(self):
        self.ffn_inputs = []
        self.ffn_outputs = []
        self.attn_inputs = []
        self.attn_outputs = []
        
    def clear(self):
        self.ffn_inputs.clear()
        self.ffn_outputs.clear()
        self.attn_inputs.clear()
        self.attn_outputs.clear()


def load_correct_answers(file_path: str) -> List[Dict]:
    """Load correct_answers_data.json and return only questions from specified three datasets (complete list)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Compatible with different json structures
    if isinstance(data, dict):
        # Common keys
        for key in ['correct_answers', 'results', 'questions', 'data']:
            if key in data: 
                items = data[key]
                break
        else:
            # If it's directly a dictionary list
            # Try to find the first value that is a list
            found = None
            for v in data.values():
                if isinstance(v, list):
                    found = v
                    break
            items = found if found is not None else []
    elif isinstance(data, list):
        items = data
    else: 
        items = []
    
    target_sets = set(CONFIG['target_datasets'])
    filtered = []
    for it in items:
        ds = it.get('dataset') or it.get('bench') or it.get('dataset_name')
        if ds in target_sets:
            # Normalize fields:  prompt, question_index, label/ground_truth/answer
            prompt = it.get('prompt') or it.get('question') or it.get('problem') or it.get('text', '')
            gt = it.get('ground_truth') or it.get('answer') or it.get('label') or it.get('solution', '')
            qidx = it.get('question_index', it.get('index', it.get('id')))
            filtered.append({
                'dataset': ds,
                'question_index': qidx,
                'prompt': prompt,
                'label': gt
            })
    
    print(f"Loaded {len(filtered)} questions from {file_path} (filtered to target datasets)")
    return filtered


def capture_activations(
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask:  torch.Tensor,
        cache: ActivationCache,
        device: str
) -> torch.Tensor:
    """
    Run model and capture activations for all layers
    
    Returns:
        output_ids: Generated token IDs
    """
    cache.clear()
    
    # Ensure input tensors are on the correct device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Register hooks to capture activations
    hooks = []
    
    def create_attn_pre_hook(layer_idx):
        def hook(module, input):
            if isinstance(input, tuple) and len(input) > 0:
                hidden_states = input[0].detach().clone()
                if layer_idx == len(cache.attn_inputs):
                    # Ensure cached tensors are on the correct device
                    cache.attn_inputs.append(hidden_states.to(device))
        return hook
    
    def create_attn_post_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                attn_output = output[0].detach().clone()
            else:
                attn_output = output.detach().clone()
            
            if layer_idx == len(cache.attn_outputs):
                # Ensure cached tensors are on the correct device
                cache.attn_outputs.append(attn_output.to(device))
        return hook
    
    def create_mlp_pre_hook(layer_idx):
        def hook(module, input):
            if isinstance(input, tuple) and len(input) > 0:
                hidden_states = input[0].detach().clone()
                if layer_idx == len(cache.ffn_inputs):
                    # Ensure cached tensors are on the correct device
                    cache.ffn_inputs.append(hidden_states.to(device))
        return hook
    
    def create_mlp_post_hook(layer_idx):
        def hook(module, input, output):
            mlp_output = output.detach().clone()
            if layer_idx == len(cache.ffn_outputs):
                # Ensure cached tensors are on the correct device
                cache.ffn_outputs.append(mlp_output.to(device))
        return hook
    
    # Register hooks
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_pre_hook(create_attn_pre_hook(i)))
        hooks.append(layer.self_attn.register_forward_hook(create_attn_post_hook(i)))
        hooks.append(layer.mlp.register_forward_pre_hook(create_mlp_pre_hook(i)))
        hooks.append(layer.mlp.register_forward_hook(create_mlp_post_hook(i)))
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=CONFIG['max_new_tokens'],
            do_sample=True,
            temperature=CONFIG['temperature'],
            pad_token_id=model.config.eos_token_id,
        )
    
    # Remove hooks
    for hook in hooks: 
        hook.remove()
    
    return outputs


def generate_with_patching(
    model: AutoModelForCausalLM,
    tokenizer:  AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_cache: ActivationCache,
    patch_layer:  int,
    patch_component: str,
    device: str
) -> List[str]:
    """
    Run model, replace activations at specified layer, and generate multiple times
    
    Args:
        model: Model to run
        tokenizer: Tokenizer
        input_ids: Input token IDs
        attention_mask:  Attention mask
        patch_cache:  Activation cache to inject
        patch_layer: Layer index to replace
        patch_component: Component to replace ('ffn' or 'attn')
        device: Device
    
    Returns:
        List[str]: Multiple generated answer texts
    """
    generated_texts = []
    
    # Ensure input tensors are on the correct device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Ensure all tensors in patch_cache are on the correct device
    patch_cache.ffn_inputs = [t.to(device) for t in patch_cache.ffn_inputs]
    patch_cache.ffn_outputs = [t.to(device) for t in patch_cache.ffn_outputs]
    patch_cache.attn_inputs = [t.to(device) for t in patch_cache.attn_inputs]
    patch_cache.attn_outputs = [t.to(device) for t in patch_cache.attn_outputs]
    
    for rollout in range(CONFIG['num_rollouts']):
        # Register patching hook
        patch_applied = [False]  # Wrapped in list to modify in closure
        
        def create_patch_hook(layer_idx, component):
            def hook(module, input, output):
                if layer_idx == patch_layer and not patch_applied[0]:
                    patch_applied[0] = True
                    
                    if component == 'attn':
                        # Replace attention output
                        patched_output = patch_cache.attn_outputs[layer_idx]
                        if isinstance(output, tuple):
                            # Ensure output is also on the correct device
                            patched_output = patched_output.to(output[0].device)
                            return (patched_output,) + output[1:]
                        else:
                            # Ensure output is also on the correct device
                            return patched_output.to(output.device)
                    
                    elif component == 'ffn':
                        # Replace FFN output
                        patched_output = patch_cache.ffn_outputs[layer_idx]
                        # Ensure output is also on the correct device
                        return patched_output.to(output.device)
                
                return output
            
            return hook
        
        # Register hook
        target_layer = model.model.layers[patch_layer]
        if patch_component == 'attn': 
            hook = target_layer.self_attn.register_forward_hook(
                create_patch_hook(patch_layer, 'attn')
            )
        else:  # ffn
            hook = target_layer.mlp.register_forward_hook(
                create_patch_hook(patch_layer, 'ffn')
            )
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=CONFIG['max_new_tokens'],
                do_sample=True,
                temperature=CONFIG['temperature'],
                pad_token_id=model.config.eos_token_id,
            )
        
        # Remove hook
        hook.remove()
        
        # Decode
        generated_ids = outputs[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts


def verify_generations(generations: List[str], label: str) -> Tuple[float, bool]:
    """
    Verify if generated answers are correct
    
    Returns:
        (accuracy, any_correct): Accuracy and whether at least one is correct
    """
    # Construct fake output object for verification
    fake_outputs = []
    for text in generations:
        fake_output = type('obj', (), {
            'outputs': [type('obj', (), {'text': text})()]
        })()
        fake_outputs.append(fake_output)
    
    # Verify
    results = auto_verify('math', 1, fake_outputs, [label])
    
    accuracy = np.mean(results)
    any_correct = any(results)
    
    return accuracy, any_correct


def analyze_question_patching(
        question_data: Dict,
        model_before: AutoModelForCausalLM,
        model_after: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str
) -> Dict:
    """
    Perform path patching analysis on a single question
    """
    prompt = question_data['prompt']
    label = question_data['label']
    
    print(f"    Tokenizing...")
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # 1.Run before model and save activations
    print(f"    Running before_rlvr (clean run)...")
    cache_before = ActivationCache()
    outputs_before_ids = capture_activations(
        model_before, input_ids, attention_mask, cache_before, device
    )
    
    # Decode before output
    generated_before = tokenizer.decode(
        outputs_before_ids[0, input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    # No longer verify if before is wrong, continue execution
    
    # 2.Run after model and save activations
    print(f"    Running after_rlvr (patching source)...")
    cache_after = ActivationCache()
    outputs_after_ids = capture_activations(
        model_after, input_ids, attention_mask, cache_after, device
    )
    
    # Decode after output
    generated_after = tokenizer.decode(
        outputs_after_ids[0, input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    # No longer verify if after is correct, continue execution
    
    # Removed logic "if doesn't meet 'before wrong, after correct' condition, skip"
    
    # 3.Perform patching experiment for each layer
    num_layers = len(cache_before.ffn_outputs)
    print(f"    Patching {num_layers} layers...")
    
    patching_results = {
        'ffn':  [],
        'attn': [],
    }
    
    for layer_idx in tqdm(range(num_layers), desc="    Patching layers"):
        # Patch FFN
        try:
            ffn_generations = generate_with_patching(
                model_before, tokenizer, input_ids, attention_mask,
                cache_after, layer_idx, 'ffn', device
            )
            ffn_accuracy, ffn_any_correct = verify_generations(ffn_generations, label)
            
            patching_results['ffn'].append({
                'layer':  layer_idx,
                'accuracy': ffn_accuracy,
                'any_correct': ffn_any_correct,
                'num_correct': int(ffn_accuracy * CONFIG['num_rollouts'])
            })
        except Exception as e:
            print(f"\n      FFN patching failed at layer {layer_idx}: {e}")
            patching_results['ffn'].append({
                'layer': layer_idx,
                'accuracy': 0.0,
                'any_correct': False,
                'num_correct': 0
            })
        
        # Patch Attention
        try:
            attn_generations = generate_with_patching(
                model_before, tokenizer, input_ids, attention_mask,
                cache_after, layer_idx, 'attn', device
            )
            attn_accuracy, attn_any_correct = verify_generations(attn_generations, label)
            
            patching_results['attn'].append({
                'layer': layer_idx,
                'accuracy': attn_accuracy,
                'any_correct':  attn_any_correct,
                'num_correct': int(attn_accuracy * CONFIG['num_rollouts'])
            })
        except Exception as e:
            print(f"\n      Attention patching failed at layer {layer_idx}: {e}")
            patching_results['attn'].append({
                'layer': layer_idx,
                'accuracy': 0.0,
                'any_correct': False,
                'num_correct': 0
            })
        
        torch.cuda.empty_cache()
    
    return {
        'dataset': question_data['dataset'],
        'question_index': question_data['question_index'],
        'prompt':  prompt,
        'label': label,
        'before_output': generated_before,
        'after_output': generated_after,
        'patching_results': patching_results,
        'num_layers': num_layers
    }


def visualize_patching_results(results: List[Dict], output_dir: str):
    """
    Visualize patching results
    - Save overall plots (original)
    - Additionally save separate visualizations for MATH-500, MinervaMath, LiveMathBench
    """
    if len(results) == 0:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by dataset
    by_dataset = defaultdict(list)
    for r in results:
        ds = r.get('dataset', 'unknown')
        by_dataset[ds].append(r)
    
    # First draw overall (original logic)
    _plot_recovery(results, os.path.join(output_dir, 'patching_recovery_analysis_all.png'))
    
    # Draw separately for each target dataset (only draw existing datasets)
    for ds in CONFIG['target_datasets']:
        subset = by_dataset.get(ds, [])
        if not subset:
            print(f"No results for dataset {ds}, skip visualization")
            continue
        out_path = os.path.join(output_dir, f'patching_recovery_analysis_{ds}.png')
        _plot_recovery(subset, out_path, title=f'Path Patching:  {ds}')
    
    print(f"\nVisualizations saved to {output_dir}")
    
    # Print top layer information overall and top 5 layers for each dataset (brief)
    def _top_layers_for(results_list):
        if not results_list: 
            return []
        num_layers = results_list[0]['num_layers']
        agg = np.zeros(num_layers)
        for r in results_list:
            for layer_data in r['patching_results']['ffn']:
                agg[layer_data['layer']] += layer_data['accuracy']
        agg /= max(1, len(results_list))
        return np.argsort(agg)[: :-1][:5].tolist()
    
    print("Top-5 overall FFN layers:", _top_layers_for(results))
    for ds, subset in by_dataset.items():
        print(f"Top-5 FFN layers for {ds}:", _top_layers_for(subset))


def _plot_recovery(results: List[Dict], out_path: str, title: str = 'Path Patching:  Accuracy Recovery by Layer'):
    """Draw recovery rate curve for a single set (overall or single dataset), remove heatmap and adjust font size"""
    num_layers = results[0]['num_layers']
    
    ffn_recovery_rates = np.zeros(num_layers)
    attn_recovery_rates = np.zeros(num_layers)
    
    for result in results:
        for layer_data in result['patching_results']['ffn']:
            ffn_recovery_rates[layer_data['layer']] += layer_data['accuracy']
        for layer_data in result['patching_results']['attn']:
            attn_recovery_rates[layer_data['layer']] += layer_data['accuracy']
    
    ffn_recovery_rates /= len(results)
    attn_recovery_rates /= len(results)
    
    # Create larger figure, only include line plot
    plt.figure(figsize=(16, 10))
    
    # Draw recovery rate curve, increase font size
    plt.plot(range(num_layers), ffn_recovery_rates, marker='o', label='FFN Patching',
             linewidth=3, markersize=8, color='#FF6B6B')
    plt.plot(range(num_layers), attn_recovery_rates, marker='s', label='Attention Patching',
             linewidth=3, markersize=8, color='#4ECDC4')
    plt.xlabel('Layer Index', fontsize=22, fontweight='bold')
    plt.ylabel('Accuracy Recovery Rate', fontsize=22, fontweight='bold')
    plt.title(title, fontsize=25, fontweight='bold')
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    
    # Adjust tick label font size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Adjust legend position to avoid blocking
    plt.legend(fontsize=18, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function"""
    print("=" * 60)
    print("Path Patching / Causal Tracing Analysis")
    print("=" * 60)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Load data
    print(f"\nLoading:  {CONFIG['correct_answers_file']}")
    answers = load_correct_answers(CONFIG['correct_answers_file'])
    print(f"Loaded {len(answers)} questions")
    
    questions = answers
    print(f"Analyzing {len(questions)} questions")
    
    # Load models
    print(f"\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model_checkpoints']['before']['path'],
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  Before (float32)")
    model_before = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_checkpoints']['before']['path'],
        device_map='auto',
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model_before.eval()
    
    print(f"  After (float32)")
    model_after = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_checkpoints']['after']['path'],
        device_map='auto',
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model_after.eval()
    
    device = CONFIG['device']
    results = []
    
    # New:  analyze at most max_per_dataset questions per dataset
    dataset_counter = {}
    max_per_dataset = CONFIG['max_per_dataset']
    
    # Analyze each question
    print(f"\nAnalyzing...")
    for idx, q in enumerate(questions):
        ds = q.get('dataset', 'unknown')
        dataset_counter.setdefault(ds, 0)
        if dataset_counter[ds] >= max_per_dataset:
            continue  # Already analyzed enough, skip
        
        print(f"\nQuestion {idx+1}/{len(questions)}: {ds} #{q['question_index']}")
        
        try:
            result = analyze_question_patching(
                q, model_before, model_after, tokenizer, device
            )
            
            if result:
                results.append(result)
                dataset_counter[ds] += 1  # Only increment when analysis succeeds
                print(f"    Completed")
            
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
        
        # If all target datasets are full, stop early
        if all(dataset_counter.get(ds, 0) >= max_per_dataset for ds in CONFIG['target_datasets']):
            print("\nEnough questions analyzed for all datasets, stopping early.")
            break
    
    # Save results
    if results:
        print(f"\nSaving results...")
        
        # Simplify results for saving (remove large tensors)
        saved_results = []
        for r in results:
            saved_results.append({
                'dataset':  r['dataset'],
                'question_index': r['question_index'],
                'num_layers': r['num_layers'],
                'ffn_recovery':  [layer['accuracy'] for layer in r['patching_results']['ffn']],
                'attn_recovery': [layer['accuracy'] for layer in r['patching_results']['attn']],
                'ffn_critical_layers': sorted(
                    range(r['num_layers']),
                    key=lambda i:  r['patching_results']['ffn'][i]['accuracy'],
                    reverse=True
                )[:5],
                'attn_critical_layers': sorted(
                    range(r['num_layers']),
                    key=lambda i: r['patching_results']['attn'][i]['accuracy'],
                    reverse=True
                )[:5]
            })
        
        output_file = os.path.join(CONFIG['output_dir'], CONFIG['output_file'])
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': CONFIG,
                'num_analyzed':  len(results),
                'results': saved_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Saved:  {output_file}")
        
        # Visualize
        print(f"\nGenerating visualizations...")
        visualize_patching_results(results, CONFIG['output_dir'])
        
        print(f"Success:  {len(results)} questions")
    else:
        print("\nNo valid results")
    
    print("\nDone!")


if __name__ == "__main__":
    main()