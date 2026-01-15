"""
MLP Logit Lens Analysis

Analyzes how MLP layers transform hidden states and their predictions at each layer.
"""

import os
import json
import re
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

warnings.filterwarnings('ignore')


# Configuration parameters
CONFIG = {
    'correct_answers_file': 'correct_answers_data.json',
    
    'model_checkpoint': {
        'name': 'after_rlvr',
        'path': '../rethink_rlvr_reproduce-incorrect-qwen2.5_math_7b-lr5e-7-kl0.00-step150',
        'description': 'rlvr_150'
    },
    
    'output_dir': 'qwen_logits_lens',
    'output_file': 'logit_lens_mlp_results.json',
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    
    'num_questions_per_dataset': 999,
    'target_datasets': ['MATH-500'],
    'target_question_index': 141,
    
    'max_new_tokens': 4096,
    'temperature': 0.7,
}


def load_correct_answers(file_path: str) -> List[Dict]:
    """Load correct answer data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        if 'results' in data:
            return data['results']
        elif 'questions' in data:
            return data['questions']
        elif 'data' in data:
            return data['data']
        elif 'correct_answers' in data:
            return data['correct_answers']
        else:
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    return value
            return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected data format in {file_path}")


def clean_token_text(token_str: str) -> str:
    """Clean token text by removing special characters"""
    token_str = token_str.replace('Ġ', ' ')
    token_str = token_str.replace('Ċ', '\n')
    token_str = token_str.replace('ĉ', '\t')
    return token_str


def find_answer_token_position(
    generated_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    prompt_length: int
) -> Optional[Tuple[int, str]]:
    """Find position of first answer token after boxed"""
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    generated_text = tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=False)
    
    print(f"      Generated text preview:  {generated_text[:300]}...")
    
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
            after_answer = full_text[answer_start:].lstrip()
            answer_content = after_answer[:50]
            print(f"      Using answer tag fallback: {answer_content}...")
        else:
            print(f"      Using last token as fallback")
            return None, None
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
        return None, None
    
    answer_preview = answer_content[:20]
    answer_tokens = tokenizer(answer_content, add_special_tokens=False)['input_ids']
    if not answer_tokens:
        print(f"      Warning: Answer tokenization failed")
        return None, None
    
    full_token_list = generated_ids[0].cpu().tolist()
    first_answer_token_id = answer_tokens[0]
    
    for i in range(prompt_length, len(full_token_list)):
        if full_token_list[i] == first_answer_token_id:
            match_len = min(3, len(answer_tokens))
            if full_token_list[i:i+match_len] == answer_tokens[:match_len]:
                print(f"      Found answer token at position {i}")
                print(f"      Answer preview: '{answer_preview}'")
                return i, answer_content
    
    print(f"      Warning: Could not precisely locate answer token")
    fallback_pos = len(full_token_list) - min(5, len(answer_tokens))
    print(f"      Using fallback position: {fallback_pos}")
    return fallback_pos, answer_content


def mlp_logit_lens_analysis(
    prompt: str,
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    device: str,
    top_k: int = 10,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> Dict:
    """MLP Logit Lens analysis"""
    try:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
        input_ids = inputs['input_ids'].to(device)
        prompt_length = input_ids.shape[1]
        
        print(f"      Input length: {prompt_length} tokens")
        
    except Exception as e:
        print(f"      Tokenization failed: {e}")
        return None
    
    model.eval()
    
    with torch.no_grad():
        try:
            # Step 1: Normal generation and analysis
            print(f"      [Step 1] Generating response (max {max_new_tokens} tokens)...")
            
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                top_p=0.95 if temperature > 0 else None,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            generated_text = tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=False)
            print(f"      Generated {generated_ids.shape[1] - prompt_length} tokens")
            
            answer_position, answer_content = find_answer_token_position(
                generated_ids, tokenizer, prompt_length
            )
            
            if answer_position is None:
                answer_position = generated_ids.shape[1] - 1
                answer_content = "N/A"
                print(f"      Using last token at position {answer_position}")
            
            # Step 2: Normal analysis of all layers
            print(f"      [Step 2] Running normal analysis...")
            logits, cache = model.run_with_cache(generated_ids)
            
            target_logits = logits[0, answer_position, :]
            target_token_id = torch.argmax(target_logits).item()
            target_token = tokenizer.decode([target_token_id])
            
            print(f"      Position {answer_position} prediction: {clean_token_text(target_token)}")
            
            n_layers = model.cfg.n_layers
            layer_results_normal = []
            
            def get_top_k_predictions(hidden_state, top_k):
                """Get top-k predictions from hidden state"""
                if hasattr(model, 'ln_final'):
                    normed = model.ln_final(hidden_state.unsqueeze(0)).squeeze(0)
                else: 
                    normed = hidden_state
                
                layer_logits = model.unembed(normed)
                layer_probs = torch.softmax(layer_logits, dim=-1)
                
                top_k_probs, top_k_indices = torch.topk(layer_probs, k=top_k)
                
                top_k_tokens = []
                for prob, idx in zip(top_k_probs.cpu().numpy(), top_k_indices.cpu().numpy()):
                    token_str = tokenizer.decode([idx])
                    top_k_tokens.append({
                        'token': clean_token_text(token_str),
                        'token_id': int(idx),
                        'probability': float(prob)
                    })
                
                return {
                    'top_k':  top_k_tokens,
                    'predicted_token_id': int(top_k_indices[0].item()),
                    'predicted_token': top_k_tokens[0]['token'],
                    'logits': layer_logits.cpu().numpy()
                }
            
            for layer_idx in range(n_layers):
                pre_mlp_hidden = cache[f'blocks.{layer_idx}.hook_resid_mid'][0, answer_position, :]
                mlp_output = cache[f'blocks.{layer_idx}.hook_mlp_out'][0, answer_position, :]
                resid_pre = cache[f'blocks.{layer_idx}.hook_resid_pre'][0, answer_position, :]
                attention_output = pre_mlp_hidden - resid_pre
                
                pre_mlp_pred = get_top_k_predictions(pre_mlp_hidden, top_k)
                post_mlp_pred = get_top_k_predictions(mlp_output, top_k)
                attention_pred = get_top_k_predictions(attention_output, top_k)
                
                layer_results_normal.append({
                    'layer':  layer_idx,
                    'pre_mlp':  pre_mlp_pred,
                    'post_mlp': post_mlp_pred,
                    'attention_output': attention_pred
                })
            
            # Final output layer prediction
            final_pred = {
                'top_k': [],
                'predicted_token_id': target_token_id,
                'predicted_token': clean_token_text(target_token),
                'logits': target_logits.cpu().numpy()
            }
            
            final_probs = torch.softmax(target_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(final_probs, k=top_k)
            for prob, idx in zip(top_k_probs.cpu().numpy(), top_k_indices.cpu().numpy()):
                token_str = tokenizer.decode([idx])
                final_pred['top_k'].append({
                    'token': clean_token_text(token_str),
                    'token_id': int(idx),
                    'probability': float(prob)
                })
            
            print(f"\n      Analysis complete")
            
        except Exception as e:
            print(f"      Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return {
        'prompt': prompt,
        'generated_text': generated_text,
        'answer_content': answer_content,
        'answer_position': answer_position,
        'total_length': generated_ids.shape[1],
        'layer_results': layer_results_normal,
        'final_prediction': final_pred,
        'n_layers': n_layers,
    }


def visualize_mlp_logit_lens(result:  Dict, output_dir: str, dataset:  str, question_idx: int):
    """Visualize MLP Logit Lens results"""
    question_dir = os.path.join(output_dir, f"{dataset}_q{question_idx}")
    os.makedirs(question_dir, exist_ok=True)
    
    layer_results = result['layer_results']
    final_pred = result['final_prediction']
    n_layers = result['n_layers']
    final_token = final_pred['predicted_token']
    
    print(f"    Visualizing MLP Logit Lens for {n_layers} layers...")
    
    # Figure 1: Top-1 prediction comparison for three states
    fig, axes = plt.subplots(1, 3, figsize=(24, max(10, n_layers * 0.4)))
    
    states = ['pre_mlp', 'post_mlp', 'attention_output']
    state_titles = [
        'Pre-MLP\n(Before MLP)',
        'MLP Output\n(MLP Contribution)',
        'Attention Output\n(Attention Contribution)'
    ]
    state_colors = ['steelblue', 'coral', 'purple']
    
    for ax, state, title, color in zip(axes, states, state_titles, state_colors):
        layers = [r['layer'] for r in layer_results]
        predicted_tokens = [r[state]['predicted_token'] for r in layer_results]
        predicted_probs = [r[state]['top_k'][0]['probability'] for r in layer_results]
        
        colors_bar = ['darkgreen' if token == final_token else color for token in predicted_tokens]
        
        bars = ax.barh(layers, predicted_probs, color=colors_bar, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(layers)
        ax.set_yticklabels([f'L{l}' for l in layers], fontsize=13)
        ax.set_xlabel('Top-1 Probability', fontsize=25, fontweight='bold')
        ax.set_ylabel('Layer', fontsize=25, fontweight='bold')
        ax.set_title(title, fontsize=25, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        ax.set_xlim([0, 1])
        
        for i, (layer, token, prob) in enumerate(zip(layers, predicted_tokens, predicted_probs)):
            token_display = token[:10] + '...' if len(token) > 10 else token
            ax.text(prob + 0.02, i, f'{token_display}', va='center', fontsize=13, fontweight='bold')
    
    plt.suptitle(
        f'MLP Logit Lens:  Top-1 Predictions at Each State\n'
        f'Final:  "{final_token}" | {dataset} Q{question_idx}',
        fontsize=25, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(question_dir, '1_mlp_logit_lens_comparison.svg'), format='svg', dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    
    # Figure 2: MLP contribution
    fig, ax = plt.subplots(figsize=(14, max(10, n_layers * 0.4)))
    
    layers = [r['layer'] for r in layer_results]
    mlp_changes = []
    mlp_prob_changes = []
    
    for r in layer_results:
        pre_token = r['pre_mlp']['predicted_token']
        post_token = r['post_mlp']['predicted_token']
        pre_prob = r['pre_mlp']['top_k'][0]['probability']
        post_prob = r['post_mlp']['top_k'][0]['probability']
        
        changed = (pre_token != post_token)
        prob_change = post_prob - pre_prob
        
        mlp_changes.append(changed)
        mlp_prob_changes.append(prob_change)
    
    colors = ['red' if changed else 'green' for changed in mlp_changes]
    bars = ax.barh(layers, mlp_prob_changes, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axvline(x=0, color='black', linewidth=2, linestyle='--', alpha=0.5)
    ax.set_yticks(layers)
    ax.set_yticklabels([f'Layer {l}' for l in layers], fontsize=9)
    ax.set_xlabel('Probability Change (Post-MLP - Pre-MLP)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Layer', fontsize=13, fontweight='bold')
    ax.set_title(
        f'MLP Impact on Top-1 Prediction\n'
        f'Red = Changed prediction, Green = Same prediction\n'
        f'{dataset} Q{question_idx}',
        fontsize=14, fontweight='bold'
    )
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    for i, (layer, changed, prob_change) in enumerate(zip(layers, mlp_changes, mlp_prob_changes)):
        if changed:
            pre_token = layer_results[i]['pre_mlp']['predicted_token'][:8]
            post_token = layer_results[i]['post_mlp']['predicted_token'][:8]
            ax.text(prob_change, i, f' {pre_token}->{post_token}', va='center', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(question_dir, '2_mlp_impact.svg'), format='svg', dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    
    # Figure 3: Top-10 Tokens heatmap for each MLP layer
    fig, ax = plt.subplots(figsize=(15, max(10, n_layers * 0.5)))
    
    # Collect top-10 tokens from MLP output for all layers
    all_tokens = []
    all_probs = []
    layer_labels = []
    
    for r in layer_results:
        layer_idx = r['layer']
        top_k_tokens = r['post_mlp']['top_k']
        
        tokens = [item['token'] for item in top_k_tokens]
        probs = [item['probability'] for item in top_k_tokens]
        
        all_tokens.append(tokens)
        all_probs.append(probs)
        layer_labels.append(f'Layer {layer_idx}')
    
    # Convert to numpy array for heatmap
    probs_array = np.array(all_probs)
    
    # Create heatmap
    im = ax.imshow(probs_array, cmap='YlOrRd', aspect='auto')
    
    # Set axes
    ax.set_yticks(range(len(layer_labels)))
    ax.set_yticklabels(layer_labels, fontsize=10)
    
    # Set x-axis labels (token position index)
    ax.set_xticks(range(len(all_tokens[0])))
    ax.set_xticklabels([f'Top-{i+1}' for i in range(len(all_tokens[0]))], fontsize=13)
    
    # Add token name and probability value in each cell
    for i in range(len(layer_labels)):
        for j in range(len(all_tokens[0])):
            # Truncate long token names to fit cells
            token_name = all_tokens[i][j]
            if len(token_name) > 8:
                token_display = token_name[:8] + '..'
            else:
                token_display = token_name
                
            prob_value = probs_array[i, j]
            # Display token name and probability value in cell
            text_content = f'{token_display}\n{prob_value:.3f}'
            text = ax.text(j, i, text_content,
                        ha="center", va="center", color="black", fontsize=11)
    
    ax.set_xlabel('Top-K Position', fontsize=25, fontweight='bold')
    ax.set_ylabel('Layer', fontsize=25, fontweight='bold')
    ax.set_title(f'Top-10 Tokens Probability Distribution for MLP Outputs\n{dataset} Q{question_idx}',
                fontsize=25, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability', fontsize=25, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(question_dir, '5_mlp_top10_tokens_heatmap.svg'), format='svg', dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"    Saved MLP top-10 tokens heatmap")
    print(f"    Saved visualizations to {question_dir}")
    
    return {
        'question_dir': question_dir,
        'num_mlp_changes': sum(mlp_changes),
        'final_token': final_token
    }


def analyze_question(
    question_data: Dict,
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    device: str
) -> Dict:
    """Analyze single question"""
    prompt = question_data['prompt']
    print(f"    Running MLP Logit Lens analysis...")
    
    result = mlp_logit_lens_analysis(
        prompt, model, tokenizer, device,
        top_k=10,
        max_new_tokens=CONFIG['max_new_tokens'],
        temperature=CONFIG['temperature']
    )
    
    if result is None:
        print(f"    Warning: Skip")
        return None
    
    torch.cuda.empty_cache()
    
    print(f"    Visualizing...")
    viz_result = visualize_mlp_logit_lens(
        result,
        CONFIG['output_dir'],
        question_data['dataset'],
        question_data['question_index']
    )
    
    return {
        'dataset': question_data['dataset'],
        'question_index': question_data['question_index'],
        'prompt': prompt,
        'result': result,
        'visualization':  viz_result,
    }


def is_single_token_answer(answer: str, tokenizer: AutoTokenizer) -> bool:
    """Check if answer is a single token"""
    if not answer or not isinstance(answer, str):
        return False
    
    answer = answer.strip()
    tokens = tokenizer(answer, add_special_tokens=False)['input_ids']
    is_single = len(tokens) == 1
    
    if is_single:
        token_text = tokenizer.decode(tokens)
        print(f"      Single token answer: '{answer}' -> '{token_text}' (ID: {tokens[0]})")
    
    return is_single


def main():
    """Main function - evaluate only MATH-500 Question specified in CONFIG"""
    print("=" * 60)
    print("MLP Logit Lens Analysis")
    print(f"Target:  {CONFIG['target_datasets'][0]} Question #{CONFIG['target_question_index']}")
    print("=" * 60)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    print(f"\nLoading:  {CONFIG['correct_answers_file']}")
    answers = load_correct_answers(CONFIG['correct_answers_file'])
    print(f"Total {len(answers)} questions")
    
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model_checkpoint']['path'],
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")
    
    # Select only target question
    target_question = None
    for item in answers:
        if not isinstance(item, dict):
            continue
        
        if item.get('dataset') != CONFIG['target_datasets'][0]: 
            continue
        
        if item.get('question_index') == CONFIG['target_question_index']:
            target_question = item
            break
    
    if target_question is None:
        print(f"Target question not found: {CONFIG['target_datasets'][0]} Q{CONFIG['target_question_index']}!")
        return
    
    print(f"\nFound target question:")
    print(f"  Dataset: {target_question['dataset']}")
    print(f"  Index: {target_question['question_index']}")
    answer = target_question.get('answer') or target_question.get('label', 'N/A')
    print(f"  Answer: {answer}")
    
    if not is_single_token_answer(str(answer), tokenizer):
        print(f"  Warning: Answer is not a single token, but continuing anyway...")
    
    print(f"\nLoading model with transformer_lens...")
    print(f"  Path: {CONFIG['model_checkpoint']['path']}")
    
    print("  Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_checkpoint']['path'],
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    
    print("  Converting to HookedTransformer...")
    model = HookedTransformer.from_pretrained(
        model_name="Qwen/Qwen2.5-7B",
        hf_model=hf_model,
        device=CONFIG['device'],
        fold_ln=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=tokenizer,
        dtype=torch.float32,
    )
    
    model = model.to(CONFIG['device'])
    model.eval()
    
    del hf_model
    torch.cuda.empty_cache()
    
    print("Model loaded successfully")
    
    print(f"\n{'='*60}")
    print(f"Processing {CONFIG['target_datasets'][0]} Q{CONFIG['target_question_index']}...")
    print(f"Ground truth answer: '{answer}'")
    print(f"{'='*60}")
    
    try:
        result = analyze_question(target_question, model, tokenizer, CONFIG['device'])
        
        if result: 
            result['ground_truth'] = answer
            
            print(f"\n{'='*60}")
            print(f"Saving results...")
            print(f"{'='*60}")
            
            out_file = os.path.join(CONFIG['output_dir'], CONFIG['output_file'])
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'config': CONFIG,
                    'question': {
                        'dataset': result['dataset'],
                        'question_index': result['question_index'],
                        'ground_truth': result.get('ground_truth', 'N/A'),
                        'prompt_preview': result['result']['prompt'][:200] + '...',
                        'generated_text_preview': result['result']['generated_text'][:200] + '...',
                        'answer_content': result['result']['answer_content'],
                        'answer_position':  result['result']['answer_position'],
                    },
                    'output_dir': result['visualization']['question_dir']
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\nSaved:  {out_file}")
    
    except Exception as e: 
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    
    if result: 
        # Remove non-serializable numpy arrays (like logits), or recursively convert to list
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.cpu().tolist()
            return str(obj)
        
        out_file = os.path.join(CONFIG['output_dir'], "logit_lens_mlp_full_results.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=to_serializable)
        print(f"\nAll analysis data saved to: {out_file}")


if __name__ == "__main__":
    main()