"""
Calculate perplexity of correct answers on models at different stages, 
combined with accuracy analysis.
Overlay accuracy increment annotations on heatmap.
"""

import json
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict


# Configuration parameters
CONFIG = {
    'correct_answers_file': 'correct_answers_data.json',
    
    'model_checkpoints': [
        {
            'name': 'before_rlvr',
            'prefix': 'Qwen2.5-Math-7B',
            'path': '../Qwen2.5/Qwen2.5-Math-7B',
            'description': 'Before RLVR',
            'step': 0
        },
        {
            'name': 'rlvr_step50',
            'prefix': 'rethink_rlvr_reproduce-incorrect-qwen2.5_math_7b-lr5e-7-kl0.00-step50',
            'path':  '../rethink_rlvr_reproduce-incorrect-qwen2.5_math_7b-lr5e-7-kl0.00-step50',
            'description': 'Step 50',
            'step': 50
        },
        {
            'name': 'rlvr_step100',
            'prefix': 'rethink_rlvr_reproduce-incorrect-qwen2.5_math_7b-lr5e-7-kl0.00-step100',
            'path': '../rethink_rlvr_reproduce-incorrect-qwen2.5_math_7b-lr5e-7-kl0.00-step100',
            'description': 'Step 100',
            'step': 100
        },
        {
            'name': 'rlvr_step150',
            'prefix': 'rethink_rlvr_reproduce-incorrect-qwen2.5_math_7b-lr5e-7-kl0.00-step150',
            'path': '../rethink_rlvr_reproduce-incorrect-qwen2.5_math_7b-lr5e-7-kl0.00-step150',
            'description': 'Step 150',
            'step': 150
        }
    ],
    
    'eval_results_dir': '../outputs/eval_outputs',
    'output_file': 'perplexity_acc_analysis_results.json',
    'visualization_dir': 'perplexity_acc_visualizations/qwen',
    
    'device': 'cuda: 0' if torch.cuda.is_available() else 'cpu',
    'batch_size': 1,
    'max_length': 4096
}


def load_correct_answers(file_path: str) -> List[Dict]:
    """Load correct answer data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['correct_answers']


def load_accuracy_results(eval_results_dir: str, checkpoint_info: Dict) -> Dict[str, float]:
    """
    Load accuracy data from evaluation result files
    
    Returns:
        Dict[dataset_name, accuracy]:  Accuracy for each dataset
    """
    checkpoint_prefix = checkpoint_info['prefix']
    accuracy_by_dataset = {}
    
    print(f"\n  Looking for evaluation result files (prefix: {checkpoint_prefix})...")
    
    # Find all evaluation result files for this checkpoint
    pattern = os.path.join(eval_results_dir, f"{checkpoint_prefix}_*.json")
    result_files = glob.glob(pattern)
    
    print(f"  Found {len(result_files)} files")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            dataset_name = result_data.get('dataset', 'Unknown')
            
            # Try to get accuracy from different keys
            accuracy = None
            metadata = result_data.get('metadata', [])
            
            # Method 1: Calculate accuracy from metadata
            if metadata:
                correct_count = sum(1 for item in metadata if item.get('correct', False))
                total_count = len(metadata)
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                print(f"    {os.path.basename(result_file)}: {dataset_name} - {correct_count}/{total_count} = {accuracy*100:.2f}%")
            
            # Method 2: Get accuracy from result accuracy field
            if accuracy is None: 
                for key in result_data.keys():
                    if 'pass@' in key or 'avg@' in key or 'accuracy' in key.lower():
                        accuracy = result_data[key]
                        print(f"    {os.path.basename(result_file)}: {dataset_name} - {key} = {accuracy*100:.2f}%")
                        break
            
            if accuracy is not None:
                accuracy_by_dataset[dataset_name] = accuracy
            else:
                print(f"    Warning: Unable to extract accuracy from {result_file}")
                
        except Exception as e:
            print(f"    Warning: Failed to load evaluation result file {result_file}: {e}")
    
    if not accuracy_by_dataset:
        print(f"    Warning: No accuracy data found")
    
    return accuracy_by_dataset


def calculate_perplexity_for_prompt_and_answer(
        prompt: str,
        answer: str,
        model: AutoModelForCausalLM,
        tokenizer:  AutoTokenizer,
        device:  str
) -> Dict[str, float]:
    """
    Calculate perplexity for prompt and answer separately, and for full text
    """
    # Full text
    full_text = prompt + answer
    
    # Tokenize
    full_encodings = tokenizer(
        full_text,
        return_tensors='pt',
        truncation=True,
        max_length=CONFIG['max_length']
    ).to(device)
    
    prompt_encodings = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=CONFIG['max_length']
    ).to(device)
    
    full_input_ids = full_encodings['input_ids']
    prompt_length = prompt_encodings['input_ids'].shape[1]
    answer_length = full_input_ids.shape[1] - prompt_length
    
    # Calculate perplexity for full text
    with torch.no_grad():
        outputs = model(full_input_ids, labels=full_input_ids)
        
        # Get loss for each token
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = full_input_ids[..., 1:].contiguous()
        
        # Calculate negative log-likelihood for each token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Average loss for full text
        full_loss = token_losses.mean().item()
        full_ppl = np.exp(full_loss)
        
        # Calculate loss only for answer part
        if answer_length > 0 and prompt_length > 0:
            answer_token_losses = token_losses[prompt_length - 1:]
            answer_loss = answer_token_losses.mean().item()
            answer_ppl = np.exp(answer_loss)
        else:
            answer_ppl = full_ppl
    
    return {
        'full_text_ppl': full_ppl,
        'answer_only_ppl': answer_ppl,
        'prompt_length': prompt_length,
        'answer_length': answer_length,
        'total_length': full_input_ids.shape[1]
    }


def evaluate_perplexity_on_checkpoint(
        checkpoint_info: Dict,
        correct_answers: List[Dict]
) -> List[Dict]:
    """
    Evaluate perplexity of all correct answers on one checkpoint
    """
    print(f"\nLoading model:  {checkpoint_info['name']}")
    print(f"  Path: {checkpoint_info['path']}")
    print(f"  Description: {checkpoint_info['description']}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_info['path'],
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_info['path'],
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    
    device = CONFIG['device']
    
    results = []
    
    print(f"  Calculating perplexity...")
    for answer_data in tqdm(correct_answers, desc=f"  Processing {checkpoint_info['name']}"):
        prompt = answer_data['prompt']
        answer_text = answer_data['correct_answer_text']
        
        try:
            ppl_info = calculate_perplexity_for_prompt_and_answer(
                prompt, answer_text, model, tokenizer, device
            )
            
            result = {
                'dataset': answer_data['dataset'],
                'question_index': answer_data['question_index'],
                'checkpoint_name': checkpoint_info['name'],
                'checkpoint_path': checkpoint_info['path'],
                'step': checkpoint_info['step'],
                **ppl_info
            }
            
            results.append(result)
            
        except Exception as e: 
            print(f"\n  Error processing question (dataset={answer_data['dataset']}, "
                  f"index={answer_data['question_index']}): {e}")
            continue
    
    # Clean GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return results


def create_visualizations_with_accuracy(results_by_question: Dict, statistics: Dict, accuracy_data: Dict):
    """Create visualization charts with accuracy annotations"""
    print(f"\nCreating visualization charts...")
    
    # Create visualization directory
    vis_dir = CONFIG['visualization_dir']
    os.makedirs(vis_dir, exist_ok=True)
    
    # Prepare data
    checkpoint_names = [cp['name'] for cp in CONFIG['model_checkpoints']]
    checkpoint_labels = [cp['description'] for cp in CONFIG['model_checkpoints']]
    
    # 1.Line plot of perplexity change during training steps
    plt.figure(figsize=(12, 8))
    
    # Collect data for each checkpoint
    full_ppl_means = [statistics[cp]['full_text_ppl_mean'] for cp in checkpoint_names]
    full_ppl_stds = [statistics[cp]['full_text_ppl_std'] for cp in checkpoint_names]
    answer_ppl_means = [statistics[cp]['answer_only_ppl_mean'] for cp in checkpoint_names]
    answer_ppl_stds = [statistics[cp]['answer_only_ppl_std'] for cp in checkpoint_names]
    
    # Plot full text perplexity
    plt.subplot(2, 2, 1)
    plt.errorbar(range(len(checkpoint_names)), full_ppl_means, yerr=full_ppl_stds,
                 marker='o', capsize=5, capthick=2, label='Full Text')
    plt.xticks(range(len(checkpoint_names)), checkpoint_labels, rotation=45, ha='right', fontsize=14)
    plt.ylabel('Perplexity', fontsize=20, fontweight='bold')
    plt.title('Full Text Perplexity Across Training Steps', fontsize=17, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yticks(fontsize=14)
    
    # Plot answer-only perplexity
    plt.subplot(2, 2, 2)
    plt.errorbar(range(len(checkpoint_names)), answer_ppl_means, yerr=answer_ppl_stds,
                 marker='s', capsize=5, capthick=2, color='orange', label='Answer Only')
    plt.xticks(range(len(checkpoint_names)), checkpoint_labels, rotation=45, ha='right', fontsize=14)
    plt.ylabel('Perplexity', fontsize=20, fontweight='bold')
    plt.title('Answer-only Perplexity Across Training Steps', fontsize=17, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yticks(fontsize=14)
    
    # 2.Perplexity distribution box plots
    full_ppl_data = []
    answer_ppl_data = []
    
    for cp_name in checkpoint_names:
        full_ppls = []
        answer_ppls = []
        for question_data in results_by_question.values():
            if cp_name in question_data['checkpoints']:
                full_ppls.append(question_data['checkpoints'][cp_name]['full_text_ppl'])
                answer_ppls.append(question_data['checkpoints'][cp_name]['answer_only_ppl'])
        full_ppl_data.append(full_ppls)
        answer_ppl_data.append(answer_ppls)
    
    plt.subplot(2, 2, 3)
    plt.boxplot(full_ppl_data, labels=checkpoint_labels, patch_artist=True)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.ylabel('Perplexity', fontsize=20, fontweight='bold')
    plt.title('Full Text Perplexity Distribution', fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yticks(fontsize=14)
    
    plt.subplot(2, 2, 4)
    plt.boxplot(answer_ppl_data, labels=checkpoint_labels, patch_artist=True, boxprops=dict(facecolor='orange'))
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.ylabel('Perplexity', fontsize=20, fontweight='bold')
    plt.title('Answer-only Perplexity Distribution', fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'perplexity_trends.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    # 3.Perplexity comparison heatmap for each dataset with accuracy annotations
    datasets = list(set(q['dataset'] for q in results_by_question.values()))
    if len(datasets) > 1:
        # Calculate average perplexity for each dataset at each checkpoint
        dataset_full_ppl_matrix = np.zeros((len(datasets), len(checkpoint_names)))
        dataset_answer_ppl_matrix = np.zeros((len(datasets), len(checkpoint_names)))
        
        # Calculate accuracy matrix
        dataset_accuracy_matrix = np.zeros((len(datasets), len(checkpoint_names)))
        
        for i, dataset in enumerate(datasets):
            for j, cp_name in enumerate(checkpoint_names):
                # Perplexity
                full_ppls = []
                answer_ppls = []
                for question_data in results_by_question.values():
                    if question_data['dataset'] == dataset and cp_name in question_data['checkpoints']:
                        full_ppls.append(question_data['checkpoints'][cp_name]['full_text_ppl'])
                        answer_ppls.append(question_data['checkpoints'][cp_name]['answer_only_ppl'])
                dataset_full_ppl_matrix[i, j] = np.mean(full_ppls) if full_ppls else 0
                dataset_answer_ppl_matrix[i, j] = np.mean(answer_ppls) if answer_ppls else 0
                
                # Accuracy
                dataset_accuracy_matrix[i, j] = accuracy_data.get(cp_name, {}).get(dataset, 0)
        
        # Create two versions of accuracy annotation matrix: 
        # 1.First checkpoint shows actual accuracy
        # 2.Other checkpoints show improvement relative to first checkpoint
        accuracy_annotations_matrix = np.zeros((len(datasets), len(checkpoint_names)))
        for i in range(len(datasets)):
            base_acc = dataset_accuracy_matrix[i, 0]  # Baseline is first checkpoint
            for j in range(len(checkpoint_names)):
                if j == 0:  # First checkpoint shows actual accuracy
                    accuracy_annotations_matrix[i, j] = dataset_accuracy_matrix[i, j] * 100
                else:  # Other checkpoints show improvement
                    accuracy_annotations_matrix[i, j] = (dataset_accuracy_matrix[i, j] - base_acc) * 100
        
        # Plot full text perplexity heatmap with accuracy annotations
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(dataset_full_ppl_matrix,
                    xticklabels=checkpoint_labels,
                    yticklabels=datasets,
                    annot=True,
                    fmt='.2f',
                    cmap='YlOrRd',
                    ax=ax,
                    cbar_kws={'label': 'Perplexity'})
        
        # Add accuracy annotations
        for i in range(len(datasets)):
            for j in range(len(checkpoint_names)):
                ppl_val = dataset_full_ppl_matrix[i, j]
                acc_value = accuracy_annotations_matrix[i, j]
                
                # Use different colors and symbols based on whether it's actual accuracy or improvement
                if j == 0:  # Actual accuracy
                    color = 'black'
                    text = f'{acc_value:.1f}%'
                else:  # Improvement
                    if acc_value >= 0:
                        color = 'blue'
                        symbol = 'up'
                    else: 
                        color = 'red'
                        symbol = 'down'
                    text = f'{symbol}{abs(acc_value):.1f}%'
                
                # Add accuracy annotation in heatmap cell
                ax.text(j + 0.5, i + 0.7,
                        text,
                        ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)
        
        plt.title(
            'Full Text Perplexity with Accuracy Annotations\n(First column shows actual accuracy, others show change from baseline)',
            fontsize=16, fontweight='bold')
        plt.xlabel('Checkpoint', fontsize=20, fontweight='bold')
        plt.ylabel('Dataset', fontsize=20, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(rotation=0, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'dataset_full_text_perplexity_with_acc.svg'),
                    format='svg', bbox_inches='tight')
        plt.close()
        
        # Plot answer-only perplexity heatmap with accuracy annotations
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(dataset_answer_ppl_matrix,
                    xticklabels=checkpoint_labels,
                    yticklabels=datasets,
                    annot=True,
                    fmt='.2f',
                    cmap='YlOrRd',
                    ax=ax,
                    cbar_kws={'label': 'Perplexity'})
        
        # Add accuracy annotations
        for i in range(len(datasets)):
            for j in range(len(checkpoint_names)):
                ppl_val = dataset_answer_ppl_matrix[i, j]
                acc_value = accuracy_annotations_matrix[i, j]
                
                # Use different colors and symbols based on whether it's actual accuracy or improvement
                if j == 0:  # Actual accuracy
                    color = 'black'
                    text = f'{acc_value:.1f}%'
                else:  # Improvement
                    if acc_value >= 0:
                        color = 'blue'
                        symbol = 'up'
                    else:
                        color = 'black'
                        symbol = 'down'
                    text = f'{symbol}{abs(acc_value):.1f}%'
                
                # Add accuracy annotation in heatmap cell
                ax.text(j + 0.5, i + 0.7,
                        text,
                        ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)
        
        plt.title(
            'Answer-only Perplexity with Accuracy Annotations\n(First column shows actual accuracy, others show change from baseline)',
            fontsize=16, fontweight='bold')
        plt.xlabel('Checkpoint', fontsize=20, fontweight='bold')
        plt.ylabel('Dataset', fontsize=20, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(rotation=0, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'dataset_answer_perplexity_with_acc.svg'),
                    format='svg', bbox_inches='tight')
        plt.close()
    
    # 4.Perplexity improvement during training
    if len(checkpoint_names) > 1:
        initial_cp = checkpoint_names[0]
        final_cp = checkpoint_names[-1]
        
        improvements = []
        for question_data in results_by_question.values():
            if initial_cp in question_data['checkpoints'] and final_cp in question_data['checkpoints']:
                initial_ppl = question_data['checkpoints'][initial_cp]['answer_only_ppl']
                final_ppl = question_data['checkpoints'][final_cp]['answer_only_ppl']
                improvement = initial_ppl - final_ppl
                improvements.append(improvement)
        
        if improvements: 
            plt.figure(figsize=(10, 6))
            plt.hist(improvements, bins=30, alpha=0.7, color='green')
            plt.xlabel('Perplexity Improvement (Initial - Final)', fontsize=20, fontweight='bold')
            plt.ylabel('Number of Questions', fontsize=20, fontweight='bold')
            plt.title('Distribution of Answer-only Perplexity Improvement During Training',
                      fontsize=20, fontweight='bold')
            plt.axvline(0, color='red', linestyle='--', label='No Improvement Line')
            plt.axvline(np.mean(improvements), color='blue', linestyle='-',
                        label=f'Mean Improvement: {np.mean(improvements):.2f}')
            plt.legend(fontsize=14, loc='best', framealpha=0.9)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'perplexity_improvement.svg'),
                        format='svg', bbox_inches='tight')
            plt.close()
    
    print(f"Visualization charts saved to {vis_dir}")


def main():
    """Main function"""
    print("=" * 60)
    print("Starting perplexity and accuracy comprehensive analysis...")
    print("=" * 60)
    
    # Load correct answer data
    print(f"\nLoading correct answer data:  {CONFIG['correct_answers_file']}")
    correct_answers = load_correct_answers(CONFIG['correct_answers_file'])
    print(f"Loaded {len(correct_answers)} correct answers")
    
    # Load accuracy data for all checkpoints
    print(f"\nLoading accuracy data...")
    accuracy_data = {}
    for checkpoint_info in CONFIG['model_checkpoints']: 
        print(f"\nProcessing {checkpoint_info['name']}...")
        accuracy_by_dataset = load_accuracy_results(
            CONFIG['eval_results_dir'], 
            checkpoint_info
        )
        accuracy_data[checkpoint_info['name']] = accuracy_by_dataset
        print(f"  Total:  {len(accuracy_by_dataset)} datasets")
    
    # Calculate perplexity for each checkpoint
    all_results = []
    
    for checkpoint_info in CONFIG['model_checkpoints']: 
        checkpoint_results = evaluate_perplexity_on_checkpoint(
            checkpoint_info, correct_answers
        )
        all_results.extend(checkpoint_results)
    
    # Organize results
    results_by_question = {}
    for result in all_results:
        key = (result['dataset'], result['question_index'])
        if key not in results_by_question: 
            results_by_question[key] = {
                'dataset': result['dataset'],
                'question_index': result['question_index'],
                'checkpoints':  {}
            }
        
        checkpoint_name = result['checkpoint_name']
        results_by_question[key]['checkpoints'][checkpoint_name] = {
            'full_text_ppl': result['full_text_ppl'],
            'answer_only_ppl': result['answer_only_ppl'],
            'prompt_length': result['prompt_length'],
            'answer_length': result['answer_length'],
            'total_length': result['total_length']
        }
    
    # Calculate statistics
    statistics = {}
    for checkpoint_info in CONFIG['model_checkpoints']: 
        checkpoint_name = checkpoint_info['name']
        full_ppls = []
        answer_ppls = []
        
        for question_data in results_by_question.values():
            if checkpoint_name in question_data['checkpoints']:
                full_ppls.append(question_data['checkpoints'][checkpoint_name]['full_text_ppl'])
                answer_ppls.append(question_data['checkpoints'][checkpoint_name]['answer_only_ppl'])
        
        if full_ppls:
            statistics[checkpoint_name] = {
                'full_text_ppl_mean': float(np.mean(full_ppls)),
                'full_text_ppl_std': float(np.std(full_ppls)),
                'full_text_ppl_median': float(np.median(full_ppls)),
                'answer_only_ppl_mean':  float(np.mean(answer_ppls)),
                'answer_only_ppl_std': float(np.std(answer_ppls)),
                'answer_only_ppl_median': float(np.median(answer_ppls)),
                'num_samples': len(full_ppls)
            }
    
    # Save results
    output_data = {
        'config': CONFIG,
        'total_questions': len(results_by_question),
        'statistics': statistics,
        'accuracy_data': accuracy_data,
        'results_by_question': list(results_by_question.values()),
        'all_results': all_results
    }
    
    print(f"\nSaving results to:  {CONFIG['output_file']}")
    with open(CONFIG['output_file'], 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    # Create visualization charts
    create_visualizations_with_accuracy(results_by_question, statistics, accuracy_data)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Perplexity and Accuracy Statistical Summary:")
    print("-" * 60)
    for checkpoint_info in CONFIG['model_checkpoints']: 
        checkpoint_name = checkpoint_info['name']
        if checkpoint_name in statistics:
            stats = statistics[checkpoint_name]
            print(f"\n{checkpoint_info['description']} (Step {checkpoint_info['step']}):")
            print(f"  Full text perplexity: {stats['full_text_ppl_mean']:.4f} +/- {stats['full_text_ppl_std']:.4f}")
            print(f"  Answer-only perplexity: {stats['answer_only_ppl_mean']:.4f} +/- {stats['answer_only_ppl_std']:.4f}")
            print(f"  Number of samples: {stats['num_samples']}")
            
            if checkpoint_name in accuracy_data and accuracy_data[checkpoint_name]: 
                print(f"  Accuracy:")
                for dataset, acc in sorted(accuracy_data[checkpoint_name].items()):
                    print(f"    {dataset}: {acc*100:.2f}%")
            else:
                print(f"  Accuracy: No data")
    print("=" * 60)
    
    print("\nAnalysis complete!")


if __name__ == "__main__": 
    main()