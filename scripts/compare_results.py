import json
import os
import argparse
import csv
from typing import Dict, Any, Tuple


def load_result(file_path: str) -> Dict[Any, Any]:
    """Load evaluation result file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_models(base_result: Dict[Any, Any], finetuned_result: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Compare two model results

    Returns:
    - base_acc: base model accuracy
    - finetuned_acc: finetuned model accuracy
    - base_correct_finetuned_wrong: count where base is correct but finetuned is wrong
    - base_wrong_finetuned_correct: count where base is wrong but finetuned is correct
    - both_correct: count where both are correct
    - both_wrong: count where both are wrong
    - changed_items: indices where answers changed
    """
    base_metadata = base_result.get('metadata', [])
    finetuned_metadata = finetuned_result.get('metadata', [])

    # Ensure both results have the same number of questions
    assert len(base_metadata) == len(finetuned_metadata), "Mismatch in number of questions between files"

    base_correct_count = 0
    finetuned_correct_count = 0
    base_correct_finetuned_wrong = 0
    base_wrong_finetuned_correct = 0
    both_correct = 0
    both_wrong = 0

    # Track question changes
    changed_items = {
        'base_correct_finetuned_wrong': [],  # base correct, finetuned wrong
        'base_wrong_finetuned_correct': []   # base wrong, finetuned correct
    }

    # Detailed question info
    detailed_changed_items = {
        'base_correct_finetuned_wrong': [],
        'base_wrong_finetuned_correct': []
    }

    for base_item, finetuned_item in zip(base_metadata, finetuned_metadata):
        # Ensure same question
        assert base_item['index'] == finetuned_item['index'], "Question index mismatch"

        base_correct = base_item['correct']
        finetuned_correct = finetuned_item['correct']

        if base_correct:
            base_correct_count += 1

        if finetuned_correct:
            finetuned_correct_count += 1

        if base_correct and not finetuned_correct:
            base_correct_finetuned_wrong += 1
            changed_items['base_correct_finetuned_wrong'].append(base_item['index'])
            detailed_changed_items['base_correct_finetuned_wrong'].append({
                'index': base_item['index'],
                'base_item': base_item,
                'finetuned_item': finetuned_item
            })
        elif not base_correct and finetuned_correct:
            base_wrong_finetuned_correct += 1
            changed_items['base_wrong_finetuned_correct'].append(base_item['index'])
            detailed_changed_items['base_wrong_finetuned_correct'].append({
                'index': base_item['index'],
                'base_item': base_item,
                'finetuned_item': finetuned_item
            })
        elif base_correct and finetuned_correct:
            both_correct += 1
        else:
            both_wrong += 1

    total = len(base_metadata)
    base_acc = base_correct_count / total if total > 0 else 0
    finetuned_acc = finetuned_correct_count / total if total > 0 else 0

    return {
        'base_acc': base_acc,
        'finetuned_acc': finetuned_acc,
        'base_correct_finetuned_wrong': base_correct_finetuned_wrong,
        'base_wrong_finetuned_correct': base_wrong_finetuned_correct,
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        'total': total,
        'changed_items': changed_items,
        'detailed_changed_items': detailed_changed_items
    }


def analyze_dataset(base_file: str, finetuned_file: str) -> None:
    """Analyze results for a single dataset"""
    print(f"Analyzing: {os.path.basename(base_file)} vs {os.path.basename(finetuned_file)}")

    base_result = load_result(base_file)
    finetuned_result = load_result(finetuned_file)

    # Get dataset name
    dataset_name = base_result.get('dataset', 'Unknown')

    comparison = compare_models(base_result, finetuned_result)

    print(f"\nDataset: {dataset_name}")
    print(f"Total questions: {comparison['total']}")
    print(f"Base accuracy: {comparison['base_acc']:.4f} ({comparison['base_acc'] * 100:.2f}%)")
    print(f"Finetuned accuracy: {comparison['finetuned_acc']:.4f} ({comparison['finetuned_acc'] * 100:.2f}%)")
    print(f"Base correct but finetuned wrong: {comparison['base_correct_finetuned_wrong']} items")
    print(f"Base wrong but finetuned correct: {comparison['base_wrong_finetuned_correct']} items")
    print(f"Both correct: {comparison['both_correct']} items")
    print(f"Both wrong: {comparison['both_wrong']} items")
    print("-" * 50)


def save_changed_questions_to_csv(all_changed_items: list, output_file: str = "changed_questions.csv"):
    """
    Save questions whose answers changed after finetuning to CSV

    Args:
        all_changed_items: list of all datasets' changed question info
        output_file: CSV file path
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['dataset', 'question_index'])

        # Only save questions where base was wrong and finetuned is correct
        for item in all_changed_items:
            dataset_name = item['dataset']
            changed_indices = item['changed_items']['base_wrong_finetuned_correct']

            for index in changed_indices:
                writer.writerow([dataset_name, index])

    print(f"Saved 'base wrong, finetuned correct' question indices to {output_file}")


def save_correct_answers_json(all_detailed_changed_items: list, config: dict,
                              output_file: str = "correct_answers_data.json"):
    """
    Save questions that became correct after finetuning to JSON

    Args:
        all_detailed_changed_items: list of detailed changed question info for all datasets
        config: configuration info
        output_file: JSON file path
    """
    correct_answers = []

    for item in all_detailed_changed_items:
        dataset_name = item['dataset']
        changed_details = item['detailed_changed_items']['base_wrong_finetuned_correct']

        for detail in changed_details:
            index = detail['index']
            base_item = detail['base_item']
            finetuned_item = detail['finetuned_item']

            answer_info = {
                "dataset": dataset_name,
                "question_index": index,
                "prompt": "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n" + base_item.get('prompt', ''),
                "label": base_item.get('label', ''),
                "correct_answer_text": finetuned_item.get('response', ''),  # use finetuned response
                "before_rlvr_correct": base_item.get('correct', False),
                "after_rlvr_correct": finetuned_item.get('correct', True)
            }

            if 'response' in base_item:
                answer_info["before_response"] = base_item['response']

            correct_answers.append(answer_info)

    json_data = {
        "config": config,
        "total_count": len(correct_answers),
        "correct_answers": correct_answers
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"Saved improved question details to {output_file}")


def find_matching_files(base_pattern: str, finetuned_pattern: str, search_dir: str = ".") -> list:
    """Find matching file pairs"""
    base_files = []
    finetuned_files = []

    for file in os.listdir(search_dir):
        if file.endswith('.json'):
            if base_pattern in file:
                base_files.append(os.path.join(search_dir, file))
            elif finetuned_pattern in file:
                finetuned_files.append(os.path.join(search_dir, file))

    file_pairs = []
    for base_file in base_files:
        base_name = os.path.basename(base_file)
        base_parts = base_name.split('_')
        if len(base_parts) >= 2:
            dataset_name = '_'.join(base_parts[1:]).replace('.json', '')

            for finetuned_file in finetuned_files:
                finetuned_name = os.path.basename(finetuned_file)
                if dataset_name in finetuned_name:
                    file_pairs.append((base_file, finetuned_file, dataset_name))
                    break

    return file_pairs


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results between base and finetuned models")
    parser.add_argument("--base_pattern", type=str, default="OLMo-2-1124-7B",
                        help="Pattern contained in base model filename (e.g., Qwen2.5-Math-7B)")
    parser.add_argument("--finetuned_pattern", type=str, default="olmo-150",
                        help="Pattern contained in finetuned model filename (e.g., rethink_rlvr_reproduce)")
    parser.add_argument("--dir", type=str, default="../outputs/olmo_outputs",
                        help="Directory to search for files")
    parser.add_argument("--output_csv", type=str, default="improved_questions_olmo.csv",
                        help="Output CSV file path for questions improved after finetuning")
    parser.add_argument("--output_json", type=str, default="correct_answers_data_olmo.json",
                        help="Output JSON file path for detailed improved questions")

    args = parser.parse_args()

    file_pairs = find_matching_files(args.base_pattern, args.finetuned_pattern, args.dir)

    if not file_pairs:
        print("No matching file pairs found")
        return

    print(f"Found {len(file_pairs)} dataset comparison pairs")

    total_base_correct_finetuned_wrong = 0
    total_base_wrong_finetuned_correct = 0
    total_both_correct = 0
    total_both_wrong = 0
    total_questions = 0

    all_changed_items = []
    all_detailed_changed_items = []

    for base_file, finetuned_file, dataset_name in file_pairs:
        try:
            base_result = load_result(base_file)
            finetuned_result = load_result(finetuned_file)

            comparison = compare_models(base_result, finetuned_result)

            print(f"\nDataset: {dataset_name}")
            print(f"Total questions: {comparison['total']}")
            print(f"Base accuracy: {comparison['base_acc']:.4f} ({comparison['base_acc'] * 100:.2f}%)")
            print(f"Finetuned accuracy: {comparison['finetuned_acc']:.4f} ({comparison['finetuned_acc'] * 100:.2f}%)")
            print(f"Base correct but finetuned wrong: {comparison['base_correct_finetuned_wrong']} items")
            print(f"Base wrong but finetuned correct: {comparison['base_wrong_finetuned_correct']} items")
            print(f"Both correct: {comparison['both_correct']} items")
            print(f"Both wrong: {comparison['both_wrong']} items")

            total_base_correct_finetuned_wrong += comparison['base_correct_finetuned_wrong']
            total_base_wrong_finetuned_correct += comparison['base_wrong_finetuned_correct']
            total_both_correct += comparison['both_correct']
            total_both_wrong += comparison['both_wrong']
            total_questions += comparison['total']

            all_changed_items.append({
                'dataset': dataset_name,
                'changed_items': comparison['changed_items']
            })

            all_detailed_changed_items.append({
                'dataset': dataset_name,
                'changed_items': comparison['changed_items'],
                'detailed_changed_items': comparison['detailed_changed_items']
            })

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

    if total_questions > 0:
        print("\n" + "=" * 60)
        print("Overall statistics:")
        print(f"Total questions: {total_questions}")
        total_base_acc = (total_both_correct + total_base_wrong_finetuned_correct) / total_questions
        total_finetuned_acc = (total_both_correct + total_base_correct_finetuned_wrong) / total_questions
        print(f"Base overall accuracy: {total_base_acc:.4f} ({total_base_acc * 100:.2f}%)")
        print(f"Finetuned overall accuracy: {total_finetuned_acc:.4f} ({total_finetuned_acc * 100:.2f}%)")
        print(f"Base correct but finetuned wrong: {total_base_correct_finetuned_wrong} items")
        print(f"Base wrong but finetuned correct: {total_base_wrong_finetuned_correct} items")
        print(f"Both correct: {total_both_correct} items")
        print(f"Both wrong: {total_both_wrong} items")

    config = {
        "before_rlvr_dir": args.dir,
        "after_rlvr_dir": args.dir,
        "before_model_prefix": args.base_pattern,
        "after_model_prefix": args.finetuned_pattern,
        "output_file": args.output_json,
        "datasets": [item['dataset'] for item in all_changed_items]
    }

    save_changed_questions_to_csv(all_changed_items, args.output_csv)
    save_correct_answers_json(all_detailed_changed_items, config, args.output_json)


if __name__ == "__main__":
    main()