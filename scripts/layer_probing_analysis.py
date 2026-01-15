"""
Independent Probing Analysis
Train multiple independent classifiers to evaluate information content of each layer

Based on paper:  Detecting Memorization in Large Language Models
Core idea: Train 28 independent probes, each using only single layer hidden states
Locate which layer is most important for distinguishing "memorization vs generalization" by comparing accuracy
"""

import os
import json
import pickle
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')


# Configuration
CONFIG = {
    'gen_csv':  'comparison_results_incorrect/yielding_memory_samples.csv',
    'leak_csv': 'comparison_results_incorrect/leak_samples.csv',
    'model_path': '../rethink_rlvr_reproduce-incorrect-qwen2.5_math_7b-lr5e-7-kl0.00-step150',
    'output_dir': 'layer_probing',
    
    'pooling_method': 'mean',  # mean, last, max
    'probe_type': 'logistic',  # logistic (Logistic Regression - recommended), mlp (Small MLP)
    'mlp_epochs': 50,
    'mode': 'all',  # extract, probe, all
    
    'max_length': 2048,
}

BENCHMARKS = {
    "MATH-500": "../data/MATH-TTT/test.json",
    "AIME-2024": "../data/AIME-TTT/test.json",
    "AIME-2025": "../data/AIME2025-TTT/test.json",
    "AMC":  "../data/AMC-TTT/test.json",
    "LiveMathBench": "../data/LiveMathBench/livemathbench_2504_v2.json",
    "MinervaMath": "../data/MinervaMath/minervamath.json",
}


# Data preparation

def load_samples(gen_csv: str, leak_csv: str) -> Tuple[List[Dict], List[Dict]]:
    """Load generalization and leakage samples"""
    print("=" * 60)
    print("Loading sample data...")
    print("=" * 60)
    
    gen_df = pd.read_csv(gen_csv)
    leak_df = pd.read_csv(leak_csv)
    
    generalization_samples = []
    for _, row in gen_df.iterrows():
        generalization_samples.append({
            'dataset': row['dataset'],
            'question_index': int(row['question_index']),
            'label': 0  # Generalization
        })
    
    leak_samples = []
    for _, row in leak_df.iterrows():
        leak_samples.append({
            'dataset': row['dataset'],
            'question_index': int(row['question_index']),
            'label': 1  # Leakage
        })
    
    print(f"Generalization samples: {len(generalization_samples)}")
    print(f"Leakage samples: {len(leak_samples)}")
    
    return generalization_samples, leak_samples


def load_question_data(dataset_name: str, question_index: int) -> Dict:
    """Load question data"""
    if dataset_name not in BENCHMARKS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_path = BENCHMARKS[dataset_name]
    with open(dataset_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    return data[question_index]


# Feature extraction:  Hidden states for each layer

def extract_layer_hidden_states(
    model, 
    tokenizer, 
    samples: List[Dict],
    target_device: torch.device,
    max_length: int = 2048,
    pooling_method: str = "mean"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract hidden states for each layer as features
    
    Args: 
        pooling_method: mean, last, max
    
    Returns:
        features: [num_samples, num_layers, hidden_dim]
        labels: [num_samples]
    """
    print("\n" + "=" * 60)
    print(f"Extracting layer-wise hidden states (pooling:  {pooling_method})...")
    print("=" * 60)
    
    all_features = []
    all_labels = []
    skipped = 0
    
    for sample in tqdm(samples, desc="Extracting features"):
        try:
            # Load question
            question_data = load_question_data(sample['dataset'], sample['question_index'])
            prompt = question_data.get("prompt", "")
            
            # Get answer
            answer = None
            for answer_key in ["answer", "solution", "output", "ground_truth"]:
                if answer_key in question_data: 
                    answer = question_data[answer_key]
                    break
            
            if answer is None or str(answer).strip() == "":
                skipped += 1
                continue
            
            # Build full text
            full_text = prompt + "\nAnswer: " + str(answer)
            
            # Tokenize
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs.input_ids.to(target_device)
            attention_mask = inputs.attention_mask.to(target_device)
            
            # Capture hidden states for each layer
            layer_hidden_states = []
            
            def create_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    
                    # Pooling
                    if pooling_method == "mean":
                        if attention_mask is not None:
                            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                            pooled = sum_hidden / sum_mask
                        else:
                            pooled = hidden_states.mean(dim=1)
                    
                    elif pooling_method == "last":
                        if attention_mask is not None:
                            seq_lengths = attention_mask.sum(dim=1) - 1
                            pooled = hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
                        else:
                            pooled = hidden_states[: , -1, :]
                    
                    elif pooling_method == "max":
                        pooled, _ = torch.max(hidden_states, dim=1)
                    
                    layer_hidden_states.append(pooled.detach().cpu().numpy())
                
                return hook
            
            # Register hooks
            hooks = []
            
            # Embedding layer
            embed_hook = model.model.embed_tokens.register_forward_hook(create_hook(-1))
            hooks.append(embed_hook)
            
            # Transformer layers
            for i, layer in enumerate(model.model.layers):
                hook = layer.register_forward_hook(create_hook(i))
                hooks.append(hook)
            
            # Forward pass
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Remove hooks
            for hook in hooks: 
                hook.remove()
            
            # Organize features:  [num_layers, hidden_dim]
            features = np.concatenate(layer_hidden_states, axis=0)
            
            all_features.append(features)
            all_labels.append(sample['label'])
        
        except Exception as e: 
            print(f"\nError:  {sample['dataset']}#{sample['question_index']}:  {e}")
            skipped += 1
            continue
    
    print(f"\nSuccess:  {len(all_features)} | Skipped: {skipped}")
    
    # Convert to numpy arrays
    features = np.array(all_features)  # [num_samples, num_layers, hidden_dim]
    labels = np.array(all_labels)      # [num_samples]
    
    print(f"Feature shape: {features.shape}")
    print(f"Label shape:  {labels.shape}")
    
    return features, labels


# Probe definition

class MLPProbe(nn.Module):
    """
    MLP probe (optional)
    
    Simple multi-layer perceptron for classification
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(MLPProbe, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


def train_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = torch.device('cpu')
) -> Tuple[MLPProbe, Dict]:
    """Train MLP probe"""
    
    # Create dataset
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = MLPProbe(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_preds = []
        
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                val_preds.append(preds.cpu().numpy())
        
        val_preds = np.concatenate(val_preds).flatten()
        val_acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        train_preds = model(torch.FloatTensor(X_train).to(device)).cpu().numpy().flatten()
        val_preds = model(torch.FloatTensor(X_val).to(device)).cpu().numpy().flatten()
    
    metrics = {
        'train_acc': accuracy_score(y_train, (train_preds > 0.5).astype(int)),
        'val_acc': accuracy_score(y_val, (val_preds > 0.5).astype(int)),
        'val_auc': roc_auc_score(y_val, val_preds),
        'val_f1': f1_score(y_val, (val_preds > 0.5).astype(int))
    }
    
    return model, metrics


# Independent probe training

def train_independent_probes(
    features: np.ndarray,
    labels: np.ndarray,
    probe_type: str = "logistic",
    mlp_epochs: int = 50,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Core function:  Train independent probes
    
    Train an independent classifier for each layer
    
    Args:
        features: [num_samples, num_layers, hidden_dim]
        labels:  [num_samples]
        probe_type: "logistic" (Logistic Regression) or "mlp" (Small MLP)
    
    Returns:
        results:  Evaluation results for each layer
    """
    num_samples, num_layers, hidden_dim = features.shape
    
    print("\n" + "=" * 60)
    print(f"Training independent probes (type: {probe_type})...")
    print(f"Layers: {num_layers}, Feature dim: {hidden_dim}")
    print("=" * 60)
    
    # Split dataset
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    # Train probe for each layer
    results = {
        'layers': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': [],
        'test_auc': [],
        'test_f1': [],
        'probes': []
    }
    
    for layer_idx in tqdm(range(num_layers), desc="Training probes"):
        # Extract features for this layer
        X_train_layer = X_train[:, layer_idx, :]  # [num_train, hidden_dim]
        X_val_layer = X_val[: , layer_idx, :]
        X_test_layer = X_test[:, layer_idx, :]
        
        if probe_type == "logistic": 
            # Logistic Regression probe (most commonly used)
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_layer)
            X_val_scaled = scaler.transform(X_val_layer)
            X_test_scaled = scaler.transform(X_test_layer)
            
            # Train logistic regression
            probe = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
            probe.fit(X_train_scaled, y_train)
            
            # Predict
            train_preds = probe.predict(X_train_scaled)
            val_preds = probe.predict(X_val_scaled)
            test_preds = probe.predict(X_test_scaled)
            test_probs = probe.predict_proba(X_test_scaled)[:, 1]
            
            # Evaluate
            train_acc = accuracy_score(y_train, train_preds)
            val_acc = accuracy_score(y_val, val_preds)
            test_acc = accuracy_score(y_test, test_preds)
            test_auc = roc_auc_score(y_test, test_probs)
            test_f1 = f1_score(y_test, test_preds)
            
            results['probes'].append((probe, scaler))
        
        elif probe_type == "mlp":
            # MLP probe
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_layer)
            X_val_scaled = scaler.transform(X_val_layer)
            X_test_scaled = scaler.transform(X_test_layer)
            
            # Train MLP
            probe, train_metrics = train_mlp_probe(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                input_dim=hidden_dim,
                num_epochs=mlp_epochs,
                device=device
            )
            
            # Test
            probe.eval()
            with torch.no_grad():
                test_probs = probe(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy().flatten()
            test_preds = (test_probs > 0.5).astype(int)
            
            train_acc = train_metrics['train_acc']
            val_acc = train_metrics['val_acc']
            test_acc = accuracy_score(y_test, test_preds)
            test_auc = roc_auc_score(y_test, test_probs)
            test_f1 = f1_score(y_test, test_preds)
            
            results['probes'].append((probe, scaler))
        
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}")
        
        # Record results
        results['layers'].append(layer_idx)
        results['train_acc'].append(train_acc)
        results['val_acc'].append(val_acc)
        results['test_acc'].append(test_acc)
        results['test_auc'].append(test_auc)
        results['test_f1'].append(test_f1)
        
        # Print results every 5 layers
        if (layer_idx + 1) % 5 == 0:
            print(f"\n  Layer {layer_idx}:  Test Acc={test_acc:.4f}, AUC={test_auc:.4f}")
    
    return results


# Visualization and analysis

def visualize_probing_results(results: Dict, output_dir: str = "probing_analysis"):
    """
    Visualize probe results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    layers = results['layers']
    train_acc = results['train_acc']
    val_acc = results['val_acc']
    test_acc = results['test_acc']
    test_auc = results['test_auc']
    test_f1 = results['test_f1']
    
    # Find best layers
    best_layer_acc = np.argmax(test_acc)
    best_layer_auc = np.argmax(test_auc)
    
    print("\n" + "=" * 60)
    print("Probe Analysis Results")
    print("=" * 60)
    print(f"\nBest layer (by accuracy): Layer {best_layer_acc}")
    print(f"   Test Accuracy: {test_acc[best_layer_acc]:.4f}")
    print(f"   Test AUC: {test_auc[best_layer_acc]:.4f}")
    
    print(f"\nBest layer (by AUC): Layer {best_layer_auc}")
    print(f"   Test Accuracy: {test_acc[best_layer_auc]:.4f}")
    print(f"   Test AUC: {test_auc[best_layer_auc]:.4f}")
    
    # Find Top-5 layers
    top5_layers = np.argsort(test_acc)[::-1][:5]
    print(f"\nTop-5 layers (by accuracy):")
    for rank, layer_idx in enumerate(top5_layers, 1):
        print(f"   {rank}.Layer {layer_idx}:  Acc={test_acc[layer_idx]:.4f}, AUC={test_auc[layer_idx]:.4f}")
    
    # Visualization 1: Accuracy curves
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Train/Validation/Test accuracy
    axes[0, 0].plot(layers, train_acc, marker='o', label='Train', linewidth=2, alpha=0.7)
    axes[0, 0].plot(layers, val_acc, marker='s', label='Validation', linewidth=2, alpha=0.7)
    axes[0, 0].plot(layers, test_acc, marker='^', label='Test', linewidth=2.5, alpha=0.9)
    axes[0, 0].scatter([best_layer_acc], [test_acc[best_layer_acc]], 
                      s=300, c='red', marker='*', edgecolors='black', linewidths=2, zorder=5)
    axes[0, 0].set_xlabel("Layer", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    axes[0, 0].set_title("Layer-wise Probe Accuracy", fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    
    # Subplot 2: AUC-ROC curve
    axes[0, 1].plot(layers, test_auc, marker='o', linewidth=2.5, color='#E74C3C')
    axes[0, 1].scatter([best_layer_auc], [test_auc[best_layer_auc]],
                      s=300, c='red', marker='*', edgecolors='black', linewidths=2, zorder=5)
    axes[0, 1].set_xlabel("Layer", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("AUC-ROC", fontsize=12, fontweight='bold')
    axes[0, 1].set_title("Layer-wise AUC-ROC Score", fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Subplot 3: F1 Score
    axes[1, 0].plot(layers, test_f1, marker='s', linewidth=2.5, color='#3498DB')
    axes[1, 0].set_xlabel("Layer", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("F1 Score", fontsize=12, fontweight='bold')
    axes[1, 0].set_title("Layer-wise F1 Score", fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: Top-5 layers comparison
    top5_indices = top5_layers
    metrics_names = ['Accuracy', 'AUC', 'F1']
    metrics_values = [
        [test_acc[i] for i in top5_indices],
        [test_auc[i] for i in top5_indices],
        [test_f1[i] for i in top5_indices]
    ]
    
    x = np.arange(len(top5_indices))
    width = 0.25
    
    for i, (name, values) in enumerate(zip(metrics_names, metrics_values)):
        axes[1, 1].bar(x + i * width, values, width, label=name, alpha=0.8)
    
    axes[1, 1].set_xlabel("Top-5 Layers", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("Score", fontsize=12, fontweight='bold')
    axes[1, 1].set_title("Top-5 Layers Comparison", fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels([f"L{i}" for i in top5_indices])
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probing_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved:  {output_dir}/probing_results.png")
    
    # Visualization 2: Layer-wise information gain
    
    # Calculate improvement relative to random classification
    random_acc = 0.5
    info_gain = np.array(test_acc) - random_acc
    
    plt.figure(figsize=(12, 6))
    
    colors = ['#2ECC71' if gain > 0.3 else '#3498DB' if gain > 0.2 else '#95A5A6' 
              for gain in info_gain]
    
    plt.bar(layers, info_gain, color=colors, alpha=0.8, edgecolor='black')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='High Info Threshold')
    
    # Annotate best layer
    plt.scatter([best_layer_acc], [info_gain[best_layer_acc]],
               s=300, c='red', marker='*', edgecolors='black', linewidths=2, zorder=5)
    plt.text(best_layer_acc, info_gain[best_layer_acc], f'L{best_layer_acc}',
            fontsize=12, fontweight='bold', ha='center', va='bottom')
    
    plt.xlabel("Layer", fontsize=14, fontweight='bold')
    plt.ylabel("Information Gain (vs Random)", fontsize=14, fontweight='bold')
    plt.title("Layer-wise Information Gain for Leakage Detection", fontsize=16, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'information_gain.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir}/information_gain.png")
    
    # Save results
    results_summary = {
        'best_layer_acc': int(best_layer_acc),
        'best_layer_auc': int(best_layer_auc),
        'best_test_acc': float(test_acc[best_layer_acc]),
        'best_test_auc': float(test_auc[best_layer_auc]),
        'top5_layers': [int(i) for i in top5_layers],
        'layer_metrics': {
            'layers': [int(i) for i in layers],
            'train_acc': [float(x) for x in train_acc],
            'val_acc': [float(x) for x in val_acc],
            'test_acc': [float(x) for x in test_acc],
            'test_auc': [float(x) for x in test_auc],
            'test_f1': [float(x) for x in test_f1],
            'info_gain': [float(x) for x in info_gain]
        }
    }
    
    with open(os.path.join(output_dir, 'probing_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"Saved: {output_dir}/probing_results.json")
    
    return results_summary


# Main function

def main():
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    print(f"Using device:  {device}")
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Stage 1: Load samples
    gen_samples, leak_samples = load_samples(CONFIG['gen_csv'], CONFIG['leak_csv'])
    all_samples = gen_samples + leak_samples
    
    # Stage 2: Extract features
    features_file = os.path.join(CONFIG['output_dir'], f'features_{CONFIG["pooling_method"]}.pkl')
    
    if CONFIG['mode'] in ['extract', 'all']:
        print("\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_path'], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG['model_path'],
            device_map="auto",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.eval()
        
        features, labels = extract_layer_hidden_states(
            model, tokenizer, all_samples, device,
            pooling_method=CONFIG['pooling_method']
        )
        
        with open(features_file, 'wb') as f:
            pickle.dump({'features': features, 'labels':  labels}, f)
        print(f"Features saved:  {features_file}")
        
        del model
        torch.cuda.empty_cache()
    
    # Stage 3: Train probes
    if CONFIG['mode'] in ['probe', 'all']:
        print(f"\nLoading features:  {features_file}")
        with open(features_file, 'rb') as f:
            data = pickle.load(f)
            features = data['features']
            labels = data['labels']
        
        # Train independent probes
        results = train_independent_probes(
            features, labels,
            probe_type=CONFIG['probe_type'],
            mlp_epochs=CONFIG['mlp_epochs'],
            device=device
        )
        
        # Save probes
        probes_file = os.path.join(CONFIG['output_dir'], f'probes_{CONFIG["probe_type"]}.pkl')
        with open(probes_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nProbes saved: {probes_file}")
        
        # Visualize analysis
        analysis_dir = os.path.join(CONFIG['output_dir'], 'analysis')
        summary = visualize_probing_results(results, analysis_dir)
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - Feature data: {CONFIG['output_dir']}/features_{CONFIG['pooling_method']}.pkl")
    print(f"  - Probe models: {CONFIG['output_dir']}/probes_{CONFIG['probe_type']}.pkl")
    print(f"  - Probe results: {CONFIG['output_dir']}/analysis/probing_results.png")
    print(f"  - Information gain: {CONFIG['output_dir']}/analysis/information_gain.png")
    print(f"  - Analysis results: {CONFIG['output_dir']}/analysis/probing_results.json")
    print("=" * 60)


if __name__ == "__main__": 
    main()