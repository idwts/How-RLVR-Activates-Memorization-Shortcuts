"""
Neural Differential Equations (NDEs) for Modeling LLM Hidden State Dynamics

Core objectives:
1.Capture hidden state evolution throughout the complete generation process
2.Evaluate knowledge leakage (memorization vs generalization)
3.Locate critical layers (which layer has the most impact)

Simplified version:  Remove complex pooling mechanisms, focus on core NDEs modeling
"""

import os
import json
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List, Dict, Tuple
from torchdiffeq import odeint_adjoint as odeint


# Configuration
CONFIG = {
    'gen_csv': 'comparison_results_incorrect/generalization_samples.csv',
    'leak_csv': 'comparison_results_incorrect/leak_samples.csv',
    'model_path': '../rethink_rlvr_reproduce-incorrect-qwen2.5_math_7b-lr5e-7-kl0.00-step150',
    'output_dir': 'ndes_latent_dynamics',
    
    'pooling_method': 'mean',  # mean, last, max
    'use_pca': False,
    'pca_dim': 512,
    'latent_dim': 128,
    'batch_size': 16,
    'num_epochs':  50,
    'learning_rate':  1e-3,
    'mode': 'all',  # extract, train, analyze, all
    
    'max_length': 2048,
    'ode_hidden_dim': 256,
    'random_seed': 42,
}

BENCHMARKS = {
    "MATH-500": "../data/MATH-TTT/test.json",
    "AIME-2024": "../data/AIME-TTT/test.json",
    "AIME-2025": "../data/AIME2025-TTT/test.json",
    "AMC": "../data/AMC-TTT/test.json",
    "LiveMathBench": "../data/LiveMathBench/livemathbench_2504_v2.json",
    "MinervaMath": "../data/MinervaMath/minervamath.json",
}


# Set fixed random seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(CONFIG['random_seed'])


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
            'label': 0  # Generalization/Reasoning
        })

    leak_samples = []
    for _, row in leak_df.iterrows():
        leak_samples.append({
            'dataset': row['dataset'],
            'question_index': int(row['question_index']),
            'label': 1  # Memorization/Leakage
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


# Trajectory extraction:  Capture hidden state evolution

def extract_hidden_state_trajectories(
    model,
    tokenizer,
    samples: List[Dict],
    target_device: torch.device,
    max_length: int = 2048,
    pooling_method: str = "mean"
) -> List[Dict]:
    """
    Core function:  Extract hidden state trajectory for each layer
    
    Args:
        pooling_method: How to pool the seq_len dimension to fixed dimension
            - "mean": Average pooling (recommended)
            - "last": Take last token
            - "max": Max pooling
    
    Returns:
        [
            {
                'label': 0/1,
                'trajectory': [num_layers, hidden_dim],  # Hidden state of each layer
                'dataset': str,
                'question_index':  int
            }
        ]
    """
    print("\n" + "=" * 60)
    print(f"Extracting hidden state trajectories (pooling method: {pooling_method})...")
    print("=" * 60)

    trajectories_data = []
    skipped_count = 0

    for sample in tqdm(samples, desc="Extracting trajectories"):
        try:
            # 1.Load question
            question_data = load_question_data(sample['dataset'], sample['question_index'])
            prompt = question_data.get("prompt", "")

            # 2.Get answer
            answer = None
            for answer_key in ["answer", "solution", "output", "ground_truth"]:
                if answer_key in question_data: 
                    answer = question_data[answer_key]
                    break

            if answer is None or str(answer).strip() == "":
                skipped_count += 1
                continue

            # 3.Build full text (prompt + answer)
            full_text = prompt + "\nAnswer: " + str(answer)

            # 4.Tokenize
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs.input_ids.to(target_device)
            attention_mask = inputs.attention_mask.to(target_device)

            # 5.Capture hidden states for each layer
            layer_hidden_states = []

            def create_hook(layer_idx):
                def hook(module, input, output):
                    # output might be tuple, take first element
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output

                    # hidden_states:  [batch, seq_len, hidden_dim]
                    # Pooling to fixed dimension
                    if pooling_method == "mean":
                        # Average pooling (ignore padding)
                        if attention_mask is not None:
                            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                            pooled = sum_hidden / sum_mask
                        else:
                            pooled = hidden_states.mean(dim=1)

                    elif pooling_method == "last":
                        # Take last non-padding token
                        if attention_mask is not None:
                            seq_lengths = attention_mask.sum(dim=1) - 1
                            pooled = hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
                        else:
                            pooled = hidden_states[: , -1, :]

                    elif pooling_method == "max":
                        # Max pooling
                        pooled, _ = torch.max(hidden_states, dim=1)

                    else:
                        raise ValueError(f"Unknown pooling_method: {pooling_method}")

                    layer_hidden_states.append(pooled.detach().cpu())

                return hook

            # 6.Register hooks
            hooks = []

            # Embedding layer
            embed_hook = model.model.embed_tokens.register_forward_hook(create_hook(-1))
            hooks.append(embed_hook)

            # Transformer layers
            for i, layer in enumerate(model.model.layers):
                hook = layer.register_forward_hook(create_hook(i))
                hooks.append(hook)

            # 7.Forward pass
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            # Remove hooks
            for hook in hooks: 
                hook.remove()

            # 8.Organize trajectory data
            # trajectory: [num_layers+1, hidden_dim]
            trajectory = torch.cat(layer_hidden_states, dim=0).numpy()

            trajectories_data.append({
                'label': sample['label'],
                'trajectory':  trajectory,  # [num_layers+1, hidden_dim]
                'dataset': sample['dataset'],
                'question_index': sample['question_index']
            })

        except Exception as e:
            print(f"\nError:  {sample['dataset']}#{sample['question_index']}:  {e}")
            skipped_count += 1
            continue

    print(f"\nSuccess: {len(trajectories_data)} | Skipped: {skipped_count}")

    return trajectories_data


# NDEs model:  Modeling hidden state dynamics

class LatentDynamicsDataset(Dataset):
    """Hidden state trajectory dataset"""

    def __init__(self, trajectories_data: List[Dict], pca:  PCA = None, use_pca: bool = False):
        self.use_pca = use_pca
        self.pca = pca

        self.trajectories = []
        self.labels = []

        for item in trajectories_data:
            traj = item['trajectory']  # [num_layers, hidden_dim]

            if self.use_pca and self.pca is not None:
                traj = self.pca.transform(traj)

            self.trajectories.append(torch.FloatTensor(traj))
            self.labels.append(item['label'])

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx], self.labels[idx]


class ODEFunc(nn.Module):
    """
    Neural ODE function:  Define how hidden states evolve with layer depth
    
    dz/dt = f(z(t), t; theta)
    
    Where: 
    - z(t): Hidden state at time t
    - t: Layer depth (discrete index 0, 1, 2, ..., L)
    - f: Dynamics function parameterized by neural network
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super(ODEFunc, self).__init__()

        # Time embedding
        self.time_embed = nn.Linear(1, latent_dim)

        # Dynamics function network
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Initialize
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        """
        Args:
            t:  Scalar time (layer depth)
            z: [batch_size, latent_dim] Current hidden state
        
        Returns: 
            dz/dt: [batch_size, latent_dim] State change rate
        """
        batch_size = z.shape[0]

        # Time encoding
        t_vec = torch.ones(batch_size, 1).to(z.device) * t
        t_embed = self.time_embed(t_vec)

        # Dynamic evolution
        z_with_time = z + t_embed
        dz_dt = self.net(z_with_time)

        return dz_dt


class LatentDynamicsNDE(nn.Module):
    """
    Hidden state dynamics system NDEs model
    
    Architecture:
    1.Encoder: Initial few layers to initial latent state z0
    2.NDE: Continuous evolution dz/dt = f(z, t)
    3.Classifier: Final state zL to P(leakage)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 28,
        ode_hidden_dim: int = 256
    ):
        super(LatentDynamicsNDE, self).__init__()

        self.num_layers = num_layers
        self.latent_dim = latent_dim

        # Encoder:  First 3 layers to z0
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim)
        )

        # NDE function
        self.ode_func = ODEFunc(latent_dim, ode_hidden_dim)

        # Classifier: zL to P(leakage)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, trajectory):
        """
        Args: 
            trajectory: [batch_size, num_layers, input_dim]
        
        Returns:
            prob: [batch_size, 1] - P(leakage)
            z_trajectory: [num_layers, batch_size, latent_dim] - Complete evolution trajectory
        """
        batch_size = trajectory.shape[0]

        # 1.Encode initial state (first 3 layers)
        initial_states = trajectory[:, : 3, : ].reshape(batch_size, -1)
        z0 = self.encoder(initial_states)

        # 2.ODE solving:  Continuous evolution
        t = torch.linspace(0, self.num_layers - 1, self.num_layers).to(trajectory.device)

        z_trajectory = odeint(
            self.ode_func,
            z0,
            t,
            method='rk4',  # Runge-Kutta 4th order
            options={'step_size': 0.5}
        )
        # z_trajectory: [num_layers, batch_size, latent_dim]

        # 3.Classification
        zL = z_trajectory[-1]  # Final state
        prob = self.classifier(zL)

        return prob, z_trajectory


# Training

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: LatentDynamicsNDE,
    device: torch.device,
    num_epochs: int = 50,
    lr: float = 1e-3,
    save_dir: str = "checkpoints"
):
    """Train NDEs model"""

    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print("\n" + "=" * 60)
    print("Training hidden state dynamics system...")
    print("=" * 60)

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for trajectories, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            trajectories = trajectories.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            probs, _ = model(trajectories)
            loss = criterion(probs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predictions = (probs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for trajectories, labels in val_loader:
                trajectories = trajectories.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                probs, _ = model(trajectories)
                loss = criterion(probs, labels)

                val_loss += loss.item()
                predictions = (probs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}:  Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':  epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"  Saved best model (Val Acc:  {val_acc:.4f})")

    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")

    return history


# Analysis:  Evaluate knowledge leakage and locate critical layers

def analyze_leakage_and_critical_layers(
    model: LatentDynamicsNDE,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str = "analysis"
):
    """
    Core analysis: 
    1.Evaluate knowledge leakage (through trajectory separation)
    2.Locate critical layers (through layer-wise impact analysis)
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    # Collect all trajectories
    leak_trajectories = []  # Leakage samples
    gen_trajectories = []   # Generalization samples

    print("\n" + "=" * 60)
    print("Extracting test set trajectories...")
    print("=" * 60)

    with torch.no_grad():
        for trajectories, labels in tqdm(test_loader, desc="Analyzing"):
            trajectories = trajectories.to(device)

            _, z_traj = model(trajectories)
            z_traj = z_traj.cpu().numpy()  # [num_layers, batch, latent_dim]

            for i, label in enumerate(labels):
                traj = z_traj[: , i, :]  # [num_layers, latent_dim]

                if label == 1: 
                    leak_trajectories.append(traj)
                else: 
                    gen_trajectories.append(traj)

    leak_avg = np.mean(leak_trajectories, axis=0)  # [num_layers, latent_dim]
    gen_avg = np.mean(gen_trajectories, axis=0)

    print(f"Leakage trajectories:  {len(leak_trajectories)}")
    print(f"Generalization trajectories: {len(gen_trajectories)}")

    # Analysis 1: Trajectory separation visualization (evaluate knowledge leakage)

    print("\n" + "=" * 60)
    print("Analysis 1: Trajectory separation (evaluate knowledge leakage degree)...")
    print("=" * 60)

    # PCA dimension reduction to 2D
    all_points = np.concatenate([leak_avg, gen_avg], axis=0)
    pca_2d = PCA(n_components=2)
    pca_2d.fit(all_points)

    leak_2d = pca_2d.transform(leak_avg)
    gen_2d = pca_2d.transform(gen_avg)

    # Calculate trajectory distance (evaluate separation degree)
    trajectory_distances = np.linalg.norm(leak_avg - gen_avg, axis=1)
    avg_distance = np.mean(trajectory_distances)
    max_distance = np.max(trajectory_distances)
    max_distance_layer = np.argmax(trajectory_distances)

    print(f"\nTrajectory separation:")
    print(f"  Average distance: {avg_distance:.4f}")
    print(f"  Max distance: {max_distance:.4f} (Layer {max_distance_layer})")

    # Plot trajectory separation diagram
    plt.figure(figsize=(12, 8))

    plt.plot(leak_2d[:, 0], leak_2d[:, 1],
             marker='o', linewidth=2.5, markersize=8,
             color='#E74C3C', alpha=0.8, label='Leaked (Memorization)')

    plt.plot(gen_2d[:, 0], gen_2d[:, 1],
             marker='s', linewidth=2.5, markersize=8,
             color='#3498DB', alpha=0.8, label='Generalized (Reasoning)')

    # Annotate start and end points
    num_layers = leak_2d.shape[0]
    for idx, label in [(0, 'L0'), (num_layers-1, f'L{num_layers-1}')]:
        plt.scatter(leak_2d[idx, 0], leak_2d[idx, 1],
                   s=200, c='red', marker='*', edgecolors='black', linewidths=2, zorder=5)
        plt.scatter(gen_2d[idx, 0], gen_2d[idx, 1],
                   s=200, c='blue', marker='*', edgecolors='black', linewidths=2, zorder=5)
        plt.text(leak_2d[idx, 0], leak_2d[idx, 1] + 0.15, label,
                fontsize=12, fontweight='bold', ha='center', va='bottom')
        plt.text(gen_2d[idx, 0], gen_2d[idx, 1] + 0.15, label,
                fontsize=12, fontweight='bold', ha='center', va='bottom')

    plt.xlabel("PC1", fontsize=20, fontweight='bold')
    plt.ylabel("PC2", fontsize=20, fontweight='bold')
    plt.title(f"Latent Space Trajectory:  Leakage vs Stable\n(Avg Distance: {avg_distance:.4f})",
             fontsize=25, fontweight='bold')
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_separation.svg'), format='svg', dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Saved:  {output_dir}/trajectory_separation.svg")


    # Analysis 2: Layer-wise impact analysis (locate critical layers)

    print("\n" + "=" * 60)
    print("Analysis 2: Layer-wise impact analysis (locate critical layers)...")
    print("=" * 60)

    # Method 1: Layer-wise distance difference (which layer has largest leakage vs generalization difference)
    layer_distances = np.linalg.norm(leak_avg - gen_avg, axis=1)

    # Method 2: Dynamic evolution velocity (which layer changes most dramatically)
    leak_velocities = np.linalg.norm(np.diff(leak_avg, axis=0), axis=1)
    gen_velocities = np.linalg.norm(np.diff(gen_avg, axis=0), axis=1)
    velocity_diff = np.abs(leak_velocities - gen_velocities)

    # Method 3: Separation force (calculated through NDE function)
    separation_forces = []
    leak_avg_torch = torch.FloatTensor(leak_avg).to(device)
    gen_avg_torch = torch.FloatTensor(gen_avg).to(device)

    with torch.no_grad():
        for t in range(num_layers):
            t_tensor = torch.tensor([float(t)]).to(device)

            z_leak_t = leak_avg_torch[t].unsqueeze(0)
            dz_leak_dt = model.ode_func(t_tensor, z_leak_t)

            z_gen_t = gen_avg_torch[t].unsqueeze(0)
            dz_gen_dt = model.ode_func(t_tensor, z_gen_t)

            force = torch.norm(dz_leak_dt - dz_gen_dt, p=2).item()
            separation_forces.append(force)

    # Comprehensive score (sum after normalization)
    norm_distances = layer_distances / (layer_distances.max() + 1e-9)
    norm_velocities = np.pad(velocity_diff, (0, 1), 'edge') / (velocity_diff.max() + 1e-9)
    norm_forces = np.array(separation_forces) / (np.max(separation_forces) + 1e-9)

    importance_scores = 0*norm_distances + norm_velocities + norm_forces

    # Find Top-K critical layers
    top_k = 5
    critical_layers = np.argsort(importance_scores)[: :-1][:top_k]

    print(f"\nTop-{top_k} critical layers:")
    for rank, layer_idx in enumerate(critical_layers, 1):
        print(f"  {rank}.Layer {layer_idx}:  Importance={importance_scores[layer_idx]:.4f}")
        print(f"      - Trajectory distance: {layer_distances[layer_idx]:.4f}")
        print(f"      - Separation force: {separation_forces[layer_idx]:.4f}")

    # Plot layer impact diagram
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    layers = range(num_layers)

    # Subplot 1: Layer-wise distance
    axes[0, 0].plot(layers, layer_distances, marker='o', linewidth=2, color='#9B59B6')
    axes[0, 0].scatter(critical_layers, layer_distances[critical_layers],
                      s=200, c='red', marker='*', edgecolors='black', linewidths=2, zorder=5)
    axes[0, 0].set_xlabel("Layer", fontsize=20, fontweight='bold')
    axes[0, 0].set_ylabel("Trajectory Distance", fontsize=20, fontweight='bold')
    axes[0, 0].set_title("Layer-wise Trajectory Distance", fontsize=25, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='both', labelsize=16)

    # Subplot 2: Separation force
    axes[0, 1].plot(layers, separation_forces, marker='s', linewidth=2, color='#E74C3C')
    axes[0, 1].scatter(critical_layers, [separation_forces[i] for i in critical_layers],
                      s=200, c='red', marker='*', edgecolors='black', linewidths=2, zorder=5)
    axes[0, 1].set_xlabel("Layer", fontsize=20, fontweight='bold')
    axes[0, 1].set_ylabel("Separation Force", fontsize=20, fontweight='bold')
    axes[0, 1].set_title("Layer-wise Separation Force", fontsize=25, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='both', labelsize=16)

    # Subplot 3: Evolution velocity difference
    axes[1, 0].plot(range(len(velocity_diff)), velocity_diff, marker='^', linewidth=2, color='#3498DB')
    axes[1, 0].set_xlabel("Layer", fontsize=20, fontweight='bold')
    axes[1, 0].set_ylabel("Velocity Difference", fontsize=20, fontweight='bold')
    axes[1, 0].set_title("Layer-wise Evolution Velocity Difference", fontsize=25, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='both', labelsize=16)

    # Subplot 4: Comprehensive importance score
    axes[1, 1].bar(layers, importance_scores, color='#2ECC71', alpha=0.7)
    axes[1, 1].bar(critical_layers, importance_scores[critical_layers],
                   color='#E74C3C', alpha=0.9, label='Critical Layers')
    axes[1, 1].set_xlabel("Layer", fontsize=20, fontweight='bold')
    axes[1, 1].set_ylabel("Importance Score", fontsize=20, fontweight='bold')
    axes[1, 1].set_title("Layer Importance (Normalized Sum)", fontsize=25, fontweight='bold')
    axes[1, 1].legend(fontsize=16)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_importance.svg'), format='svg', dpi=600, bbox_inches='tight')
    plt.close()

    # Create combined figure of plots 2 and 3 (modify star sorting logic)
    fig_combined, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 2: Separation force (stars sorted by separation force data)
    separation_force_top_layers = np.argsort(separation_forces)[::-1][:5]

    ax2.plot(layers, separation_forces, marker='s', linewidth=2, color='#E74C3C')
    ax2.scatter(separation_force_top_layers, [separation_forces[i] for i in separation_force_top_layers],
                s=200, c='red', marker='*', edgecolors='black', linewidths=2, zorder=5)
    ax2.set_xlabel("Layer", fontsize=20, fontweight='bold')
    ax2.set_ylabel("Separation Force", fontsize=20, fontweight='bold')
    ax2.set_title("Layer-wise Separation Force", fontsize=25, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=16)

    # Plot 3: Evolution velocity difference (stars sorted by evolution velocity difference data)
    velocity_diff_top_indices = np.argsort(velocity_diff)[::-1][:5]

    ax3.plot(range(len(velocity_diff)), velocity_diff, marker='^', linewidth=2, color='#3498DB')
    ax3.scatter(velocity_diff_top_indices, [velocity_diff[i] for i in velocity_diff_top_indices],
                s=200, c='red', marker='*', edgecolors='black', linewidths=2, zorder=5)
    ax3.set_xlabel("Layer", fontsize=20, fontweight='bold')
    ax3.set_ylabel("Velocity Difference", fontsize=20, fontweight='bold')
    ax3.set_title("Layer-wise Evolution Velocity Difference", fontsize=25, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_dynamics_combined.svg'), format='svg', dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Saved combined figure:  {output_dir}/layer_dynamics_combined.svg")
    print(f"Saved:  {output_dir}/layer_importance.svg")

    # Save analysis results
    results = {
        'trajectory_separation': {
            'avg_distance': float(avg_distance),
            'max_distance': float(max_distance),
            'max_distance_layer': int(max_distance_layer)
        },
        'critical_layers': {
            f'rank_{i+1}': {
                'layer':  int(layer_idx),
                'importance_score': float(importance_scores[layer_idx]),
                'trajectory_distance': float(layer_distances[layer_idx]),
                'separation_force': float(separation_forces[layer_idx])
            }
            for i, layer_idx in enumerate(critical_layers)
        },
        'layer_distances': layer_distances.tolist(),
        'separation_forces': separation_forces,
        'importance_scores': importance_scores.tolist()
    }

    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Saved:  {output_dir}/analysis_results.json")

    return results


# Main function

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device:  {device}")

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Stage 1: Load samples
    gen_samples, leak_samples = load_samples(CONFIG['gen_csv'], CONFIG['leak_csv'])
    all_samples = gen_samples + leak_samples

    # Stage 2: Extract trajectories
    trajectories_file = os.path.join(CONFIG['output_dir'], f'trajectories_{CONFIG["pooling_method"]}.pkl')

    # If hidden state file exists, load directly; otherwise extract and save
    if os.path.exists(trajectories_file):
        print(f"\nDetected existing hidden state file, loading directly:  {trajectories_file}")
        with open(trajectories_file, 'rb') as f:
            trajectories_data = pickle.load(f)
    elif CONFIG['mode'] in ['extract', 'all']:
        print("\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_path'], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG['model_path'],
            device_map="auto",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.eval()

        trajectories_data = extract_hidden_state_trajectories(
            model, tokenizer, all_samples, device,
            pooling_method=CONFIG['pooling_method']
        )

        with open(trajectories_file, 'wb') as f:
            pickle.dump(trajectories_data, f)
        print(f"Trajectories saved:  {trajectories_file}")

        del model
        torch.cuda.empty_cache()
    else:
        raise RuntimeError(f"Hidden state file {trajectories_file} not found, and extract/all mode not specified, cannot continue.")


    # Stage 3: Training
    if CONFIG['mode'] in ['train', 'analyze', 'all']:
        print(f"\nLoading trajectories:  {trajectories_file}")
        with open(trajectories_file, 'rb') as f:
            trajectories_data = pickle.load(f)

        # PCA dimension reduction (optional)
        pca = None
        if CONFIG['use_pca']:
            print(f"\nPCA dimension reduction to {CONFIG['pca_dim']} dimensions...")
            all_trajs = np.concatenate([item['trajectory'] for item in trajectories_data], axis=0)
            pca = PCA(n_components=CONFIG['pca_dim'])
            pca.fit(all_trajs)
            print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

        # Split dataset
        train_data, test_data = train_test_split(trajectories_data, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)

        print(f"\nDataset:  Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

        train_dataset = LatentDynamicsDataset(train_data, pca, CONFIG['use_pca'])
        val_dataset = LatentDynamicsDataset(val_data, pca, CONFIG['use_pca'])
        test_dataset = LatentDynamicsDataset(test_data, pca, CONFIG['use_pca'])

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

        # Create model
        input_dim = CONFIG['pca_dim'] if CONFIG['use_pca'] else trajectories_data[0]['trajectory'].shape[1]

        nde_model = LatentDynamicsNDE(
            input_dim=input_dim,
            latent_dim=CONFIG['latent_dim'],
            num_layers=trajectories_data[0]['trajectory'].shape[0]
        )

        checkpoint_dir = os.path.join(CONFIG['output_dir'], 'checkpoints')

        if CONFIG['mode'] in ['train', 'all']: 
            history = train_model(
                train_loader, val_loader, nde_model, device,
                num_epochs=CONFIG['num_epochs'], lr=CONFIG['learning_rate'], save_dir=checkpoint_dir
            )

        # Stage 4: Analysis
        if CONFIG['mode'] in ['analyze', 'all']:
            checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
            nde_model.load_state_dict(checkpoint['model_state_dict'])
            nde_model = nde_model.to(device)

            analysis_dir = os.path.join(CONFIG['output_dir'], 'analysis')
            results = analyze_leakage_and_critical_layers(
                nde_model, test_loader, device, analysis_dir
            )

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - Trajectory data:  {CONFIG['output_dir']}/trajectories_{CONFIG['pooling_method']}.pkl")
    print(f"  - Model weights: {CONFIG['output_dir']}/checkpoints/best_model.pt")
    print(f"  - Trajectory separation: {CONFIG['output_dir']}/analysis/trajectory_separation.png")
    print(f"  - Layer importance: {CONFIG['output_dir']}/analysis/layer_importance.png")
    print(f"  - Analysis results: {CONFIG['output_dir']}/analysis/analysis_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()