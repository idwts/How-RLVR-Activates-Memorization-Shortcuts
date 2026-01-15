"""
Jensen-Shannon Divergence (JSD) Analysis for Feedforward Network Components

Experimental Design:
- Control Gate and vary Up:  measures the marginal contribution of Up (content)
- Control Up and vary Gate: measures the marginal contribution of Gate (gating)
- For Down, use a random initialized matrix with the same norm to replace the original Down projection
"""

import os
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.spatial.distance import jensenshannon


# Configuration
CONFIG = {
    'input_csv': 'all_improved_questions.csv',
    'model_path': '../rethink_rlvr_reproduce-incorrect-qwen2.5_math_7b-lr5e-7-kl0.00-step150',
    'target_device': 'cuda: 0',
    'dtype': torch.float32,
    'output_dir': './',
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


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj


def load_csv_questions(file_path):
    """Load questions from CSV file and return first 3"""
    selected_questions = []
    with open(file_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader: 
            dataset = row['dataset']
            question_index = int(row['question_index'])
            selected_questions.append((dataset, question_index))
    return selected_questions[:3]


def load_original_question(dataset_name, question_index):
    """Load a single question from the specified dataset"""
    if dataset_name not in BENCHMARKS:
        raise ValueError(f"Dataset {dataset_name} is not found in BENCHMARKS.")
    dataset_path = BENCHMARKS[dataset_name]
    with open(dataset_path, "r") as f:
        data = json.load(f)
        question_data = data[question_index]
    return question_data


def calculate_jsd(p, q):
    """Calculate Jensen-Shannon divergence between two distributions"""
    p = np.array(p).flatten()
    q = np.array(q).flatten()
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return jensenshannon(p, q)


def get_attention_head_outputs_correct(model, input_ids, attention_mask, target_device):
    """
    Correctly capture the original output of each attention head (before o_proj)

    Returns:
        List[Dict]: Data for each layer, including: 
            - 'input':  input to the attention layer [batch, seq_len, hidden_dim]
            - 'head_outputs': outputs of each head List[Tensor], each [batch, seq_len, head_dim]
    """
    attention_head_data = []
    layer_inputs = []

    def create_layer_pre_hook(layer_idx):
        """Capture hidden states before entering the layer"""
        def hook(module, input):
            if isinstance(input, tuple) and len(input) > 0:
                hidden_states = input[0].detach().clone().to(target_device)
                if layer_idx == len(layer_inputs):
                    layer_inputs.append(hidden_states)
        return hook

    def create_attention_hook(layer_idx, num_heads, head_dim):
        def hook(module, input, output):
            if layer_idx < len(layer_inputs):
                attn_input = layer_inputs[layer_idx]

                with torch.no_grad():
                    hidden_states = attn_input
                    bsz, q_len, _ = hidden_states.shape

                    # Compute Q, K, V projections
                    query_states = module.q_proj(hidden_states)
                    key_states = module.k_proj(hidden_states)
                    value_states = module.v_proj(hidden_states)

                    # Calculate actual number of heads and dimensions from projection output features
                    q_proj_out_features = module.q_proj.out_features
                    k_proj_out_features = module.k_proj.out_features

                    actual_num_heads = q_proj_out_features // head_dim
                    actual_num_kv_heads = k_proj_out_features // head_dim
                    num_key_value_groups = actual_num_heads // actual_num_kv_heads

                    q_head_dim = q_proj_out_features // actual_num_heads
                    kv_head_dim = k_proj_out_features // actual_num_kv_heads

                    # Reshape
                    query_states = query_states.view(bsz, q_len, actual_num_heads, q_head_dim).transpose(1, 2)
                    key_states = key_states.view(bsz, q_len, actual_num_kv_heads, kv_head_dim).transpose(1, 2)
                    value_states = value_states.view(bsz, q_len, actual_num_kv_heads, kv_head_dim).transpose(1, 2)

                    # Expand KV to match Q's number of heads
                    if num_key_value_groups > 1:
                        key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
                        value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

                    # Calculate attention scores
                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / np.sqrt(q_head_dim)

                    # Apply attention mask (ensure device consistency)
                    if attention_mask is not None:
                        attn_device = attn_weights.device
                        mask = attention_mask.to(attn_device)

                        if mask.dim() == 2:
                            expanded_mask = mask.unsqueeze(1).unsqueeze(2)
                        elif mask.dim() == 4:
                            expanded_mask = mask
                        else:
                            expanded_mask = mask.view(bsz, 1, 1, -1)

                        expanded_mask = expanded_mask.to(attn_device)
                        attn_weights = attn_weights + expanded_mask

                    # Softmax
                    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                        query_states.dtype)

                    # Apply attention to V
                    attn_output = torch.matmul(attn_weights, value_states)

                    # Separate outputs for each head
                    head_outputs = []
                    for head_idx in range(actual_num_heads):
                        head_out = attn_output[:, head_idx, :, : ].contiguous()
                        head_outputs.append(head_out.to(target_device))

                    layer_data = {
                        'input':  attn_input,
                        'head_outputs': head_outputs,
                        'num_heads': actual_num_heads,
                        'head_dim': kv_head_dim
                    }

                    if layer_idx == len(attention_head_data):
                        attention_head_data.append(layer_data)

        return hook

    # Register hooks
    hooks = []
    model_layers = model.model.layers

    config = model.config
    num_heads = config.num_attention_heads
    hidden_dim = config.hidden_size
    head_dim = hidden_dim // num_heads

    for i, layer in enumerate(model_layers):
        layer_pre_hook = layer.register_forward_pre_hook(create_layer_pre_hook(i))
        hooks.append(layer_pre_hook)

        attn_hook = layer.self_attn.register_forward_hook(
            create_attention_hook(i, num_heads, head_dim)
        )
        hooks.append(attn_hook)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Remove hooks
    for hook in hooks: 
        hook.remove()

    return attention_head_data


def get_module_io_states(model, input_ids, attention_mask, target_device):
    """
    Capture input and output of Attention and MLP for each layer using hook mechanism
    """
    attention_io = []
    mlp_io = []
    layer_residual_before_attn = []
    layer_residual_after_attn = []

    def create_layer_pre_hook(layer_idx):
        def hook(module, input):
            if isinstance(input, tuple) and len(input) > 0:
                hidden_states = input[0].detach().clone().to(target_device)
                if layer_idx == len(layer_residual_before_attn):
                    layer_residual_before_attn.append(hidden_states)
        return hook

    def create_attention_post_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                attn_output = output[0].detach().clone().to(target_device)
            else:
                attn_output = output.detach().clone().to(target_device)

            if layer_idx < len(layer_residual_before_attn):
                attn_input = layer_residual_before_attn[layer_idx]
                attention_io.append((attn_input, attn_output))

                if layer_idx == len(layer_residual_after_attn):
                    layer_residual_after_attn.append(attn_output)
        return hook

    def create_mlp_post_hook(layer_idx):
        def hook(module, input, output):
            mlp_output = output.detach().clone().to(target_device)
            if layer_idx < len(layer_residual_after_attn):
                mlp_input_residual = layer_residual_after_attn[layer_idx]
                mlp_io.append((mlp_input_residual, mlp_output))
        return hook

    hooks = []
    model_layers = model.model.layers

    for i, layer in enumerate(model_layers):
        layer_pre_hook = layer.register_forward_pre_hook(create_layer_pre_hook(i))
        hooks.append(layer_pre_hook)

        attn_post_hook = layer.self_attn.register_forward_hook(create_attention_post_hook(i))
        hooks.append(attn_post_hook)

        mlp_post_hook = layer.mlp.register_forward_hook(create_mlp_post_hook(i))
        hooks.append(mlp_post_hook)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    for hook in hooks:
        hook.remove()

    return attention_io, mlp_io


def get_mlp_counterfactual_states(model, input_ids, attention_mask, target_device):
    """
    Capture normal MLP output (O_normal) and three counterfactual outputs: 
    1.O_cf_gate (gating contribution): keep up_proj unchanged, permute gate_proj
    2.O_cf_up (content contribution): keep gate_proj unchanged, permute up_proj
    3.O_cf_down (projection contribution): keep gate*up unchanged, permute down_proj
    """
    mlp_cf_data = []
    mlp_inputs = []

    def create_mlp_pre_hook(layer_idx):
        def hook(module, input):
            if isinstance(input, tuple) and len(input) > 0:
                hidden_states = input[0].detach().clone().to(target_device)
                if layer_idx == len(mlp_inputs):
                    mlp_inputs.append(hidden_states)
        return hook

    def create_mlp_post_hook(layer_idx):
        def hook(module, input, output):
            O_normal = output.detach().clone().to(target_device)

            if layer_idx < len(mlp_inputs):
                x = mlp_inputs[layer_idx]

                with torch.no_grad():
                    # Recompute internal states
                    gate_output = module.gate_proj(x)
                    up_output = module.up_proj(x)
                    activated_gate = torch.nn.functional.silu(gate_output)
                    intermediate = activated_gate * up_output

                    # Counterfactual 1: O_cf_up (content contribution)
                    # Ablate up_output, keep gate_output
                    up_output_ablated = up_output.mean(dim=1, keepdim=True).expand_as(up_output)
                    intermediate_cf_up = activated_gate * up_output_ablated
                    O_cf_up = module.down_proj(intermediate_cf_up)

                    # Counterfactual 2: O_cf_gate (gating contribution)
                    # Ablate gate_output, keep up_output
                    gate_output_ablated = gate_output.mean(dim=1, keepdim=True).expand_as(gate_output)
                    activated_gate_ablated = torch.nn.functional.silu(gate_output_ablated)
                    intermediate_cf_gate = activated_gate_ablated * up_output
                    O_cf_gate = module.down_proj(intermediate_cf_gate)

                    # Counterfactual 3: O_cf_down (projection contribution)
                    # Use random initialized matrix with same norm to replace down_proj
                    down_proj_weight = module.down_proj.weight
                    random_weight = torch.randn_like(down_proj_weight)
                    random_weight = random_weight * (torch.norm(down_proj_weight) / torch.norm(random_weight))
                    O_cf_down = torch.nn.functional.linear(intermediate, random_weight, module.down_proj.bias)

                    layer_data = {
                        'O_normal': O_normal,
                        'O_cf_up': O_cf_up.detach().clone().to(target_device),
                        'O_cf_gate':  O_cf_gate.detach().clone().to(target_device),
                        'O_cf_down': O_cf_down.detach().clone().to(target_device)
                    }
                    if layer_idx == len(mlp_cf_data):
                        mlp_cf_data.append(layer_data)

        return hook

    hooks = []
    model_layers = model.model.layers
    for i, layer in enumerate(model_layers):
        hooks.append(layer.mlp.register_forward_pre_hook(create_mlp_pre_hook(i)))
        hooks.append(layer.mlp.register_forward_hook(create_mlp_post_hook(i)))

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    for hook in hooks: 
        hook.remove()

    return mlp_cf_data


def calculate_layer_jsd(hidden_state_p, hidden_state_q, lm_head_weight, target_device, attention_mask=None):
    """
    Calculate JSD between two hidden states P and Q using average over all tokens
    """
    hidden_state_p = hidden_state_p.to(target_device).detach()
    hidden_state_q = hidden_state_q.to(target_device).detach()
    lm_head_weight = lm_head_weight.to(target_device).detach()

    batch_size = hidden_state_p.shape[0]
    seq_len_p = hidden_state_p.shape[1]
    seq_len_q = hidden_state_q.shape[1]

    dim_p = hidden_state_p.shape[-1]
    dim_q = hidden_state_q.shape[-1]

    # Ensure sequence lengths match
    if seq_len_p != seq_len_q: 
        min_len = min(seq_len_p, seq_len_q)
        hidden_state_p = hidden_state_p[:, :min_len, :]
        hidden_state_q = hidden_state_q[: , :min_len, :]
        seq_len = min_len
    else:
        seq_len = seq_len_p

    # Ensure dimensions match
    if dim_p != dim_q:
        print(f"Warning: dimension mismatch {dim_p} vs {dim_q}. JSD may be meaningless.")

    # Reshape:  [batch * seq_len, dim]
    input_flat = hidden_state_p.reshape(-1, dim_p)
    output_flat = hidden_state_q.reshape(-1, dim_q)

    # Project to vocabulary space
    with torch.no_grad():
        logits_input = torch.matmul(input_flat, lm_head_weight.T).detach()
        logits_output = torch.matmul(output_flat, lm_head_weight.T).detach()

    # Convert to probability distributions
    prob_input = torch.softmax(logits_input, dim=-1).detach().cpu().numpy()
    prob_output = torch.softmax(logits_output, dim=-1).detach().cpu().numpy()

    # Use attention_mask to filter padding
    if attention_mask is not None:
        attention_mask = attention_mask.to(target_device)
        if attention_mask.shape[1] != seq_len:
            attention_mask = attention_mask[:, : seq_len]
        mask_flat = attention_mask.reshape(-1).cpu().numpy()
        valid_indices = mask_flat > 0
        if len(valid_indices) > 0:
            prob_input = prob_input[valid_indices]
            prob_output = prob_output[valid_indices]

        if len(prob_input) == 0:
            return 0.0

    # Calculate average JSD over all valid tokens
    jsd_scores = []
    for i in range(len(prob_input)):
        jsd = calculate_jsd(prob_input[i], prob_output[i])
        if not np.isnan(jsd):
            jsd_scores.append(jsd)

    if len(jsd_scores) == 0:
        return 0.0

    return np.mean(jsd_scores)


def calculate_attention_heads_jsd_correct(layer_data, model_layer, lm_head_weight, target_device, attention_mask=None):
    """
    Correctly calculate JSD score for each attention head
    """
    attn_input = layer_data['input'].to(target_device).detach()
    head_outputs = layer_data['head_outputs']
    num_heads = layer_data['num_heads']
    head_dim = layer_data['head_dim']

    hidden_dim = attn_input.shape[-1]

    o_proj_weight = model_layer.self_attn.o_proj.weight.detach().to(target_device)
    o_proj_per_head = o_proj_weight.view(hidden_dim, num_heads, head_dim)

    head_jsd_scores = []

    for head_idx, head_output in enumerate(head_outputs):
        batch_size, seq_len, _ = head_output.shape
        head_projection = o_proj_per_head[:, head_idx, :].contiguous()
        head_output_flat = head_output.reshape(-1, head_dim)
        head_projected = torch.matmul(head_output_flat, head_projection.T)
        head_projected = head_projected.view(batch_size, seq_len, hidden_dim)

        jsd = calculate_layer_jsd(
            attn_input, head_projected, lm_head_weight, target_device, attention_mask
        )
        head_jsd_scores.append(jsd)

    return head_jsd_scores


def analyze_model_jsd(model, input_ids, attention_mask, lm_head_weight, target_device):
    """
    Analyze JSD scores for Attention and MLP of each layer in the model
    """
    attention_io, mlp_io = get_module_io_states(model, input_ids, attention_mask, target_device)

    attention_jsd_scores = []
    mlp_jsd_scores = []

    for attn_input, attn_output in attention_io: 
        jsd = calculate_layer_jsd(attn_input, attn_output, lm_head_weight, target_device, attention_mask)
        attention_jsd_scores.append(jsd)

    for mlp_input, mlp_output in mlp_io:
        jsd = calculate_layer_jsd(mlp_input, mlp_output, lm_head_weight, target_device, attention_mask)
        mlp_jsd_scores.append(jsd)

    return mlp_jsd_scores, attention_jsd_scores


def analyze_attention_heads_jsd(model, input_ids, attention_mask, lm_head_weight, target_device):
    """
    Analyze JSD scores for each attention head in each layer using the corrected method
    """
    attention_head_data = get_attention_head_outputs_correct(model, input_ids, attention_mask, target_device)

    head_jsd_matrix = []
    model_layers = model.model.layers

    for layer_idx, layer_data in enumerate(attention_head_data):
        model_layer = model_layers[layer_idx]
        head_scores = calculate_attention_heads_jsd_correct(
            layer_data, model_layer, lm_head_weight, target_device, attention_mask
        )
        head_jsd_matrix.append(head_scores)

    return np.array(head_jsd_matrix)


def analyze_mlp_counterfactual_jsd(model, input_ids, attention_mask, lm_head_weight, target_device):
    """
    Analyze three counterfactual JSD metrics for MLP: 
    1.Gate JSD (gate_proj)
    2.Content JSD (up_proj)
    3.Projection JSD (down_proj)
    """
    mlp_cf_data = get_mlp_counterfactual_states(model, input_ids, attention_mask, target_device)

    up_jsd_scores = []
    gate_jsd_scores = []
    down_jsd_scores = []

    for layer_data in mlp_cf_data:
        O_normal = layer_data['O_normal']
        O_cf_up = layer_data['O_cf_up']
        O_cf_gate = layer_data['O_cf_gate']
        O_cf_down = layer_data['O_cf_down']

        # JSD(Normal, Ablated_Up) -> measures contribution of Up (up_proj)
        jsd_up = calculate_layer_jsd(
            O_normal, O_cf_up, lm_head_weight, target_device, attention_mask
        )

        # JSD(Normal, Ablated_Gate) -> measures contribution of Gate (gate_proj)
        jsd_gate = calculate_layer_jsd(
            O_normal, O_cf_gate, lm_head_weight, target_device, attention_mask
        )

        # JSD(Normal, Ablated_Down) -> measures contribution of Down (down_proj)
        jsd_down = calculate_layer_jsd(
            O_normal, O_cf_down, lm_head_weight, target_device, attention_mask
        )

        up_jsd_scores.append(jsd_up)
        gate_jsd_scores.append(jsd_gate)
        down_jsd_scores.append(jsd_down)

    return up_jsd_scores, gate_jsd_scores, down_jsd_scores


def visualize_jsd(mlp_jsd_averages, attention_jsd_averages, output_path="jsd_scores.png"):
    """
    Visualize average JSD scores for MLP and Attention layers
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    layers = range(len(mlp_jsd_averages))

    ax1.plot(layers, mlp_jsd_averages, label="MLP JSD", marker='o',
             linewidth=2.5, markersize=7, color='#FF6B6B', alpha=0.8)
    ax1.plot(layers, attention_jsd_averages, label="Attention JSD", marker='s',
             linewidth=2.5, markersize=7, color='#4ECDC4', alpha=0.8)
    ax1.set_xlabel("Layer Index", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Average JSD Score (All Tokens)", fontsize=13, fontweight='bold')
    ax1.set_title("MLP vs Attention JSD Scores", fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')

    width = 0.35
    x = np.arange(len(mlp_jsd_averages))
    ax2.bar(x - width / 2, mlp_jsd_averages, width, label='MLP JSD',
            alpha=0.8, color='#FF6B6B')
    ax2.bar(x + width / 2, attention_jsd_averages, width, label='Attention JSD',
            alpha=0.8, color='#4ECDC4')
    ax2.set_xlabel("Layer Index", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Average JSD Score (All Tokens)", fontsize=13, fontweight='bold')
    ax2.set_title("JSD Score Distribution", fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11)

    step = max(1, len(mlp_jsd_averages) // 10)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels(x[::step])
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to:  {output_path}")
    plt.close()


def visualize_attention_heads_jsd(head_jsd_matrix, output_path="attention_heads_jsd_heatmap.png"):
    """
    Visualize JSD scores for each attention head in each layer as heatmap
    """
    num_layers, num_heads = head_jsd_matrix.shape

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ax1 = axes[0]
    im = ax1.imshow(head_jsd_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax1.set_xlabel("Attention Head Index", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Layer Index", fontsize=13, fontweight='bold')
    ax1.set_title("Attention Head JSD Scores Heatmap", fontsize=15, fontweight='bold')
    ax1.set_xticks(np.arange(0, num_heads, max(1, num_heads // 10)))
    ax1.set_yticks(np.arange(0, num_layers, max(1, num_layers // 10)))
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('JSD Score', fontsize=11, fontweight='bold')

    ax2 = axes[1]
    layer_avg_jsd = np.mean(head_jsd_matrix, axis=1)
    head_avg_jsd = np.mean(head_jsd_matrix, axis=0)
    x = np.arange(num_layers)
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(x, layer_avg_jsd, marker='o', linewidth=2.5, markersize=6,
                     color='#FF6B6B', label='Avg JSD per Layer', alpha=0.8)
    ax2.set_xlabel("Layer Index", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Average JSD (across heads)", fontsize=11, fontweight='bold', color='#FF6B6B')
    ax2.tick_params(axis='y', labelcolor='#FF6B6B')
    ax2.grid(True, alpha=0.3, linestyle='--')
    x2 = np.arange(num_heads)
    line2 = ax2_twin.plot(x2, head_avg_jsd, marker='s', linewidth=2.5, markersize=6,
                          color='#4ECDC4', label='Avg JSD per Head', alpha=0.8)
    ax2_twin.set_ylabel("Average JSD (across layers)", fontsize=11, fontweight='bold', color='#4ECDC4')
    ax2_twin.tick_params(axis='y', labelcolor='#4ECDC4')
    ax2.set_title("Average JSD Scores", fontsize=15, fontweight='bold')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Attention head JSD heatmap saved to: {output_path}")
    plt.close()


def visualize_head_statistics(head_jsd_matrix, output_path="attention_heads_statistics.png"):
    """
    Visualize statistical information of attention heads
    """
    num_layers, num_heads = head_jsd_matrix.shape
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    max_heads = np.argmax(head_jsd_matrix, axis=1)
    min_heads = np.argmin(head_jsd_matrix, axis=1)
    ax1.scatter(range(num_layers), max_heads, c='red', marker='^', s=100, label='Max JSD Head', alpha=0.7)
    ax1.scatter(range(num_layers), min_heads, c='blue', marker='v', s=100, label='Min JSD Head', alpha=0.7)
    ax1.set_xlabel("Layer Index", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Head Index", fontsize=12, fontweight='bold')
    ax1.set_title("Heads with Max/Min JSD per Layer", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    layer_std = np.std(head_jsd_matrix, axis=1)
    ax2.bar(range(num_layers), layer_std, color='#9B59B6', alpha=0.7)
    ax2.set_xlabel("Layer Index", fontsize=12, fontweight='bold')
    ax2.set_ylabel("JSD Standard Deviation", fontsize=12, fontweight='bold')
    ax2.set_title("JSD Variance Across Heads per Layer", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = axes[1, 0]
    all_jsds = head_jsd_matrix.flatten()
    ax3.hist(all_jsds, bins=50, color='#3498DB', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(all_jsds), color='red', linestyle='--', linewidth=2, label=f'Mean:  {np.mean(all_jsds):.4f}')
    ax3.axvline(np.median(all_jsds), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(all_jsds):.4f}')
    ax3.set_xlabel("JSD Score", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax3.set_title("Distribution of All Head JSD Scores", fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    ax4 = axes[1, 1]
    k = 10
    flat_jsds = head_jsd_matrix.flatten()
    flat_indices = np.argsort(flat_jsds)
    top_k_indices = flat_indices[-k:][: :-1]
    bottom_k_indices = flat_indices[:k]
    top_k_values = flat_jsds[top_k_indices]
    bottom_k_values = flat_jsds[bottom_k_indices]
    x = np.arange(k)
    width = 0.35
    ax4.bar(x - width / 2, top_k_values, width, label=f'Top-{k} Highest JSD', color='#E74C3C', alpha=0.7)
    ax4.bar(x + width / 2, bottom_k_values, width, label=f'Top-{k} Lowest JSD', color='#2ECC71', alpha=0.7)
    ax4.set_xlabel("Rank", fontsize=12, fontweight='bold')
    ax4.set_ylabel("JSD Score", fontsize=12, fontweight='bold')
    ax4.set_title(f"Top-{k} Highest and Lowest JSD Heads", fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Attention head statistics saved to: {output_path}")
    plt.close()


def visualize_mlp_counterfactual_jsd(up_jsd_averages, gate_jsd_averages, down_jsd_averages, output_path="jsd_mlp_counterfactual.png"):
    """
    Visualize counterfactual JSD for three MLP components
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    layers = range(len(up_jsd_averages))

    ax1.plot(layers, up_jsd_averages, label="Content JSD (up_proj)", marker='o',
             linewidth=2.5, markersize=7, color='#3498DB', alpha=0.8)
    ax1.plot(layers, gate_jsd_averages, label="Gate JSD (gate_proj)", marker='s',
             linewidth=2.5, markersize=7, color='#E74C3C', alpha=0.8)
    ax1.plot(layers, down_jsd_averages, label="Projection JSD (down_proj)", marker='^',
             linewidth=2.5, markersize=7, color='#2ECC71', alpha=0.8)

    ax1.set_xlabel("Layer Index", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Average Counterfactual JSD", fontsize=13, fontweight='bold')
    ax1.set_title("MLP Counterfactual JSD:  Content vs Gate vs Projection", fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"MLP counterfactual JSD visualization saved to: {output_path}")
    plt.close()


def main(input_csv, model_path):
    print("=" * 60)
    print("Loading question data...")
    questions = load_csv_questions(input_csv)
    print(f"Loaded {len(questions)} questions")

    print(f"\nLoading model:  {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=CONFIG['dtype'],
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded successfully")

    target_device = torch.device(CONFIG['target_device'])
    lm_head_weight = model.lm_head.weight.to(target_device).detach()

    all_mlp_jsd = []
    all_attention_jsd = []
    all_head_jsd_matrices = []
    all_mlp_cf_up_jsd = []
    all_mlp_cf_gate_jsd = []
    all_mlp_cf_down_jsd = []

    print("\n" + "=" * 60)
    print("Starting question processing...")
    for idx, (dataset_name, question_index) in enumerate(tqdm(questions, desc="Processing questions")):
        try:
            question_data = load_original_question(dataset_name, question_index)
            prompt = question_data["prompt"]

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=CONFIG['max_length'])
            first_layer_device = next(model.model.layers[0].parameters()).device
            input_ids = inputs.input_ids.to(first_layer_device)
            attention_mask = inputs.attention_mask.to(first_layer_device)

            jsd_attention_mask = attention_mask

            # Analyze module-level JSD
            mlp_jsd_scores, attention_jsd_scores = analyze_model_jsd(
                model, input_ids, jsd_attention_mask, lm_head_weight, target_device
            )

            # Analyze attention head-level JSD
            head_jsd_matrix = analyze_attention_heads_jsd(
                model, input_ids, jsd_attention_mask, lm_head_weight, target_device
            )

            # Analyze MLP counterfactual JSD
            up_jsd_scores, gate_jsd_scores, down_jsd_scores = analyze_mlp_counterfactual_jsd(
                model, input_ids, jsd_attention_mask, lm_head_weight, target_device
            )

            # Ensure consistent length
            min_len = min(len(mlp_jsd_scores), len(attention_jsd_scores), len(up_jsd_scores))
            if len(mlp_jsd_scores) != min_len:
                mlp_jsd_scores = mlp_jsd_scores[:min_len]
                attention_jsd_scores = attention_jsd_scores[:min_len]
                up_jsd_scores = up_jsd_scores[:min_len]
                gate_jsd_scores = gate_jsd_scores[:min_len]
                down_jsd_scores = down_jsd_scores[:min_len]

            all_mlp_jsd.append(mlp_jsd_scores)
            all_attention_jsd.append(attention_jsd_scores)
            all_head_jsd_matrices.append(head_jsd_matrix)
            all_mlp_cf_up_jsd.append(up_jsd_scores)
            all_mlp_cf_gate_jsd.append(gate_jsd_scores)
            all_mlp_cf_down_jsd.append(down_jsd_scores)

            print(f"  Question {idx + 1}:  MLP JSD={np.mean(mlp_jsd_scores):.4f}, "
                  f"Attn JSD={np.mean(attention_jsd_scores):.4f}, "
                  f"Head JSD={np.mean(head_jsd_matrix):.4f}, "
                  f"CF Up JSD={np.mean(up_jsd_scores):.4f}, "
                  f"CF Gate JSD={np.mean(gate_jsd_scores):.4f}, "
                  f"CF Down JSD={np.mean(down_jsd_scores):.4f}")

        except Exception as e:
            print(f"\nError processing question {idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_mlp_jsd or not all_attention_jsd: 
        print("No questions processed successfully!")
        return

    mlp_jsd_averages = np.mean(all_mlp_jsd, axis=0)
    attention_jsd_averages = np.mean(all_attention_jsd, axis=0)
    head_jsd_averages = np.mean(all_head_jsd_matrices, axis=0)
    mlp_cf_up_jsd_averages = np.mean(all_mlp_cf_up_jsd, axis=0)
    mlp_cf_gate_jsd_averages = np.mean(all_mlp_cf_gate_jsd, axis=0)
    mlp_cf_down_jsd_averages = np.mean(all_mlp_cf_down_jsd, axis=0)

    print("\n" + "=" * 60)
    print("Saving analysis results...")

    analysis_results = {
        "mlp_jsd_averages":  mlp_jsd_averages.tolist(),
        "attention_jsd_averages": attention_jsd_averages.tolist(),
        "attention_head_jsd_averages": head_jsd_averages.tolist(),
        "mlp_cf_up_jsd_averages": mlp_cf_up_jsd_averages.tolist(),
        "mlp_cf_gate_jsd_averages": mlp_cf_gate_jsd_averages.tolist(),
        "mlp_cf_down_jsd_averages": mlp_cf_down_jsd_averages.tolist(),

        "mlp_jsd_all": [scores for scores in all_mlp_jsd],
        "attention_jsd_all": [scores for scores in all_attention_jsd],
        "attention_head_jsd_all":  [matrix.tolist() for matrix in all_head_jsd_matrices],
        "mlp_cf_up_jsd_all": [scores for scores in all_mlp_cf_up_jsd],
        "mlp_cf_gate_jsd_all": [scores for scores in all_mlp_cf_gate_jsd],
        "mlp_cf_down_jsd_all": [scores for scores in all_mlp_cf_down_jsd],

        "statistics": {
            "num_layers": int(len(mlp_jsd_averages)),
            "num_heads": int(head_jsd_averages.shape[1]) if len(head_jsd_averages.shape) > 1 else 0,
            "num_questions_processed": len(all_mlp_jsd),

            "mlp_mean":  float(np.mean(mlp_jsd_averages)),
            "mlp_std": float(np.std(mlp_jsd_averages)),
            "mlp_max": float(np.max(mlp_jsd_averages)),
            "mlp_min":  float(np.min(mlp_jsd_averages)),

            "attention_mean": float(np.mean(attention_jsd_averages)),
            "attention_std": float(np.std(attention_jsd_averages)),
            "attention_max":  float(np.max(attention_jsd_averages)),
            "attention_min": float(np.min(attention_jsd_averages)),

            "head_mean": float(np.mean(head_jsd_averages)),
            "head_std": float(np.std(head_jsd_averages)),
            "head_max": float(np.max(head_jsd_averages)),
            "head_min": float(np.min(head_jsd_averages)),

            "mlp_cf_up_mean": float(np.mean(mlp_cf_up_jsd_averages)),
            "mlp_cf_up_std": float(np.std(mlp_cf_up_jsd_averages)),
            "mlp_cf_gate_mean": float(np.mean(mlp_cf_gate_jsd_averages)),
            "mlp_cf_gate_std": float(np.std(mlp_cf_gate_jsd_averages)),
            "mlp_cf_down_mean": float(np.mean(mlp_cf_down_jsd_averages)),
            "mlp_cf_down_std":  float(np.std(mlp_cf_down_jsd_averages)),
        }
    }

    analysis_results = convert_numpy_types(analysis_results)

    output_json = "jsd_analysis_results.json"
    with open(output_json, "w", encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=4, ensure_ascii=False)
    print(f"Analysis results saved to: {output_json}")

    print("\n" + "=" * 60)
    print("JSD Score Statistical Summary:")
    print("-" * 60)
    stats = analysis_results['statistics']
    print(f"Successfully processed questions: {stats['num_questions_processed']}")
    print(f"Number of layers: {stats['num_layers']}")
    print(f"Number of attention heads: {stats['num_heads']}")

    print(f"\nMLP (overall): {stats['mlp_mean']:.6f} +/- {stats['mlp_std']:.6f}")
    print(f"Attention (overall): {stats['attention_mean']:.6f} +/- {stats['attention_std']:.6f}")
    print(f"Attention Head:  {stats['head_mean']:.6f} +/- {stats['head_std']:.6f}")
    print(f"MLP Content (Up): {stats['mlp_cf_up_mean']:.6f} +/- {stats['mlp_cf_up_std']:.6f}")
    print(f"MLP Gate: {stats['mlp_cf_gate_mean']:.6f} +/- {stats['mlp_cf_gate_std']:.6f}")
    print(f"MLP Projection (Down): {stats['mlp_cf_down_mean']:.6f} +/- {stats['mlp_cf_down_std']:.6f}")
    print("=" * 60)

    print("\nGenerating visualization charts...")
    visualize_jsd(mlp_jsd_averages, attention_jsd_averages, "jsd_scores.png")
    visualize_attention_heads_jsd(head_jsd_averages, "attention_heads_jsd_heatmap.png")
    visualize_head_statistics(head_jsd_averages, "attention_heads_statistics.png")
    visualize_mlp_counterfactual_jsd(
        mlp_cf_up_jsd_averages,
        mlp_cf_gate_jsd_averages,
        mlp_cf_down_jsd_averages,
        "jsd_mlp_counterfactual.png"
    )

    print("\nAnalysis complete!")
    print(f"\nGenerated files:")
    print(f"  - jsd_analysis_results.json")
    print(f"  - jsd_scores.png")
    print(f"  - attention_heads_jsd_heatmap.png")
    print(f"  - attention_heads_statistics.png")
    print(f"  - jsd_mlp_counterfactual.png (Updated:  includes Down projection JSD)")


if __name__ == "__main__":
    main(CONFIG['input_csv'], CONFIG['model_path'])