# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from kascade.attn_recovery import RecoveryMLP
import torch
import re
from pathlib import Path
from random import sample

def get_tokenizer_and_model(model_name, attn_implementation, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        if tokenizer.bos_token is not None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.bos_token})
        else:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    model_kwargs = {
        "torch_dtype": torch.float16,
        "attn_implementation": attn_implementation,
        "cache_dir": "/dev/shm",
        "pretrained_model_name_or_path": model_name,
    }

    # Check if model size with float16 is greater than single GPU memory
    model_size_gb = 0
    match = re.search(r'(\d+(\.\d+)?)([mMbB])', model_name)
    if match:
        size, _, unit = match.groups()
        size = float(size)
        if unit.lower() == 'b':
            model_size_gb = size
        elif unit.lower() == 'm':
            model_size_gb = size / 1024  # Convert MB to GB
    single_gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if model_size_gb * 2 > single_gpu_mem_gb:  # float16 is 2 bytes per parameter
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    model.eval()

    if "70B" not in model_name:
        model = model.to(device)

    return model, tokenizer


def get_attn_out_stat_profile(model, 
                                extracted_attn_in = {}, 
                                extracted_vmatrix = {},
                                attn_in_sample_size = 1000,
                                extracted_attn_out_mean = {}, 
                                extracted_attn_out_std = {}):
    
    # Single shared index list for the entire forward pass.
    # Cleared by a model pre-hook at the start of each forward pass so that
    # fresh indices are computed once (by the first v_proj that fires) and
    # reused identically across all layers within that pass.
    sel_idces = []

    def get_vmatrix_hook(layer_name):
        def hook(module, input, output):
            # output of v_proj: (batch, seqlen, num_heads * head_dim)
            v = output
            if isinstance(v, tuple):
                v = v[0]

            v_flat = v.reshape(-1, v.shape[-1])
            print("v_flat shape: ", v_flat.shape)

            # Compute indices once per forward pass (list is cleared by pre-hook)
            if len(sel_idces) == 0:
                n = v_flat.shape[0]
                if attn_in_sample_size > 0 and n > attn_in_sample_size:
                    # sel_idces[:] = sample(range(n), attn_in_sample_size)
                    sel_idces[:] = range(attn_in_sample_size)
                else:
                    sel_idces[:] = range(n)

            v_sampled = v_flat[sel_idces].detach().cpu().numpy().astype(float)
            extracted_vmatrix[layer_name] = extracted_vmatrix.get(layer_name, []) + [v_sampled]
        return hook

    def get_activation_hook(layer_name):
        def hook(module, input, kwargs, output):
            curr_out = None
            if isinstance(output, tuple):
                curr_out = output[0]
            else:
                curr_out = output

            curr_in = kwargs.get("hidden_states", None)
            if curr_in is None and len(input) > 0:
                curr_in = input[0]

            curr_in = curr_in.reshape(-1, curr_in.shape[-1])
            curr_out = curr_out.reshape(-1, curr_out.shape[-1])

            print("curr_in shape: ", curr_in.shape)
            print("curr_out shape: ", curr_out.shape)

            # Reuse indices computed by v_proj hook for this same forward pass
            idces = sel_idces
            if len(idces) == 0:
                # Fallback: v_proj hook hasn't run (shouldn't happen); sample here
                n = curr_in.shape[0]
                idces = sample(range(n), attn_in_sample_size) if n > attn_in_sample_size else list(range(n))

            curr_in_s = curr_in[idces].detach().cpu().numpy().astype(float)
            extracted_attn_in[layer_name] = extracted_attn_in.get(layer_name, []) + [curr_in_s]

            curr_out_s = curr_out[idces]
            curr_out_mean = curr_out_s.mean(dim=-1, keepdims=True).detach().cpu().numpy().astype(float)
            curr_out_std  = curr_out_s.std(dim=-1, keepdims=True).detach().cpu().numpy().astype(float)
            extracted_attn_out_mean[layer_name] = extracted_attn_out_mean.get(layer_name, []) + [curr_out_mean]
            extracted_attn_out_std[layer_name]  = extracted_attn_out_std.get(layer_name, []) + [curr_out_std]

        return hook

    model_name = model.name_or_path.lower()
    handles = []

    if "llama" in model_name or "qwen3" in model_name:
        n_layers = model.config.num_hidden_layers

        # Clear shared sel_idces at the start of every forward pass so indices
        # are resampled once per sequence and reused across all layers.
        handles.append(model.model.register_forward_pre_hook(
            lambda module, args: sel_idces.clear()
        ))

        for l in range(n_layers):
            # v_proj hook fires first (sub-module), sets sel_idces if still empty
            handles.append(model.model.layers[l].self_attn.v_proj.register_forward_hook(
                get_vmatrix_hook(f"layer_{l}")
            ))
            handles.append(model.model.layers[l].self_attn.register_forward_hook(
                get_activation_hook(f"layer_{l}"), with_kwargs=True
            ))
    
    return handles


def apply_mlp_recovery(model, mlp_model_path: Path, mlp_dim=256):
    """
    Patches an existing Llama model with the Recovery MLP.
    
    Args:
        model: The loaded HuggingFace LlamaForCausalLM instance.
        mlp_model_path: Path to your best_dual_oracle_dist.pt file.
    """
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    model_name = model.name_or_path.split("/")[-1]
    
    # Initialize and load MLP
    mlp_recovery_models = []
    print("Loading RecoveryMLP model params")
    for l in range(num_layers):
        mlp_recovery_models.append(RecoveryMLP(
            hidden_size=hidden_size,
            mlp_dim=mlp_dim
        ))
    
        mlp_weight_path = mlp_model_path / f"{model_name}_recovery_mlp_{mlp_dim}_layer_{l}.pt"
        if not mlp_weight_path.exists():
            raise FileNotFoundError(f"MLP weights not found at {mlp_weight_path}")
        
        # In multi-GPU pipeline parallelism, each layer can be on a different device.
        # We must place the MLP on the exact same device as the layer.
        layer_device = model.model.layers[l].self_attn.q_proj.weight.device
        
        state_dict = torch.load(mlp_weight_path, map_location=layer_device, weights_only=True)
        mlp_recovery_models[l].load_state_dict(state_dict)
        mlp_recovery_models[l].to(layer_device)
        mlp_recovery_models[l].eval()

    def make_recovery_hook(mlp_recovery_model):
        def recovery_hook(module, args, kwargs, output):
            attn_output = output[0]
            original_shape = attn_output.shape

            # The MLP was trained on hidden_states as input, not attn_output
            curr_in = kwargs.get("hidden_states", None)
            if curr_in is None and len(args) > 0:
                curr_in = args[0]
            x_flat = curr_in.view(-1, curr_in.shape[-1]).to(torch.float32)

            with torch.no_grad():
                pred_std = mlp_recovery_model(x_flat)
                pred_std = pred_std.view(*original_shape[:2], 1).to(attn_output.dtype)
                eps = 1e-8
                curr_mu = attn_output.mean(dim=-1, keepdim=True)
                curr_std = attn_output.std(dim=-1, keepdim=True)
                attn_output.sub_(curr_mu).div_(curr_std + eps).mul_(pred_std).add_(curr_mu)
            return (attn_output,) + output[1:]
        return recovery_hook
    
    handlers = []

    if "llama" in model_name.lower() or "qwen3" in model_name.lower():
        print(f"applying mlp recovery")
        for l in range(num_layers):
            # skip layer 0 since it's not pruned.
            if l == 0:
                continue
            handlers.append(model.model.layers[l].self_attn.register_forward_hook(
                make_recovery_hook(mlp_recovery_models[l]), with_kwargs=True
            ))
    
    return handlers
    
    
def get_inst_tokens(model_name, use_sys_token = False, enable_thinking = False):
    inst_token_dict = {
        "deepseek": ["<｜{}｜>", "", "<think>\n"],
        "llama": ("<|start_header_id|>{}<|end_header_id|>\n\n", "", "<|eot_id|>"),
        "mistral": ("INST] ", "[", " [/"),
        "qwen": ("<|im_start|>{}\n", "", "<|im_end|>\n"),
        "olmo-2": ("<|{}|>", "", ""),
    }

    model_key = next((k for k in inst_token_dict if k in model_name.lower()), None)
    if model_key is None:
        raise ValueError(f"Model name '{model_name}' does not match any known token patterns.")
    is_thinking_model = "deepseek" in model_name.lower() or "qwen3" in model_name.lower()
    inst_tokens = inst_token_dict[model_key]
    tokens_to_return = ["","",""]
    if use_sys_token and model_key not in ["mistral", "deepseek"]:
        tokens_to_return[0] = inst_tokens[1]+inst_tokens[0].format("system")
        tokens_to_return[1] = inst_tokens[2]+inst_tokens[0].format("user")
        tokens_to_return[2] = inst_tokens[2]+inst_tokens[0].format("assistant")
    elif model_key == "deepseek":
        tokens_to_return[0] = inst_tokens[1]+inst_tokens[0].format("User")
        tokens_to_return[1] = ""
        tokens_to_return[2] = inst_tokens[0].format("Assistant")+inst_tokens[2]
    elif model_key == "olmo-2":
        tokens_to_return[0] = inst_tokens[0].format("user")
        tokens_to_return[1] = ""
        tokens_to_return[2] = inst_tokens[0].format("assistant")
    else:
        tokens_to_return[0] = inst_tokens[1]+inst_tokens[0].format("user")
        tokens_to_return[1] = ""
        tokens_to_return[2] = inst_tokens[2]+inst_tokens[0].format("assistant")
    if is_thinking_model and not enable_thinking:
        tokens_to_return[2] += "<think>\n\n</think>\n\n"
    return tokens_to_return

def get_eos_token_ids(stop_strings, tokenizer):
    token_ids = []
    for token_str in stop_strings:
        # Check if token is a known token
        if token_str in tokenizer.vocab or token_str in tokenizer.get_vocab():
            token_ids.append(tokenizer.convert_tokens_to_ids(token_str))
        else:
            # Otherwise encode using a dummy prefix
            dummy_text = "dummy" + token_str
            encoded_ids = tokenizer.encode(dummy_text, add_special_tokens=False)
            # Use the id of the newly encoded token
            token_ids.append(encoded_ids[-1])
    return token_ids
