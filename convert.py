import argparse

import numpy as np
import torch
from tensorflow.python.training import py_checkpoint_reader
from torch import nn


def get_model_info(reader):
    var_to_shape = reader.get_variable_to_shape_map()
    d_model = var_to_shape["bert/pooler/dense/bias"][0]
    nlayer = 0
    while True:
        if f"bert/encoder/layer_{nlayer}/output/dense/bias" in var_to_shape:
            nlayer += 1
        else:
            break
    return d_model, nlayer


def convert_layer_from_tf_to_pt(
    reader: py_checkpoint_reader.CheckpointReader, layer_idx: int
):
    prefix = f"bert/encoder/layer_{layer_idx}"
    state_dict = {}

    attn_in_proj_weight, attn_in_proj_bias = [], []
    for x in ("query", "key", "value"):
        name = f"{prefix}/attention/self/{x}/kernel"
        data = reader.get_tensor(name).T
        attn_in_proj_weight.append(data)

        name = f"{prefix}/attention/self/{x}/bias"
        data = reader.get_tensor(name)
        attn_in_proj_bias.append(data)

    state_dict["self_attn.in_proj_weight"] = np.concatenate(attn_in_proj_weight, axis=0)
    state_dict["self_attn.in_proj_bias"] = np.concatenate(attn_in_proj_bias)

    mapper = {
        "self_attn.out_proj.weight": f"{prefix}/attention/output/dense/kernel",
        "self_attn.out_proj.bias": f"{prefix}/attention/output/dense/bias",
        "norm1.weight": f"{prefix}/attention/output/LayerNorm/gamma",
        "norm1.bias": f"{prefix}/attention/output/LayerNorm/beta",
        "linear1.weight": f"{prefix}/intermediate/dense/kernel",
        "linear1.bias": f"{prefix}/intermediate/dense/bias",
        "linear2.weight": f"{prefix}/output/dense/kernel",
        "linear2.bias": f"{prefix}/output/dense/bias",
        "norm2.weight": f"{prefix}/output/LayerNorm/gamma",
        "norm2.bias": f"{prefix}/output/LayerNorm/beta",
    }
    for pt_name, tf_name in mapper.items():
        tensor = reader.get_tensor(tf_name)
        if tf_name.endswith("kernel"):
            tensor = tensor.T
        state_dict[pt_name] = tensor

    state_dict = {
        f"layers.{layer_idx}.{k}": torch.from_numpy(v) for k, v in state_dict.items()
    }
    return state_dict


def get_pt_encoder(nlayer: int, d_model: int):
    dim_feedforward = d_model * 4
    nhead = d_model // 64

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
    )
    transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayer)
    return transformer_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reader = py_checkpoint_reader.NewCheckpointReader(args.input)
    d_model, nlayer = get_model_info(reader)
    print(f"{d_model = }, {nlayer = }")
    state_dict = {}
    for i in range(nlayer):
        state_dict.update(convert_layer_from_tf_to_pt(reader, i))

    pt_model = get_pt_encoder(nlayer, d_model)
    pt_model.load_state_dict(state_dict)
    num_params = sum(x.numel() for x in pt_model.parameters())
    print(f"num params: {num_params/1e6:.2f}M")

    torch.save(pt_model.state_dict(), args.output)


if __name__ == "__main__":
    main()
