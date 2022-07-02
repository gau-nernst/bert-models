# Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/convert_bert_original_tf_checkpoint_to_pytorch.py
#
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert BERT checkpoint."""


import argparse
import os

import torch

from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers.utils import logging


logging.set_verbosity_info()


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, pytorch_dump_path):
    bert_config_file = os.path.join(tf_checkpoint_path, "bert_config.json")
    ckpt_file = os.path.join(tf_checkpoint_path, "bert_model.ckpt")
    config_file = os.path.join(pytorch_dump_path, "config.json")
    pt_ckpt = os.path.join(pytorch_dump_path, "pytorch_model.bin")

    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print(f"Building PyTorch model from configuration: {config}")
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, ckpt_file)

    # Save config file
    print(f"Save config to {config_file}")
    config.to_json_file(config_file)

    # Save pytorch-model
    print(f"Save PyTorch model to {pt_ckpt}")
    torch.save(model.state_dict(), pt_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path",
        default=".",
        help="Path to the TensorFlow checkpoint path.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        default=".",
        help="Path to the output PyTorch model.",
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.pytorch_dump_path)
