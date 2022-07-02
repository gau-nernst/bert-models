import json
import os
import subprocess
from itertools import product

skip_list = [
    (2, 128),  # tiny
    (4, 256),  # mini
    (4, 512),  # small
    (8, 512),  # medium
    (12, 768),  # base
]

L_list = [2, 4, 6, 8, 10, 12]
H_list = [128, 256, 512, 768]

for L, H in product(L_list, H_list):
    if (L, H) in skip_list:
        continue

    dirname = f"bert-L{L}-H{H}-uncased"
    if os.path.exists(dirname):
        continue

    commands = [
        f"git clone https://huggingface.co/gaunernst/{dirname}",
        f"unzip uncased_L-{L}_H-{H}_A-{H//64}.zip -d {dirname}",
        f"python convert_bert_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path={dirname} --pytorch_dump_path={dirname}",
    ]
    for cmd in commands:
        subprocess.run(cmd, shell=True)

    with open(os.path.join(dirname, "tokenizer_config.json"), "w") as f:
        json.dump({"do_lower_case": True}, f, indent=4)
        f.write("\n")

    commands = [
        f"cd {dirname}",
        "git add config.json tokenizer_config.json pytorch_model.bin vocab.txt",
        "git commit -m 'add model'",
        "git push",
    ]
    subprocess.run(" && ".join(commands), shell=True)
