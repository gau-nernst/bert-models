# BERT models

This repo contains the conversion script to convert BERT checkpoints from [google-research/bert](https://github.com/google-research/bert) to PyTorch's `nn.TransformerEncoder`. The converted weights can be used to initialize transformer encoders in other tasks, such as multi-modal transformer fusion.

## Dependencies

The script depends on PyTorch, TensorFlow, and NumPy. Using `conda`:

```bash
conda install pytorch -c pytorch
conda install tensorflow numpy
```

## Usage

Download a checkpoint from [google-research/bert](https://github.com/google-research/bert) and unzip it.

```bash
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip -d uncased_L-12_H-768_A-12
```

Run the script

```bash
python convert.py --input=uncased_L-12_H-768_A-2/bert_model.ckpt --output=bert_L12_H768.ckpt
```
