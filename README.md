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

## HuggingFace Hub models

This repo also contains the scripts to convert smaller BERT models from [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962) to HuggingFace's format and upload to HuggingFace Hub. The converted models can be found at https://huggingface.co/gaunernst

```bash
wget https://storage.googleapis.com/bert_models/2020_02_20/all_bert_models.zip
unzip all_bert_models.zip
python upload_models_to_hf.py
```

For BERT variants that have a specific name e.g. BERT-tiny, the specific name is used instead on HuggingFace Hub.
