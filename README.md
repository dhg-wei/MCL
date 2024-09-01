# (ICML 2024) Improve Context Understanding in Multimodal Large Language Models via Multimodal Composition Learning

# Under Construction

## Llama2
```
Downloading llama2-7B from https://huggingface.co/meta-llama/Llama-2-7b
```

## Training
```
Prepare data ...
```
```
# train with 4 gpus
./train.sh
```

## Evaluation
```
cd repos
git clone https://github.com/miccunifi/CIRCO.git
git clone git clone -b cirr_dataset git@github.com:Cuberick-Orion/CIRR.git cirr
```

```
# download pretrained models
cd runs
wget https://drive.google.com/file/d/1oYfffZ6ckVjSbxG9rk0h5OACYfEWIWxv/view?usp=sharing
tar -xf icml_run.tar
```

```
# download data
cd data
download MSCOCO unlabel images from https://cocodataset.org/#download
download NLVR2 from https://huggingface.co/datasets/TIGER-Lab/NLVR2
```
```
# CIR inference
cir_inference.ipynb
```

