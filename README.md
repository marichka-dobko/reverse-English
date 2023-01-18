# Training a T5 model to reverse English news articles

## Dataset
The used dataset is AG news. It's a collection of more than 1 million news articles gathered from more than 2000 news sources.
For more information please refer to this [website](https://huggingface.co/datasets/ag_news).

The train/validation split for all experiments was the following: Train: 96,000 articles, Validation: 23,999 articles.

## Model
The architecture used in these experiments is T5 encoder-decoder model pre-trained on a 
multi-task mixture of unsupervised and supervised tasks. The size configuration is `t5-base`. 
Implementation is from Hugging Face transformers library - [link](https://huggingface.co/docs/transformers/model_doc/t5).

## Dependencies
For all necessary modules refer to [requirements.txt](requirements.txt)
```
pip3 install -r requirements.txt
```
This implementation is inspired by the tutorial from this [repository](https://github.com/abhimishra91/transformers-tutorials).


## Training
<b>To launch fine-tuning, run: </b>   
```
python finetune.py --batch_size 16 --device 'cuda:0' --train_epochs 10 --seed 42
```
After finetuning for 10 epochs on a subset of AG news training data, the loss dropped from 
11.60984 to 0.008037.

## Achieved Results
The produced predictions on validation set is shown in [predictions.csv](predictions.csv).

<b>Example:</b>

Original text: `Oracle to Issue Tender Results Saturday Oracle Corp.(ORCL.O: Quote,Profile,Research) said it would report preliminary results of its $8.8 billion hostile tender offer for PeopleSoft (PSFT.`

Target text: `.TFSP( tfoSelpoeP rof reffo rednet elitsoh noillib 8.8$ sti fo stluser yranimilerp troper dluow ti dias )hcraeseR,eliforP,etouQ :O.LCRO(.proC elcarO yadrutaS stluseR redneT eussI ot elcarO`

Generated text: `.TFSP( tfoSelpoeP rof reffo rednet elitsoh noillib 8.8$ sti fo stluser yranimilerp troper dluow ti dias )hcraeseR,eliforP,etouQ :O.LCRO(.proC elcarO yadrutaS stluseR redneT eussI ot elcarO`