# Fine-tuning T5 model to translate English news articles to reverse-English

import numpy as np
import pandas as pd
import tqdm
import argparse

import torch
from torch.utils.data import DataLoader

from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

from dataset import ReverseEnglishDataset
from training_utils import train, validate


def main():
    parser = argparse.ArgumentParser(description='Processing configuration for fine-tuning')
    parser.add_argument('--batch_size', type=int, help='length of batch size', default=16)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    device = args.device

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Initializing tokenzier for encoding
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # Loading news articles dataset
    dataset = load_dataset("ag_news")

    # Train/dev split
    split_indx = int(len(dataset['train'])*0.8)

    train_dataset = [dataset['train'][i] for i in range(len(dataset['train'])) if i < split_indx]
    dev_dataset = [dataset['train'][i] for i in range(len(dataset['train'])) if i > split_indx]
    print('Train: {}, Validation: {}'.format(len(train_dataset), len(dev_dataset)))

    # Creating the Training and Validation datasets
    training_set = ReverseEnglishDataset(train_dataset, tokenizer)
    dev_set = ReverseEnglishDataset(dev_dataset, tokenizer)

    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(dev_set, **val_params)

    # Initializing the model
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # Training
    print('Initiating Fine-Tuning for the model on our dataset')

    for epoch in tqdm.tqdm(range(args.train_epochs)):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    # Validation
    predictions, gt = validate(tokenizer, model, device, val_loader)
    final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Reversed Text': gt})
    final_df.to_csv('predictions.csv')


if __name__ == '__main__':
    main()
