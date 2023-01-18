from torch.utils.data import Dataset
import torch


class ReverseEnglishDataset(Dataset):

    def __init__(self, dataset, tokenizer):
        self.tokenizer = tokenizer
        self.data = dataset
        self.max_len = 177   # calculated on the whole dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_text = str(self.data[index]['text'])
        reversed_text = input_text[::-1]
        input_text = ' '.join(input_text.split())
        reversed_text = ' '.join(reversed_text.split())

        source = self.tokenizer.batch_encode_plus([input_text], max_length=self.max_len, pad_to_max_length=True,
                                                  return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([reversed_text], max_length=self.max_len, pad_to_max_length=True,
                                                  return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }