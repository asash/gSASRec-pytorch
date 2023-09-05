import json
import torch
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, input_file, padding_value, output_file=None, max_length=200 ):
        with open(input_file, 'r') as f:
            self.inputs = [list(map(int, line.strip().split())) for line in f.readlines()]

        if output_file:
            with open(output_file, 'r') as f:
                self.outputs = [int(line.strip()) for line in f.readlines()]
        else:
            self.outputs = None

        self.max_length = max_length
        self.padding_value = padding_value

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        rated = set(inp)
        if len(inp) > self.max_length:
            inp = inp[-self.max_length:]
        elif len(inp) < self.max_length:
            inp = [self.padding_value] * (self.max_length - len(inp)) + inp

        inp_tensor = torch.tensor(inp, dtype=torch.long)

        if self.outputs:
            out_tensor = torch.tensor(self.outputs[idx], dtype=torch.long)
            return inp_tensor, rated, out_tensor 

        return inp_tensor,

def collate_with_random_negatives(input_batch, pad_value, num_negatives):
    batch_cat = torch.stack([input_batch[i][0] for i in range(len(input_batch))], dim=0)
    negatives = torch.randint(low=1, high=pad_value, size=(batch_cat.size(0), batch_cat.size(1), num_negatives))
    return [batch_cat, negatives]

def collate_val_test(input_batch):
    input = torch.stack([input_batch[i][0] for i in range(len(input_batch))], dim=0)
    rated = [input_batch[i][1] for i in range(len(input_batch))]
    output = torch.stack([input_batch[i][2] for i in range(len(input_batch))], dim=0)
    return [input, rated, output]

def get_num_items(dataset):
    with open(f"datasets/{dataset}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    return stats['num_items']

def get_padding_value(dataset_dir):
    with open(f"{dataset_dir}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    padding_value = stats['num_items'] + 1
    return padding_value

def get_train_dataloader(dataset_name, batch_size=32, max_length=200, train_neg_per_positive=256):
    dataset_dir = f"datasets/{dataset_name}"
    padding_value = get_padding_value(dataset_dir)
    train_dataset = SequenceDataset(f"{dataset_dir}/train/input.txt", max_length=max_length + 1, padding_value=padding_value) # +1 for sequence shifting
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_with_random_negatives(x, padding_value , train_neg_per_positive))
    return train_loader

def get_val_or_test_dataloader(dataset_name, part='val', batch_size=32, max_length=200):
    dataset_dir = f"datasets/{dataset_name}"
    padding_value = get_padding_value(dataset_dir)
    dataset = SequenceDataset(f"{dataset_dir}/{part}/input.txt", padding_value,  f"{dataset_dir}/{part}/output.txt", max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_val_test)
    return dataloader

def get_val_dataloader(dataset_name, batch_size=32, max_length=200):
    return get_val_or_test_dataloader(dataset_name, 'val', batch_size, max_length)

def get_test_dataloader(dataset_name, batch_size=32, max_length=200):
    return get_val_or_test_dataloader(dataset_name, 'test', batch_size, max_length)