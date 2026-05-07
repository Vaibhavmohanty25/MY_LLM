import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    Loads a text file, tokenizes at the byte level, and returns
    (input, target) pairs of length `block_size` where target is
    input shifted by one position.
    """
    def __init__(self, path: str, block_size: int):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        data = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(1, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        # Pad if necessary (handles edge case at end of data)
        if len(chunk) < self.block_size + 1:
            pad = torch.zeros(self.block_size + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def make_dataloaders(path: str, block_size: int, batch_size: int, val_frac: float = 0.1):
    """Split data into train/val and return DataLoaders."""
    from torch.utils.data import DataLoader, random_split

    dataset = TextDataset(path, block_size)
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader
