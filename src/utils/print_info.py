from torch.utils.data import Dataset, DataLoader


def print_dataset_info(dataset: Dataset,
                       dataloader: DataLoader,
                       name: str = 'train') -> None:
    """
    Prints dataset, dataloader info to console
    :param dataset: PyTorch Dataset
    :param dataloader: PyTorch DataLoader
    :param name: Dataset identity (train/test/val/etc.)
    """
    print(f'- Dataset {name} is loaded')
    print(f' - Loaded {name} len: {len(dataset)}')
    print(f' - Loaded {name} batches len: {len(dataloader)}')
