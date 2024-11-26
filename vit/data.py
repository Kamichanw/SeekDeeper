from typing import Sequence
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import config


def _load_cifar10(num_proc, splits, **kwargs):
    if splits is not None and not all(s in ["train", "dev", "test"] for s in splits):
        raise ValueError('Splits must be a subset of ["train", "dev", "test"].')

    resolution = kwargs.get("resolution", config.resolution)
    batch_size = kwargs.get("batch_size", config.batch_size)

    transform = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataloaders = []

    if "train" in splits:
        train_dataset = datasets.CIFAR10(
            root=config.dataset_dir, train=True, download=True, transform=transform
        )

        if "dev" in splits:
            train_size = int(0.98 * len(train_dataset))
            dev_size = len(train_dataset) - train_size
            train_dataset, dev_dataset = random_split(
                train_dataset, [train_size, dev_size]
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_proc,
            )
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_proc,
            )
            dataloaders.append(train_loader)
            dataloaders.append(dev_loader)
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_proc,
            )
            dataloaders.append(train_loader)

    if "test" in splits:
        test_dataset = datasets.CIFAR10(
            root=config.dataset_dir, train=False, download=True, transform=transform
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_proc,
        )
        dataloaders.append(test_loader)

    return dataloaders


def load_data(
    name: str,
    num_proc: int = 0,
    splits: Sequence[str] = None,
    **kwargs,
):
    dispatch = {
        "cifar10": _load_cifar10,
    }

    assert name.lower() in dispatch, f"Unsupported dataset, should be 'cifar10'."

    return dispatch[name.lower()](num_proc, splits, **kwargs)
