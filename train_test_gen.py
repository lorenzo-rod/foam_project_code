import torchvision.transforms as transforms
import torchvision
import torch

def train_dataset(train_dataset_path, test_dataset_path, image_resize):
    """Generates train and test loaders from files that contained labeled images

    Args:
        train_dataset_path (str): string containing the train dataset path
        test_dataset_path (str): string containing the test dataset path
        batch_size (int): batch size used in training

    Returns:
        DataLoader: train and test data loaders
    """
    train_transforms = transforms.Compose([
        transforms.RandomCrop(image_resize),
        transforms.Resize((image_resize, image_resize)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(10)
        transforms.ToTensor(),
        # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])


    test_transforms = transforms.Compose([
        transforms.RandomCrop(image_resize),
        transforms.Resize((image_resize, image_resize)),
        transforms.ToTensor(),
        # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)

    return train_dataset


def train_test_loaders(train_dataset_path, test_dataset_path, batch_size, shuffle_train, shuffle_test, image_resize):
    """Generates train and test loaders from files that contain labeled images

    Args:
        train_dataset_path (str): string containing the train dataset path
        test_dataset_path (str): string containing the test dataset path
        batch_size (int): batch size used in training

    Returns:
        DataLoader: train and test data loaders
    """
    train_transforms = transforms.Compose([
        transforms.RandomCrop(image_resize),
        transforms.Resize((image_resize, image_resize)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(10)
        transforms.ToTensor(),
        # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])


    test_transforms = transforms.Compose([
        transforms.RandomCrop(image_resize),
        transforms.Resize((image_resize, image_resize)),
        transforms.ToTensor(),
        # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=shuffle_train)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=shuffle_test)

    return (train_loader, test_loader)
    