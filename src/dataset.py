
from torchvision import transforms, datasets
import torch

def get_dataloaders():
    ALL_DATA_DIR = '../datasets/01_FER2013_datasets/'
    train_dir = ALL_DATA_DIR + 'train'
    test_dir = ALL_DATA_DIR + 'test'
    batch_size = 128
    IMG_SIZE= 48# 300 # 80 #
    train_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ]
    )
    
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs) 
    return train_loader, test_loader