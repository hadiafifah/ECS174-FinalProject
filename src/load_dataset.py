import torchvision
import torchvision.transforms as transforms
import kagglehub
from torch.utils.data import DataLoader

def load_dataset():
    batch_size = 64
    
    path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")
    print("Path to dataset files:", path)
    
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = torchvision.datasets.ImageFolder(
        root=f"{path}/images/train", 
        transform=transform
    )
    print("Train batches:", len(trainset))
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2
    )
    
    testset = torchvision.datasets.ImageFolder(
        root=f"{path}/images/validation", 
        transform=transform
    )
    print("Validation batches:", len(testset))
    testloader = DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2
    )
    
    return trainloader, testloader