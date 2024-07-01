import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from simple_linear_NN import NeuralNetwork

torch.manual_seed(10)

def get_device():
    return (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backend.mps.is_available()
    else "cpu"
)

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs,labels)
            total_loss += loss * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total}%')
    print(f'Test Loss: {total_loss / total}')







if __name__ == '__main__':
    
    
    device = get_device()
    
    test_data = datasets.FashionMNIST(
                root = "data",
                train = False,
                download = True,
                transform = ToTensor()
)
    batch_size = 64
    
    test_data_loader = DataLoader(test_data,batch_size=batch_size)

    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("Simple_NN.pth"))
    loss_fn = nn.CrossEntropyLoss()
    
    test(test_data_loader, model)
    
