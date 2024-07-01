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

def train(dataloader, model, loss_fn, optimizer):
    
    size = len(dataloader.dataset)
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0
    for batch, (inputs, labels) in enumerate(dataloader):
        x, y = inputs.to(device), labels.to(device)

        # compute loss
        pred = model(x)
        loss = loss_fn(pred,y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss.item() * x.size(0)
        _, predicted = torch.max(pred.data, 1)
        
        train_correct += (predicted == y).sum().item()
        train_total += y.size(0)


        if batch % 100 == 0:
            loss , current = loss.item() , (batch + 1) * len(x)
            print(f"Batch no: {batch} | loss: {loss:>7f} [{current:>5d} / {size:>5d}]")

    print(f'Training Accuracy: {100 * train_correct / train_total}%')
    print(f'Training Loss: {train_loss / train_total}')





if __name__ == '__main__':
    
    
    device = get_device()
    
    training_data = datasets.FashionMNIST(
                root = "data",
                train = True,
                download = True,
                transform = ToTensor()
)
    batch_size = 64
    num_epochs = 5
    train_data_loader = DataLoader(training_data,batch_size=batch_size)

    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train(train_data_loader, model, loss_fn, optimizer)
    torch.save(model.state_dict(), "Simple_NN.pth")
