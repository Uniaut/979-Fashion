import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define a neural network
class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.BatchNorm2d(n_channels),
        )
    
    def forward(self, x):
        out_x = self.seq(x)
        return torch.relu(out_x + x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.top_block = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        rb1 = [ResBlock(16) for _ in range(10)]
        self.res_blocks_1 = nn.Sequential(
            *rb1
        )
        
        self.pool_seq_1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        rb2 = [ResBlock(32) for _ in range(10)]
        self.res_blocks_2 = nn.Sequential(
            *rb2
        )

        self.pool_seq_2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        rb3 = [ResBlock(64) for _ in range(10)]
        self.res_blocks_3 = nn.Sequential(
            *rb3
        )

        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        out_x = self.top_block(x)
        for block in self.res_blocks_1:
            out_x = block(out_x)
        out_x = self.pool_seq_1(out_x)
        for block in self.res_blocks_2:
            out_x = block(out_x)
        out_x = self.pool_seq_2(out_x)
        for block in self.res_blocks_3:
            out_x = block(out_x)
        out_x = self.fc_head(out_x)
        return out_x


# Function to show some test images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# apple m2 chip
device = torch.device("mps")
NET_PATH = "./misc/mnist_net.pth"


def train():
    # Load MNIST dataset
    train_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)),
    ])
    trainset = torchvision.datasets.MNIST(root='../data/', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))

    # Initialize the neural network and optimizer
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.1, patience=1, mode='min')

    # Training the network
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.5f}")
                running_loss = 0.0

    print("Finished Training")

    # Save the trained model
    torch.save(net.state_dict(), NET_PATH)
    print("Saved trained model")


def test():
    # Load and test the trained model
    net = Net()
    net.load_state_dict(torch.load(NET_PATH))
    net.eval()

    # Load the test dataset

    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)),
    ])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Show some images
    imshow(torchvision.utils.make_grid(images))
    print("Ground Truth:", " ".join(f"{labels[j]}" for j in range(4)))

    # Make predictions
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print("Predicted: ", " ".join(f"{predicted[j]}" for j in range(4)))

    # Test the network on the entire test dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")


if __name__ == "__main__":
    train()
    test()