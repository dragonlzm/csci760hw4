import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.init as init
import json

# Define the neural network architecture
class MyNeuralNet(nn.Module):
    def __init__(self, input_size, L1_dim, output_dim, init_val=None):
        super(MyNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, L1_dim)
        self.act_fun1 = nn.Sigmoid()
        self.fc2 = nn.Linear(L1_dim, output_dim)
        if init_val == None:
            init.normal_(self.fc1.weight, mean=0, std=0.1)
            init.normal_(self.fc2.weight, mean=0, std=0.1)
        elif init_val == 0:
            init.zeros_(self.fc1.weight)
            init.zeros_(self.fc2.weight)
        elif init_val == 'uniform':
            init.uniform_(self.fc1.weight, a=-1, b=1)
            init.uniform_(self.fc2.weight, a=-1, b=1)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fun1(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # set the hyper-params
    random_seed = 42
    num_epoch = 20
    batch_size = 32
    learning_rate = 0.04
    input_size = 784
    #L1_dim = 300
    L1_dim = 200
    output_dim = 10
    step = 16

    # set the random seed to reproduce
    torch.manual_seed(random_seed)

    # Load dataset and build the dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = MyNeuralNet(input_size, L1_dim, output_dim)
    #model = MyNeuralNet(input_size, L1_dim, output_dim, init_val=0)
    #model = MyNeuralNet(input_size, L1_dim, output_dim, init_val='uniform')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    trainset_acc_list, testset_acc_list = [], []
    
    for t in range(num_epoch):
        print('Start the epoch ' + str(t + 1))
        if step is not None and t >= step:
            learning_rate *= 0.1
        for i, (images, labels) in enumerate(train_loader):
            # flatten the image to a 1d vector
            images = images.view(-1, input_size)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # calculate the loss and acc on the train set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in train_loader:
                # flatten the image to a 1d vector and forward
                images = images.view(-1, input_size)
                outputs = model(images)
                # calculate the acc
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        trainset_acc = correct / total
        trainset_acc_list.append(trainset_acc)
        # calculate the loss and acc on the test set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                # flatten the image to a 1d vector and forward
                images = images.view(-1, input_size)
                outputs = model(images)
                # calculate the acc
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        testset_acc = correct / total
        testset_acc_list.append(testset_acc)
        print('Epoch ' + str(t+1), ' the training acc is: ', trainset_acc, 'the test acc is: ', testset_acc)


    # save log
    json.dump({'train_acc_list': trainset_acc_list, 'val_acc_list': testset_acc_list},
                open('PytorchMLP_lr' + str(learning_rate) + '_layerdim_' + str(L1_dim) + '_bs_' + str(batch_size) + '_epoch_' + str(num_epoch) +  '_step_' + str(step) + '.json', 'w'))
    # json.dump({'train_acc_list': trainset_acc_list, 'val_acc_list': testset_acc_list},
    #             open('PytorchMLP_lr' + str(learning_rate) + '_layerdim_' + str(L1_dim) + '_bs_' + str(batch_size) + '_epoch_' + str(num_epoch) +  '_step_' + str(step) + 'zero_init.json', 'w'))
    # json.dump({'train_acc_list': trainset_acc_list, 'val_acc_list': testset_acc_list},
    #             open('PytorchMLP_lr' + str(learning_rate) + '_layerdim_' + str(L1_dim) + '_bs_' + str(batch_size) + '_epoch_' + str(num_epoch) +  '_step_' + str(step) + 'unit_init.json', 'w'))
    print('Finish training.')


