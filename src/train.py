import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from . import cli

data_dir, save_dir, arch, hidden_units, learning_rate, epochs, device = cli.init_train()

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

batch_size = 64

trainloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size)
testloader = torch.utils.data.DataLoader(test_data, batch_size)

model = getattr(models, arch)(pretrained=True)

def extract_input(model):
    try:
        return model.classifier.in_features
    except Exception as e:
        return model.classifier[0].in_features

input_features = extract_input(model)
output_units = 102

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units, output_units),
        nn.LogSoftmax(dim=1)
        )

model.classifier = classifier
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

steps = 0
running_loss = 0
print_every = 1

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
            
                    valid_loss += batch_loss.item()
            
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            print(f'Epoch {epoch + 1}/{epochs}... '
                  f'Train loss: {running_loss/print_every:.3f}... '
                  f'Validation loss: {valid_loss/len(validloader):.3f}... '
                  f'Validation accuracy: {accuracy/len(validloader):.3f}... ')
            running_loss = 0
            model.train()

checkpoint = {
        'input_size': input_features,
        'hidden_units': hidden_units,
        'output_size': output_units,
        'epochs': epochs,
        'batch_size': batch_size,
        'model': getattr(models, arch)(pretrained=True),
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'optimizer_state': optimizer.state_dict(),
        'class_to_idx': train_data.class_to_idx
        }

def save_model(checkpoint):
    if save_dir:
        torch.save(checkpoint, f'./models/checkpoint-{arch}-{hidden_units}.pth')
    else:
        print('Model is not saved. Please provide save_directory')
