from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

train_val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

])
cifar_trainset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_val_transforms)
cifar_testset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=train_val_transforms)


train_loader = DataLoader(cifar_trainset, batch_size=16,
                          shuffle=True, num_workers=0)
test_loader = DataLoader(cifar_testset, batch_size=16,
                         shuffle=False, num_workers=0)


class CifarClassifier(nn.Module):
    def __init__(self):
        super(CifarClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),    
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CifarClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    acc = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc += loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {acc/3125:.4f}')

torch.save(model, 'new_model.pt')
model.eval()
with torch.no_grad():
    amt = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, predicted = torch.max(model(images), 1)
        amt += torch.sum(predicted == labels)
    print((amt/10000)*100)
