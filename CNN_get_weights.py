import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. Transformations
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Dataset
train_data = datasets.ImageFolder("dataset/train", transform=transform)
val_data   = datasets.ImageFolder("dataset/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32)

num_classes = len(train_data.classes)

# 3. Charger ResNet50 pré-entraîné
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# 4. Remplacer la couche finale
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    # tqdm sur le DataLoader pour afficher une barre
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Affiche la loss actuelle dans la barre
        loop.set_postfix(loss=total_loss/(loop.n+1))
    
    print(f"Epoch {epoch+1} terminée, loss moyenne: {total_loss/len(train_loader):.4f}")

