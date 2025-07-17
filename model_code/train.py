from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torch
from torch import optim
from tqdm import tqdm
import os

import time

from torchvision import transforms, datasets


start_time = time.time()

datapath = './Train'
validation_path = './validation'

data_trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data = ImageFolder(datapath, transform = data_trans)
validation_data = ImageFolder(validation_path, transform = data_trans)

# getting some data about the images. Showing that ImageFolder extracted images
classes = data.classes
extensions = data.extensions
classindex = data.class_to_idx
samples = data.samples

#Defining hyperparameters
learning_rate = 0.001
batch_size = 16
# looping through these many times to train
num_epochs = 30 #usually 30

train_losses = []
test_losses = []
train_correct = []
test_correct = []


dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(data, batch_size=batch_size, shuffle=True)


device = "cuda" if torch.cuda.is_available() else "cpu"


weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
#load model in future
#model.load_state_dict(torch.load('model.pt'))
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#training

i = 0
highest_accuracy = 0
while True:
    model.train()
    i += 1

    print('starting epoch '+ str(i))
    for batch, (data, target) in enumerate(tqdm(dataloader)):
        #moving data to the GPU
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad() # clears all gradients before backpropogation


        target_prediction = model(data) # getting our cnn model's preduction
        loss = criterion(target_prediction, target) # seeing how far off the answer was to the correct answer (target)

        loss.backward() # calculates the gradients to update weights to minimize loss
        optimizer.step() #updaes the weights based on the gradients in previous line

        if batch % 100 == 0:
            print("Loss value: " + str(loss.item()))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * (correct / total)
    print("Accuracy: " + str(100 * (correct / total)))

    save_path = os.path.join("model", str(i) + '.pt')
    if (accuracy > highest_accuracy):
        highest_accuracy = accuracy
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), save_path)






torch.save(model.state_dict(), "model.pt")

current_time = time.time()
print("time: " + str(current_time - start_time))
print("Training complete")









