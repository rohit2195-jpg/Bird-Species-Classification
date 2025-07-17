from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
from PIL import Image
import time
import torchvision.models as models
import os
from torchvision import transforms, datasets

def get_species(filepath):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(filepath).convert('RGB')
    image = transform(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50()

    num_classes = 12
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load('model/15.pt', map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)

    class_names = sorted(os.listdir("./Test"))
    predicted_class = class_names[predicted.item()]

    return predicted_class

def check_accuracy(filepath):


    start_time = time.time()

    data_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data = ImageFolder(filepath, data_trans)
    dataloader = DataLoader(data, batch_size=32, shuffle=True)
    correct = 0
    total = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, len(data.classes))
    model.to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()



    curr_time = time.time()
    return (correct / total, correct, total, curr_time - start_time)



print(get_species("./7ab3454df9244a15abab3f03bf895e4a.jpg"))
#print("ratio currect, correct, total, time taken", check_accuracy('./Test'))

