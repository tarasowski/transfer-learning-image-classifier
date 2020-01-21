import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from PIL import Image
from . import cli
import json

picture_path, checkpoint_name, top_k, category_names, device = cli.init_predict()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    categories = checkpoint['cat_to_name']

    for param in model.parameters():
        param.requires_grad = False

    return (model, checkpoint['class_to_idx'])

model, class_to_idx = load_checkpoint(f'./models/{checkpoint_name}.pth')


def process_image(image):
    img = Image.open(image)
    resize = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    return resize(img)

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

def predict(image_path, model, categories, topk=5):
    model.to(device)
    model.eval()
    img = process_image(image_path)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    logps = model.forward(img)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    idx = model.class_to_idx
    inverse = {v: k for k, v in idx.items()}
    category_idx = [inverse[y] for x in top_class.tolist() for y in x] 
    category_names = [categories[str(x)] for x in category_idx]
    return (top_p, category_idx, category_names)

def category_json(path):
    f = open(path, 'r').readlines()[0]
    return json.loads(f)
    

categories = category_json(f'./input/{category_names}')

probs, classes, category_names = predict(picture_path, model, categories, top_k)

print([format(y, 'f') 
        for x in probs.tolist()
        for y in x
        ])
print(classes)
print(category_names)
