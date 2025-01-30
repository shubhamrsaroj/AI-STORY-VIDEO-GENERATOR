import os
import wget

# Create models directory
os.makedirs('models', exist_ok=True)

# Download Places365 categories
categories_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
wget.download(categories_url, 'models/categories_places365.txt')

# Download Places365 model
model_url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
wget.download(model_url, 'models/resnet50_places365.pth.tar') 