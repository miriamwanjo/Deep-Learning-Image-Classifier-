import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import json
from PIL import Image
from torch.autograd import Variable

def parse_args():
    parser = argparse.ArgumentParser(description='Predicting image')
    parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--gpu', type=bool, default=False, help='device gpu or cpu')
    parser.add_argument('--cat_to_name', default='cat_to_name.json', action ='store', help="directory of flower names")
    parser.add_argument('--topk', type=int, dest='topk', action='sote', default=5, help="top classes")
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth')
    return parser.parse_args()
# image preprocessing
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array'''
    
    preprocess = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])
    from PIL import Image
    pil_image = Image.open(image)
    pil_image = preprocess(pil_image)
    
    np_image = np.array(pil_image)    

# class prediction 
def predict(image_path, model, class_to_idx, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file'''
    image = Image.open(image_path)
    image = process_image(image_path)
    print(image.shape)
    image = torch.FloatTensor(image)
    image = image.unsqueeze_(0) #since we are using only one image hence need to vonvert the dimension to (1, 224, 224, 3)
    
    
    output = model(image)
    ps = torch.exp(output)
     
    probs, top_class = torch.topk(ps,topk)
    
    return probs, top_class
    return np_image
    probs, classes = predict(image_path, model)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    prob_arr = probs.data.numpy()[0]
    pred_indexes = classes.data.numpy()[0].tolist()    
    pred_labels = [idx_to_class[x] for x in pred_indexes]
    pred_class = [cat_to_name[str(x)] for x in pred_labels]
def main():
    args = parse_args()
    gpu = args.gpu
    image_path = './flowers/test/' + str(img_num) + '/' + image  
    image_path = args.image_path
    model.class_to_idx, idx_to_class = checkpoint(args)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    probs, top_class = predict(image_path, model, int(args.topk), gpu)
    print('probs', 'top_class')
                          
if __name__ == "__main__":
    main()