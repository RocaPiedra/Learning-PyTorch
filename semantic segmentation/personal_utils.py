import numpy as np
import random
from PIL import Image,ImageDraw
import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import transforms as T


#class for different mask generation
class mask_mod():
  def __init__(self, masks, filter):
    self.masks = masks
    self.filter = filter

  #this function generates a mask for all predictions
  def generate_full_mask(self):
    full_mask = self.masks[0,0]
    for i in range(1,len(self.masks)):
      full_mask = full_mask + self.masks[i,0]
    full_mask[full_mask>=self.filter] = 1
    Image.fromarray(full_mask.mul(255).byte().cpu().numpy())

    return full_mask

  def generate_binary_mask(self):
    full_mask = self.masks[0,0]
    for i in range(1,len(self.masks)):
      full_mask = full_mask + self.masks[i,0]
    full_mask[full_mask>=self.filter] = 1
    full_mask[full_mask<self.filter] = 0
    Image.fromarray(full_mask.mul(255).byte().cpu().numpy())
    return full_mask

#array of mask prediction from GPU to PIL.Image.fromarray readable
def array_to_pillow(array):
  result = array.mul(255).byte().cpu().numpy()
  return result

def HWC2CHW(HWC):
  CHW = np.transpose(HWC, (2,0,1))
  return CHW

#function to get any image ready to the NN
def open_2_tensor(route):
  img = Image.open(route)
  #to numpy array
  img = np.array(img)
  #normalize to 0-1 values
  img = img/255
  #change to channel, height, width
  img = HWC2CHW(img)
  #prepare for gpu as FloatTensor
  img = torch.from_numpy(img)
  img = img.type(torch.cuda.FloatTensor)
  return img

#saves an image of the image with the predicted boxes
def save_predicted_boxes(prediction, img):
  og = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().cpu().numpy())
  vertex = prediction[0]['boxes'].cpu().detach().numpy()
  img_path = 'out_file_boxes.png'
  draw = ImageDraw.Draw(og)
  for v in vertex:
    r = int(random.random()*255)
    b = int(random.random()*255)
    g = int(random.random()*255) 
    color = (r,g,b)
    draw.rectangle(v, fill=None, outline=color, width=10)
  og.save(img_path)
  return img_path


#prepare for custom dataset
def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    
    return model

#transforms image if training
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)