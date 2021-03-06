import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
from personal_utils import get_instance_segmentation_model, open_2_tensor, save_predicted_boxes

torch.cuda.empty_cache()

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    #print(device)
    
    num_classes = 14 #COCO dataset has 91
    model = get_instance_segmentation_model(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9,
                                weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

    #print(model)
    
    try:
        img = open_2_tensor('./TestImages/DSC_0257.JPG')
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])
        
    except:
        print('Using PennFudanPed image')
        img = open_2_tensor('./PennFudanPed/PNGImages/FudanPed00074.png')
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])
        
    plt.imshow(np.asarray(img.mul(255).permute(1, 2, 0).byte().cpu().numpy()))
    plt.show()
    img_path = save_predicted_boxes(prediction, img)

    Image.open(img_path)

    output_info(prediction)

def output_info(prediction):
    detached_prediction = prediction[0]
    print('number of objects detected is: ', np.shape(detached_prediction[0]['boxes']))


if __name__ == "__main__":
    main()