import torch
from PIL import Image
from torchvision import models, transforms
from matplotlib import pyplot as plt
import cv2
import numpy as np



def DAF_SOD(image):                                          #调用即可    输入图片  得到图片
    print(6666)
    net = torch.load("./net.pth", map_location='cpu')

    image_to_tensor = transforms.ToTensor()                         #图片转向量

    image1 = image.resize((64,64))                                   #设置尺寸
    image2 = image1.convert('L')                                     #灰度图
    image1 = image_to_tensor(image1)                                 
    image2 = image_to_tensor(image2)

    image1 = torch.unsqueeze(image1, dim=0)
    image2 = torch.unsqueeze(image2, dim=0)
    print(7777)


    out_image = net(image1,image2,image2)                       #运行网络
    print(8888)
    tensor_to_image = transforms.ToPILImage()                  #向量转图片
    pic = tensor_to_image(out_image[0][0])

    return pic



# img = Image.open('E:\PyProject\sky\TestImage\\7.jpg')
# img_pil = DAF_SOD(img)
# img_pil = img_pil.resize((256,256))
# imgCv = cv2.cvtColor(np.asarray(img_pil),cv2.COLOR_RGB2BGR)
# cv2.imshow('6', imgCv)
# cv2.waitKey()
