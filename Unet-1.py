from torch._C import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt



class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=0)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0)

        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=0)
        self.conv10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=0)

        self.upconv5 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0,
                                          output_padding=0)

        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=0)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0)

        self.upconv4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0,
                                          output_padding=0)

        self.conv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=0)
        self.conv14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0)

        self.upconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0,
                                          output_padding=0)

        self.conv15 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=0)
        self.conv16 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0)

        self.upconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0,
                                          output_padding=0)

        self.conv17 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=0)
        self.conv18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)

        self.conv19 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, padding=0)

    def crop(self, up, cross):
        lower = int((cross.shape[2] - up.shape[2]) / 2)
        upper = int(cross.shape[2] - lower)
        cross = cross[:, :, lower:upper, lower:upper]
        return cross

    def forward(self, x):
        # 1,2 conv
        con1 = F.relu(self.conv1(x))
        # print(con1.size())
        con2 = F.relu(self.conv2(con1))
        # print(con2.size())

        # maxpool 1
        con2mp = F.max_pool2d(con2, kernel_size=2, stride=2)
        # print(con2mp.size())

        # 3,4 conv
        con3 = F.relu(self.conv3(con2mp))
        # print(con3.size())
        con4 = F.relu(self.conv4(con3))
        # print(con4.size())

        # maxpool 2
        con4mp = F.max_pool2d(con4, kernel_size=2, stride=2)
        # print(con4mp.size())

        # 5,6 conv
        con5 = F.relu(self.conv5(con4mp))
        # print(con5.size())
        con6 = F.relu(self.conv6(con5))
        # print(con6.size())

        # maxpool 3
        con6mp = F.max_pool2d(con6, kernel_size=2, stride=2)
        # print(con6mp.size())

        # 7,8 conv
        con7 = F.relu(self.conv7(con6mp))
        # print(con7.size())
        con8 = F.relu(self.conv8(con7))
        # print(con8.size())

        # maxpool 3
        con8mp = F.max_pool2d(con8, kernel_size=2, stride=2)
        # print(con8mp.size())

        # 9,10 conv
        con9 = F.relu(self.conv9(con8mp))
        # print(con9.size())
        con10 = F.relu(self.conv10(con9))
        # print(con10.size())

        # up5
        up5 = self.upconv5(con10)

        up5con = torch.cat([up5, self.crop(up5, con8)], 1)
        # print(up5con.size())

        # 11,12 conv
        con11 = F.relu(self.conv11(up5con))
        con12 = F.relu(self.conv12(con11))

        # up4
        up4 = self.upconv4(con12)
        up4con = torch.cat([up4, self.crop(up4, con6)], 1)
        # print(up4con.size())

        # 13,14 conv
        con13 = F.relu(self.conv13(up4con))
        con14 = F.relu(self.conv14(con13))

        # up3
        up3 = self.upconv3(con14)
        up3con = torch.cat([up3, self.crop(up3, con4)], 1)
        # print(up3con.size())

        # 15,16 conv
        con15 = F.relu(self.conv15(up3con))
        # print(con15.size())
        con16 = F.relu(self.conv16(con15))
        # print(con16.size())

        # up2
        up2 = self.upconv2(con16)
        # print(up2.size())

        up2con = torch.cat([up2, self.crop(up2, con2)], 1)
        # print(up2con.size())

        # 17,18 conv
        con17 = F.relu(self.conv17(up2con))
        # print(con17.size())
        con18 = F.relu(self.conv18(con17))
        # print(con17.size())

        result = self.conv19(con18)
        # print(result.size())
        return result


unet = Unet()
optimizer = optim.SGD(unet.parameters(), lr=0.01, momentum=0.99)

img = cv2.imread('car.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


pil2tensor = transforms.ToTensor()
img = pil2tensor(img)
img = torch.unsqueeze(img, dim=1)
new_img = unet(img)
print(new_img.size())

im = transforms.ToPILImage()(new_img[0])
#display(im)
print(im)
print(im.size)