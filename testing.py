# import requests
# response = requests.post(
# 'https://www.cutout.pro/api/v1/matting?mattingType=18',
# files={'file': open('faizan.png', 'rb')},
# headers={'APIKEY': '7ec9b38ba5924c308314c88b5a15bb68'},
# )
# with open('out.png', 'wb') as out:
#     out.write(response.content)


# testing

import glob
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import networks
import ImageSynthisis as IS
import newSegmentPlaces as segImages

device ="cpu"
if torch.cuda.is_available():
    device ="cuda"


def getAnotDict(anotPath):  # function to get all annotation images in a dictionary
    anotImages = glob.glob(f"{anotPath}/*.png")  # get all images
    anotImagesDict = {}     # create a empty dictionary
    for image in anotImages:
        oneImage = image[image.rindex("/") + 1:]      # get the key (it will be a int number) +extension
        if int(oneImage[:oneImage.index("_")]) not in anotImagesDict:   # get the index and checking if it already exists
            anotImagesDict[int(oneImage[:oneImage.index("_")])] = []        # if not then create an empty list for that image
        anotImagesDict[int(oneImage[:oneImage.index("_")])].append(image)   # add anot image to that list with index
    return anotImagesDict

def resizedHed(hedPath):    # function to get all the images
    mainImages = glob.glob(rf"{hedPath}/*.jpg")     # getting all images names and paths
    allImages = {}  # create a empty dictionary
    for image in mainImages:
        index = int(image[image.rindex("/") + 1:-4])   # get the index of the image (number)
        allImages[index] = image       # for the index, save the complete path of the image
    return allImages


anotImages=None
hedAfterSimo = None


def getSegments(Path):
    segments = glob.glob(Path)
    segmentsDict = {}
    for image in segments:
        index = int(image[image.rindex("/") + 1:-4])
        segmentsDict[index] = image
    return segmentsDict


anotImages=getAnotDict(r"static/anotImages/AnotImages")
print(len(anotImages))
hedAfterSimo=resizedHed(r"static/sketches(WhiteBackground)/test/test")
print(len(hedAfterSimo))
realImages = resizedHed(r"static/realImages/test/test")


l_eyeSegments=getSegments(r"static/Segments/test/l_eye/l_eye/*.jpg")
r_eyeSegments=getSegments(r"static/Segments/test/r_eye/r_eye/*.jpg")
noseSegments=getSegments(r"static/Segments/test/nose/nose/*.jpg")
mouthSegments=getSegments(r"static/Segments/test/mouth/mouth/*.jpg")
remainingSegments=getSegments(r"static/Segments/test/remaining/remaining/*.jpg")


l_eyeEncoder =networks.define_part_encoder("eye")
l_eyeEncoder.load_state_dict(torch.load(r"static/Models/l_eye.pt"))
# l_eyeEncoder.eval()

r_eyeEncoder = networks.define_part_encoder("eye")
r_eyeEncoder.load_state_dict(torch.load(r"static/Models/r_eye.pt"))
# r_eyeEncoder.eval()

noseEncoder = networks.define_part_encoder("nose")
noseEncoder.load_state_dict(torch.load(r"static/Models/nose.pt"))
# noseEncoder.eval()

mouthEncoder = networks.define_part_encoder("mouth")
mouthEncoder.load_state_dict(torch.load(r"static/Models/mouth.pt"))
# mouthEncoder.eval()

remainingEncoder = networks.define_part_encoder("face")
remainingEncoder.load_state_dict(torch.load(r"static/Models/remaining.pt"))
# remainingEncoder.eval()


l_eyeFM =networks.featureMapping("eye")
l_eyeFM.load_state_dict(torch.load(r"static/Models/l_eyeFM-5-30000.pth",map_location=torch.device("cpu")))
# l_eyeFM.eval()

r_eyeFM =networks.featureMapping("eye")
r_eyeFM.load_state_dict(torch.load(r"static/Models/r_eyeFM-5-30000.pth",map_location=torch.device('cpu')))
# r_eyeFM.eval()

noseFM =networks.featureMapping("nose")
noseFM.load_state_dict(torch.load(r"static/Models/noseFM-5-30000.pth",map_location=torch.device('cpu')))
# noseFM.eval()

mouthFM =networks.featureMapping("mouth")
mouthFM.load_state_dict(torch.load(r"static/Models/mouthFM-5-30000.pth",map_location=torch.device('cpu')))
# mouthFM.eval()

remainingFM =networks.featureMapping("remaining")
remainingFM.load_state_dict(torch.load(r"static/Models/remainingFM-5-30000.pth",map_location=torch.device('cpu')))
# remainingFM.eval()

transformFun = transforms.Compose([
        transforms.Grayscale(),     # converting to grayscale, so that we can get 1 channel
        transforms.ToTensor()   # converting to grayscale
    ])


imageGan = IS.GanModule()
imageGan.G.load_state_dict(torch.load(r"static/Models/generator-5-30000.pth", map_location=torch.device('cpu')))
imageGan.D1.load_state_dict(torch.load(r"static/Models/D1-5-30000.pth", map_location=torch.device('cpu')))
imageGan.D2.load_state_dict(torch.load(r"static/Models/D2-5-30000.pth", map_location=torch.device('cpu')))
imageGan.D3.load_state_dict(torch.load(r"static/Models/D3-5-30000.pth", map_location=torch.device('cpu')))

# imageGan.eval()


import cv2

transTensor = transforms.Compose([
        transforms.ToTensor()   # converting to grayscale
    ])


l_eyeSeg = Image.open(r"static/Segments/test/l_eye/l_eye/10.jpg")
l_eyeTen = transformFun(l_eyeSeg)
l_eyeCEInput=l_eyeTen.unsqueeze(0)
l_eyeCEOuput = l_eyeEncoder(l_eyeCEInput)
l_eyeFMOuput = l_eyeFM(l_eyeCEOuput)
print(l_eyeFMOuput.shape)


r_eyeSeg = Image.open(r"static/Segments/test/r_eye/r_eye/10.jpg")
r_eyeTen = transformFun(r_eyeSeg)
r_eyeCEInput=r_eyeTen.unsqueeze(0)
r_eyeCEOuput = r_eyeEncoder(r_eyeCEInput)
r_eyeFMOuput = r_eyeFM(r_eyeCEOuput)
print(r_eyeFMOuput.shape)


noseSeg = Image.open(r"static/Segments/test/nose/nose/10.jpg")
noseTen = transformFun(noseSeg)
noseCEInput=noseTen.unsqueeze(0)
noseCEOuput = noseEncoder(noseCEInput)
noseFMOuput = noseFM(noseCEOuput)
print(noseFMOuput.shape)


mouthSeg = Image.open(r"static/Segments/test/mouth/mouth/10.jpg")
mouthTen = transformFun(mouthSeg)
mouthCEInput=mouthTen.unsqueeze(0)
mouthCEOuput = mouthEncoder(mouthCEInput)
mouthFMOuput = mouthFM(mouthCEOuput)
print(mouthFMOuput.shape)

remainingSeg = Image.open(r"static/Segments/test/remaining/remaining/10.jpg")
remainingTen = transformFun(remainingSeg)
remainingCEInput=remainingTen.unsqueeze(0)
remainingCEOuput = remainingEncoder(remainingCEInput)
remainingFMOuput = remainingFM(remainingCEOuput)
print(remainingFMOuput.shape)


allFMOutputs = [l_eyeFMOuput,r_eyeFMOuput,noseFMOuput,mouthFMOuput,remainingFMOuput]


mergedOuput = segImages.segmentImage(anotImages,hedAfterSimo,10,allFMOutputs)
print("merged")

GanOutput = imageGan.generate(mergedOuput)
print("output generated")
image = cv2.cvtColor(GanOutput.clone().cpu().squeeze(0).permute(1,2,0).detach().numpy(),cv2.COLOR_RGB2BGR)
plt.imshow(image)
plt.show()
