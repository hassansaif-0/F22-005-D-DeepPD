# !export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512
# !apt-get update && apt-get install -y python3-opencv
# !pip install opencv-python
# importing the necessary images
import glob
import cv2
import numpy as np
import  copy
import torch
import torch.nn.functional as F


def segmentImage(anotImages,hedAfterSimo,i,allFmOuputs):  # function to create segments
        print(i)
        merged = torch.zeros((1, 32, 512, 512))

        fullImage = cv2.imread(hedAfterSimo[i])      # read the full image
        # cv2.imwrite("FeatureMapping/1.jpg",fullImage,)  #saving the image

#         print(fullImage.shape)
        remainingFace = copy.deepcopy(fullImage)   # making a deep copy of the full image so that it does not change
        dimColoredImage = (fullImage.shape[0], fullImage.shape[1])   # get the dimensions of the image

        img = cv2.imread(anotImages[i][0],cv2.IMREAD_GRAYSCALE)         # read the anot Image in grayscale form,
        img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)       # make a zero array of the image
        segmentedImages = [img, img, img, img]          # make 4 copies of the zero array , this will store all the segments

        for j in anotImages[i]:     # loop to store the segments
            attribute = j[j.index("_") + 1:j.rindex(".")]       # get the attribite name from the anot image
            if attribute == "r_brow" or attribute =="r_eye":                # from our point left side
                blackAndWhiteImg = cv2.imread(j)            # reading the anot image
                segmentedImages[0] = cv2.bitwise_or(segmentedImages[0],blackAndWhiteImg)        # bitwise or to save the eyebrow and the eye

            elif attribute =="l_brow" or attribute == "l_eye":      # for the left eye
                blackAndWhiteImg = cv2.imread(j)
                segmentedImages[1] = cv2.bitwise_or(segmentedImages[1], blackAndWhiteImg)

            elif attribute =="nose" :
                blackAndWhiteImg = cv2.imread(j)
                segmentedImages[2] = cv2.bitwise_or(segmentedImages[2], blackAndWhiteImg)

            elif attribute == "mouth" or attribute ==" l_lip"  or attribute == "u_lip":
                blackAndWhiteImg = cv2.imread(j)
                segmentedImages[3] = cv2.bitwise_or(segmentedImages[3], blackAndWhiteImg)


        merged = allFmOuputs[4]

        # k==3 mouth
        segmentedImages[3] = cv2.resize(segmentedImages[3], dimColoredImage, interpolation=cv2.INTER_AREA)      # resizing the images to 512
        gray = cv2.cvtColor(segmentedImages[3], cv2.COLOR_BGR2GRAY)         # converting the image to gray scale
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        # get the contours of the iamge
        x, y, w, h = cv2.boundingRect(contours[0])
        crop_img = fullImage[y - 40:y + h + 70, x - 25:x + w + 25]           # cropping the image (manually set)
        allFmOuputs[3] = F.interpolate(allFmOuputs[3], size=(crop_img.shape[0], crop_img.shape[1]), mode='bilinear')
        merged[:,:32,y - 40:y + h + 70, x - 25:x + w + 25  ]=allFmOuputs[3][:,:32,:,:]
        
        #k==2 nose
        segmentedImages[2] = cv2.resize(segmentedImages[2], dimColoredImage, interpolation=cv2.INTER_AREA)      # resizing the images to 512
        gray = cv2.cvtColor(segmentedImages[2], cv2.COLOR_BGR2GRAY)         # converting the image to gray scale
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        # get the contours of the iamge
        x, y, w, h = cv2.boundingRect(contours[0])
        crop_img = fullImage[y - 5:y + h + 35, x - 20:x + w + 20]           # cropping the image (manually set)
        allFmOuputs[2] = F.interpolate(allFmOuputs[2], size=(crop_img.shape[0], crop_img.shape[1]), mode='bilinear')
        merged[:,:32,y - 5:y + h + 35, x - 20:x + w + 20]=allFmOuputs[2][:,:32,:,:]
        
        #k==1 r_eye
        segmentedImages[1] = cv2.resize(segmentedImages[1], dimColoredImage, interpolation=cv2.INTER_AREA)      # resizing the images to 512
        gray = cv2.cvtColor(segmentedImages[1], cv2.COLOR_BGR2GRAY)         # converting the image to gray scale
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        # get the contours of the iamge
        x, y, w, h = cv2.boundingRect(contours[0])
        crop_img = fullImage[y - 50:y + h + 30, x - 30:x + w + 70]        # cropping the image (manually set)
        allFmOuputs[1] = F.interpolate(allFmOuputs[1], size=(crop_img.shape[0], crop_img.shape[1]), mode='bilinear')
        merged[:,:32,y - 50:y + h + 30, x - 30:x + w + 70]=allFmOuputs[1][:,:32,:,:]


         #k==0 l_eye
        segmentedImages[0] = cv2.resize(segmentedImages[0], dimColoredImage, interpolation=cv2.INTER_AREA)      # resizing the images to 512
        gray = cv2.cvtColor(segmentedImages[0], cv2.COLOR_BGR2GRAY)         # converting the image to gray scale
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        # get the contours of the iamge
        x, y, w, h = cv2.boundingRect(contours[0])
        crop_img = fullImage[y - 50:y + h + 30, x - 70:x + w + 30]         # cropping the image (manually set)
        allFmOuputs[0] = F.interpolate(allFmOuputs[0], size=(crop_img.shape[0], crop_img.shape[1]), mode='bilinear')
        merged[:,:32,y - 50:y + h + 30, x - 70:x + w + 30]=allFmOuputs[0][:,:32,:,:]


        return merged

        
        
        # image = merged.squeeze().detach().numpy()
        # cv2.imshow("Image", image[0])
        # # Wait for a key press and then close the window
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        
        