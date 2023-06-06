from flask import Flask, render_template, jsonify, request, g
import base64
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import glob
import joblib
from io import BytesIO
from PIL import Image
import glob
import copy
import torch
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from PIL import Image
import networks
import ImageSynthisis as IS
import newSegmentPlaces as segImages


transformFun = transforms.Compose([
        transforms.Grayscale(),     # converting to grayscale, so that we can get 1 channel
        transforms.ToTensor()   # converting to grayscale
    ])



male_labels = np.array(glob.glob(r"static\sketchShadowKnnSmall\male\*.jpg"))
female_labels = np.array(glob.glob(r"static\sketchShadowKnnSmall\female\*.jpg"))

train_labels = male_labels	# to give some value initially
image_no = 0
anotImages=None
hedAfterSimo = None

# =========================================================================================
# =========================================================================================
# =========================================================================================


alg = cv2.KAZE_create()
print("hello")
print("hello1")
knnMale = joblib.load("maleShadowKNN.joblib")
knnFemale = joblib.load("femaleShadowKNN.joblib")

l_eyeEncoder =None
r_eyeEncoder = None
noseEncoder = None
mouthEncoder = None
remainingEncoder = None

l_eyeFM =None
r_eyeFM =None
noseFM =None
mouthFM =None
remainingFM =None

imageGan = None

knn=knnMale

# =========================================================================================
def segmentImage(anotImages,hedAfterSimo,i):
    
    print("in segmentation" , i)
    img = plt.imread("image.png",cv2.IMREAD_UNCHANGED)
    alpha = img[:, :, 3]
    otherImage = np.zeros_like(img[:, :, :3])
    otherImage[alpha == 0] = [0, 0, 0]
    otherImage[alpha != 0] = [255, 255, 255]
    otherImage=cv2.cvtColor(otherImage, cv2.COLOR_BGR2GRAY)
    print(otherImage.shape)

    fullImage = cv2.imread(hedAfterSimo[i])      # read the full image
    remainingFace = copy.deepcopy(otherImage)   # making a deep copy of the full image so that it does not change
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

    allFMOutputs = [None,None,None,None,None]
    for k in range(4):
        segmentedImages[k] = cv2.resize(segmentedImages[k], dimColoredImage, interpolation=cv2.INTER_AREA)      # resizing the images to 512
        gray = cv2.cvtColor(segmentedImages[k], cv2.COLOR_BGR2GRAY)         # converting the image to gray scale
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        # get the contours of the iamge
        x, y, w, h = cv2.boundingRect(contours[0])      # getting the bounding box cordinates of the contour

        if k==0 or k ==1 :    # eyes   0 =left   1==right     (from our prespective)

            if k==0:        # left eye
                crop_img = otherImage[y - 50:y + h + 30, x - 70:x + w + 30]      # cropping the image (manually set)
                remainingFace[y - 50:y + h + 30, x - 70:x + w + 30] = 0                # setting the box from the boudning box to 255 , so that we can remove that segment
                crop_img=cv2.resize(crop_img, (128,128), interpolation=cv2.INTER_AREA)      # resizing
                crop_img = Image.fromarray(crop_img)
                l_eyeTen = transformFun(crop_img)
                l_eyeCEInput=l_eyeTen.unsqueeze(0)
                l_eyeCEOuput = l_eyeEncoder(l_eyeCEInput)
                l_eyeFMOuput = l_eyeFM(l_eyeCEOuput)
                allFMOutputs[0]=l_eyeFMOuput
                print(l_eyeFMOuput.shape)

            else:
                crop_img = otherImage[y - 50:y + h + 30, x - 30:x + w + 70]           # cropping the image (manually set)
                remainingFace[y - 50:y + h + 30, x - 30:x + w + 70] = 0           # setting the box from the boudning box to 255 , so that we can remove that segment
                crop_img=cv2.resize(crop_img, (128,128), interpolation=cv2.INTER_AREA)
                crop_img = Image.fromarray(crop_img)
                r_eyeTen = transformFun(crop_img)
                r_eyeCEInput=r_eyeTen.unsqueeze(0)
                r_eyeCEOuput = r_eyeEncoder(r_eyeCEInput)
                r_eyeFMOuput = r_eyeFM(r_eyeCEOuput)
                allFMOutputs[1]=r_eyeFMOuput
                print(r_eyeFMOuput.shape)



                # cv2.imwrite("r_eye.jpg",crop_img,)  #saving the image#saving the image

        if k==2:   # nose
            crop_img = otherImage[y - 5:y + h + 35, x - 20:x + w + 20]            # cropping the image (manually set)
            remainingFace[y - 5:y + h + 35, x - 20:x + w + 20] = 0         # setting the box from the boudning box to 255 , so that we can remove that segment
            crop_img = cv2.resize(crop_img, (160, 160), interpolation=cv2.INTER_AREA)
            crop_img = Image.fromarray(crop_img)
            noseTen = transformFun(crop_img)
            noseCEInput=noseTen.unsqueeze(0)
            noseCEOuput = noseEncoder(noseCEInput)
            noseFMOuput = noseFM(noseCEOuput)
            allFMOutputs[2]=noseFMOuput
            print(noseFMOuput.shape)
            # cv2.imwrite(rf"nose.jpg", crop_img, )  #saving the image


        if k==3: # mouth
            crop_img = otherImage[y - 40:y + h + 70, x - 25:x + w + 25]           # cropping the image (manually set)
            remainingFace[y - 40:y + h + 70, x - 25:x + w + 25] = 0         # setting the box from the boudning box to 255 , so that we can remove that segment
            crop_img = cv2.resize(crop_img, (192, 192), interpolation=cv2.INTER_AREA)
            crop_img = Image.fromarray(crop_img)
            mouthTen = transformFun(crop_img)
            mouthCEInput=mouthTen.unsqueeze(0)
            mouthCEOuput = mouthEncoder(mouthCEInput)
            mouthFMOuput = mouthFM(mouthCEOuput)
            allFMOutputs[3]=mouthFMOuput
            print(mouthFMOuput.shape)
            # cv2.imwrite(rf"mouth.jpg", crop_img, )  #saving the image

    remainingFace = cv2.resize(remainingFace, (512, 512), interpolation=cv2.INTER_AREA)
    crop_img = Image.fromarray(remainingFace)
    remainingTen = transformFun(crop_img)
    remainingCEInput=remainingTen.unsqueeze(0)
    remainingCEOuput = remainingEncoder(remainingCEInput)
    remainingFMOuput = remainingFM(remainingCEOuput)
    allFMOutputs[4]=remainingFMOuput

    
    mergedOuput = segImages.segmentImage(anotImages,hedAfterSimo,i,allFMOutputs)
    GanOutput = imageGan.generate(mergedOuput)
    numpy_image = GanOutput.squeeze().cpu().detach().numpy()

    # Convert from RGB to BGR
    numpy_image = numpy_image[..., ::-1]

    # Rescale the values to [0, 255] and convert to uint8
    numpy_image = (numpy_image - np.min(numpy_image)) / (np.max(numpy_image) - np.min(numpy_image)) * 255
    numpy_image = numpy_image.astype(np.uint8)

    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(np.transpose(numpy_image, (1, 2, 0)))

    pil_image.save('image.png')

        
    cv2_image = cv2.imread('image.png')
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static\images\out.png', rgb_image)

    import requests
    response = requests.post(
        'https://www.cutout.pro/api/v1/matting?mattingType=18',
        files={'file': open('static\images\out.png', 'rb')},
        headers={'APIKEY': 'ac4ffbe781b543609445b9f04d7e083f'},
    )
    with open('static\images\out.png', 'wb') as out:
        out.write(response.content)


    


    
    print("loaded models")

        # return merged
# =========================================================================================
# =========================================================================================
def resizedHed(hedPath):    # function to get all the images
    mainImages = glob.glob(fr"{hedPath}\*.jpg")     # getting all images names and paths
    allImages = {}  # create a empty dictionary
    for image in mainImages:
        # print(image)
        index = int(image[image.rindex("\\") + 1:-4])   # get the index of the image (number)
        allImages[index] = image       # for the index, save the complete path of the image
    return allImages
# =========================================================================================

# =========================================================================================
def getAnotDict(anotPath):  # function to get all annotation images in a dictionary
    anotImages = glob.glob(f"{anotPath}\*.png")  # get all images
    anotImagesDict = {}     # create a empty dictionary
    for image in anotImages:
        oneImage = image[image.rindex("\\") + 1:]      # get the key (it will be a int number) +extension
        if int(oneImage[:oneImage.index("_")]) not in anotImagesDict:   # get the index and checking if it already exists
            anotImagesDict[int(oneImage[:oneImage.index("_")])] = []        # if not then create an empty list for that image
        anotImagesDict[int(oneImage[:oneImage.index("_")])].append(image)   # add anot image to that list with index
    return anotImagesDict
# =========================================================================================
def getSegments(Path):
    segments = glob.glob(Path)
    segmentsDict = {}
    for image in segments:
        index = int(image[image.rindex("/") + 1:-4])
        segmentsDict[index] = image
    return segmentsDict


# =========================================================================================
# Extract features from the images
def extract_features(image,alg, vector_size=32):
	try:	
		kps = alg.detect(image)
		kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
		kps, dsc = alg.compute(image, kps)
		if(len(kps) != 0):
			dsc = dsc.flatten()
			needed_size = (vector_size * 64)
			if dsc.size < needed_size:
				dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
			dsc = dsc[:2048]
		else:
			return(np.zeros(vector_size * 64))
	except cv2.error as e:
		print('Error: ', e)
		return None
	idx = dsc.argsort()[-1:-411:-1]
	binarize_dsc = np.zeros(2048)
	binarize_dsc[idx[:]] = 1
	return(binarize_dsc)
# =========================================================================================

# =========================================================================================
# Fit the knn model using the extracted features
def fit_knn(train_data, train_labels):
	knn = KNeighborsClassifier(metric='cosine')
	knn.fit(train_data, train_labels)
	return knn
# =========================================================================================

# =========================================================================================
# Predict the closest neighbor using the knn model
def predict_neighbors(knn, test_data, k):
	distances, indices = knn.kneighbors(test_data, n_neighbors=20)
	neighbor_labels = train_labels[indices]
	return distances, neighbor_labels
# =========================================================================================

# =========================================================================================
def get_train_data(folder_path, alg):
	allImages = glob.glob(folder_path + "\*.jpg")
	train_data=[]
	train_labels= []
	
	count=0
	for i in allImages:
		img=cv2.imread(i)
		img = cv2.resize(img,(512, 512))
		features = extract_features(img, alg)
		train_data.append(features)
		train_labels.append(i)
		# print(i)
		count+=1
		print(count)

	return np.array(train_data), np.array(train_labels)
# =========================================================================================

# =========================================================================================
# training()
def start(image):
    global image_no

    img = cv2.resize(image, (512, 512))

    feature = extract_features(img, alg)
    distances, closestLabels = predict_neighbors(knn, np.array([feature]), k=10)

    final_image_created = np.zeros(shape=[512, 512])
    blur = np.zeros((512, 512), dtype = np.int32)
    most_matched_image = np.zeros((512, 512), dtype = np.int32)
    new_image_formed = np.zeros(shape=[512, 512])
    most_matched_image = cv2.imread(closestLabels[0][-1],cv2.IMREAD_GRAYSCALE)		


    for im in range(10):
        imgClose = cv2.imread(closestLabels[0][im],cv2.IMREAD_GRAYSCALE)	
        # print(closestLabels[0][im])
        mask = imgClose < 200
        new_image_formed[np.where(mask)] += ((2.31)**(-1*distances[0][im]))

    total_match_value = np.sum(np.power(2.31,-1*distances[0]))
    final_image_created = (1 - new_image_formed / total_match_value) * 255

    final_image_created = final_image_created.astype(np.uint8)

    blur = cv2.blur(final_image_created,(15,15))
                    
    dst = cv2.addWeighted(blur, 0.8, most_matched_image, 0.2, 0, dtype=cv2.CV_32S)	

    print("Closest matched sketch : ",closestLabels[0][0])	# this will show whole path
    full_image_name = closestLabels[0][0].split('\\')[-1]	# this will get whole image with extension
    print(full_image_name)
    image_no = int(full_image_name.split('.')[0])
    print(image_no)
    return dst
# =========================================================================================



def load_EncoderModel(model_path, model_name):
    model = networks.define_part_encoder(model_name)
    model.load_state_dict(torch.load(model_path))
    return model




# =========================================================================================
# =========================================================================================
# =========================================================================================
app = Flask(__name__)


@app.route('/')
def index():
    global hedAfterSimo
    global anotImages
    global l_eyeEncoder
    global r_eyeEncoder
    global noseEncoder
    global mouthEncoder
    global remainingEncoder

    global l_eyeFM 
    global r_eyeFM 
    global noseFM 
    global mouthFM
    global remainingFM

    global imageGan 

    hedAfterSimo = resizedHed(r'static/hedAfterSimo')
    anotImages = getAnotDict(r'static/AnotImages')

    l_eyeEncoder =networks.define_part_encoder("eye")
    l_eyeEncoder.load_state_dict(torch.load(r"static/Models/l_eye.pt"))

    r_eyeEncoder = networks.define_part_encoder("eye")
    r_eyeEncoder.load_state_dict(torch.load(r"static/Models/r_eye.pt"))

    noseEncoder = networks.define_part_encoder("nose")
    noseEncoder.load_state_dict(torch.load(r"static/Models/nose.pt"))

    mouthEncoder = networks.define_part_encoder("mouth")
    mouthEncoder.load_state_dict(torch.load(r"static/Models/mouth.pt"))

    remainingEncoder = networks.define_part_encoder("face")
    remainingEncoder.load_state_dict(torch.load(r"static/Models/remaining.pt"))

    
    l_eyeFM =networks.featureMapping("eye")
    l_eyeFM.load_state_dict(torch.load(r"static/Models/l_eyeFM-5-30000.pth",map_location=torch.device("cpu")))
    

    r_eyeFM =networks.featureMapping("eye")
    r_eyeFM.load_state_dict(torch.load(r"static/Models/r_eyeFM-5-30000.pth",map_location=torch.device('cpu')))
    
    noseFM =networks.featureMapping("nose")
    noseFM.load_state_dict(torch.load(r"static/Models/noseFM-5-30000.pth",map_location=torch.device('cpu')))
    
    mouthFM =networks.featureMapping("mouth")
    mouthFM.load_state_dict(torch.load(r"static/Models/mouthFM-5-30000.pth",map_location=torch.device('cpu')))
    
    remainingFM =networks.featureMapping("remaining")
    remainingFM.load_state_dict(torch.load(r"static/Models/remainingFM-5-30000.pth",map_location=torch.device('cpu')))
    
    
    imageGan = IS.GanModule()
    imageGan.G.load_state_dict(torch.load(r"static/Models/generator-5-30000.pth", map_location=torch.device('cpu')))

    print("loaded")
    
    
    
    return render_template('index.html')

@app.route('/generateimage')
def gotogenerate():
    
	print("image no : ",image_no)
	segmentImage(anotImages,hedAfterSimo,image_no)  # function to create segments

	return render_template('generate.html')


@app.route('/mainpage')
def conttomain():
    return render_template('mainfile.html')



train_labels = male_labels	# to give some value initially

@app.route("/change_label",methods=["POST"])
def change_label():
	global train_labels
	global knn

	data = request.get_json()
	# when user clicks on male or female button, a boolean is set and here is its value
	# true for male and false for female
	is_male = data['isMale']	
	if is_male==1:
		train_labels=male_labels
		knn=knnMale
	else:
		train_labels=female_labels
		knn=knnFemale

	# print(train_labels)
	return "Changed label"



@app.route("/update_shadow", methods=["POST"])
def update_shadow():
    # print("PRINT LABELS SKETCH", train_labels)
    dataURL = request.form.get("image")
    data = dataURL.split(',')[1]
    data = base64.b64decode(data)

    # Save the image
    with open("image.png", "wb") as f:
        f.write(data)
    
    img = plt.imread("image.png",cv2.IMREAD_UNCHANGED)

    alpha = img[:, :, 3]
    otherImage = np.zeros_like(img[:, :, :3])

    otherImage[alpha == 0] = [255, 255, 255]
    otherImage[alpha != 0] = [0, 0, 0]

    otherImage=cv2.cvtColor(otherImage, cv2.COLOR_BGR2GRAY)
 
    shadowimage = start(otherImage)
    shadowimage =cv2.merge((shadowimage,shadowimage,shadowimage))   
    # plt.imshow(shadowimage)
    # plt.show()

    _, buffer = cv2.imencode('.png', shadowimage)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return image_base64


if __name__ == '__main__':
    app.run()
