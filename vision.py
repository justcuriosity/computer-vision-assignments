import os
from pathlib import Path
import numpy as np
import cv2
from zipfile import ZipFile
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

file_name_zip = "dataset.zip"
with ZipFile(file_name_zip,'r') as  zip:
    zip.extractall();


dataset_path_tr = Path("C:/Users/Suleyman Gursoy/Desktop/vision_1_hw/dataset/train")
dataset_path_te = Path("C:/Users/Suleyman Gursoy/Desktop/vision_1_hw/dataset/query")
images = []
tips = []
tips_number = []
codes = []
filters = []
codes_type = []

for entry in dataset_path_tr.iterdir():
    fileName = str(entry).split('\\')
    tips.append(fileName[len(fileName) - 1])
    path = Path(dataset_path_tr/ entry)
    for item in path.iterdir():
        path_split = str(item).split('\\')
        number_split = path_split[len(path_split)-1].split('_')
        codes_type.append(number_split[0])
        break

def read_images():
    #reads the images in gray form
    index = 0
    for entry in dataset_path_tr.iterdir():
        train_path = Path(dataset_path_tr / entry)
        for item in train_path.iterdir():
            codes.append(index)
            item = str(item)
            im = cv2.imread(item,cv2.IMREAD_GRAYSCALE)
            images.append(im)
        index += 1


def build_filters():
    """ returns a list of kernels in several orientations
    """
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 40):
        params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta , 'lambd':5.0,
                  'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern,params))

def process(img):
    """ returns the img filtered by the filter list
    """
    means = []
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        #print(fimg)
        means.append(np.mean(fimg))

    means = np.array(means)
    return means

def gabor():
    #apply filters to all images
    gabor_applied = []
    for im in images:
        #print(gray)
        gabor_applied.append(process(im))
    gabor_applied  = np.array(gabor_applied)
    return gabor_applied



def gabor_knn(gabor_applied):
    #predict query images and calculate accuracy ratio
    predicted_lst = []
    accurate = []
    codes_np = np.array(codes)
    le = preprocessing.LabelEncoder()
    tips_encoded = le.fit_transform(tips)
    #(gabor_applied.shape)
    #print(codes)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(gabor_applied.reshape(300,40),codes_np.reshape(300))
    for entry in dataset_path_te.iterdir():
        path_split = str(entry).split('\\')
        number_split = path_split[len(path_split) - 1].split('_')
        accurate.append(codes_type.index(number_split[0]))
        entry = str(entry)
        im = cv2.imread(entry)
        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
        predicted = model.predict(process(img_gray).reshape(1,-1))
        predicted_lst.append(predicted[0])
        #print(tips[predicted[0]])
    #print(predicted_lst)
    #print(accurate)
    print("Gabor Filters Accuracy Ratio: ",accuracy_score(accurate,predicted_lst))

#sift part
def feature_vectors(images):
    #finding keypoints and creating descriptors
    siftLst = []
    for img in images:
        sift = cv2.xfeatures2d.SIFT_create()
        keyPoints, des = sift.detectAndCompute(img,None)
        #print(keyPoints,"des:",des)
        nmbr, dimension = des.shape
        key_avrg = []
        for x in range(dimension):
            total =0
            for y in range(nmbr):
                #print(x,"   " ,y)
                total += des[y][x]
            avarage = total / nmbr
            key_avrg.append(avarage)
        siftLst.append(key_avrg)
    siftLst = np.array(siftLst)
    return siftLst

def sift_knn(lst):
    #predict query images, calculate k nearest neighbours, calcualte accuracy score
    predicted_lst = []
    accurate = []
    codes_np = np.array(codes)
    model = KNeighborsClassifier(n_neighbors=50)
    model.fit(lst.reshape(300, 128), codes_np.reshape(300))
    for entry in dataset_path_te.iterdir():
        path_split = str(entry).split('\\')
        number_split = path_split[len(path_split) - 1].split('_')
        accurate.append(codes_type.index(number_split[0]))
        entry = str(entry)
        im = cv2.imread(entry,cv2.IMREAD_GRAYSCALE)
        sift = cv2.xfeatures2d.SIFT_create()
        keyPoints, des = sift.detectAndCompute(im, None)
        # print(keyPoints,"des:",des)
        nmbr, dimension = des.shape
        key_avrg = []
        for x in range(dimension):
            total = 0
            for y in range(nmbr):
                # print(x,"   " ,y)
                total += des[y][x]
            avarage = total / nmbr
            key_avrg.append(avarage)
        key_avrg = np.array(key_avrg)
        predicted = model.predict(key_avrg.reshape(1,-1))
        predicted_lst.append(predicted[0])
    #print(predicted_lst)
    #print(accurate)
    print("Avarage Sift Accuracy Ratio: ",accuracy_score(accurate, predicted_lst))

def create_codebook():
    predicted_lst = []
    accurate = []
    codes_np = np.array(codes)
    descriptor_lst = []
    sift = cv2.xfeatures2d.SIFT_create()

    for img in images:
        keyPoints, des = sift.detectAndCompute(img,None)
        descriptor_lst.append(des)

    kmeans = KMeans(50)
    arr = np.concatenate(descriptor_lst)
    kmeans.fit(arr)
    for entry in dataset_path_te.iterdir():
        path_split = str(entry).split('\\')
        number_split = path_split[len(path_split) - 1].split('_')
        accurate.append(codes_type.index(number_split[0]))
        entry = str(entry)
        im = cv2.imread(entry, cv2.IMREAD_GRAYSCALE)
        sift = cv2.xfeatures2d.SIFT_create()
        keyPoints, des = sift.detectAndCompute(im, None)
        predicted = kmeans.predict(des)
        predicted_lst.append(predicted[0])

    print("Codebook Accuracy Ratio: ",accuracy_score(accurate, predicted_lst))



#MAIN
read_images()
build_filters()
gabor_knn(gabor())
sift_knn(feature_vectors(images))
create_codebook()
