import cv2
import sys
import numpy as np

from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt

class grayscaleImageProcessing:
    def __init__(self,path):
        self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE )
        if self.image is None: raise Exception("grayscaleImageProcessing creator, could not find image ")

    def getImage(self):return self.image
    def setImage(self,im): self.image = im

    def binarize(self, th=150):
        self.binary = self.image.copy()
        self.binary[self.binary > th] = 255
        self.binary[self.binary <= th] = 0

    def save(self,path,im = None):
        if im is None: cv2.imwrite(path,self.image)
        else: cv2.imwrite(path,im)

    def load(self,path):
        self.image = cv2.imread(path)

    def cleanNoisePreProc(self):
        if self.image is None: raise Exception("grayscaleImageProcessing cleanNoise, could not find image ")
        #binarize
        self.binarize()
        # eliminate regions that were binarised as white
        self.image[self.binary == 255] = 255


    # Method to eliminate noisy regions from an image based in connected component analysis
    def cleanNoise(self, params_dict):
        self.cleanNoisePreProc()

        if "th" in params_dict: th = params_dict["th"]
        else: th = 50

        #compute connected components
        numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(255 - self.binary)

        # For every connected component
        for i in range(1,numLabels): #0 contains background
            area=stats[i,cv2.CC_STAT_AREA]

            # Eliminate connected components that are smaller than a threshold
            if area<th: self.image[labelImage == i] = 255


    # Task A, implement a blob detector method using the algorithm that obtained best results in your analysis in Homework 10.
    def detectBlobs(self, params_dict):

        if "min_s" in params_dict: min_s = params_dict["min_s"]
        else:  min_s=20
        over=.5

        #print("starting laplacion of gaussians detector with parameters "+str(min_s)+" "+str(over))
        blob_List = blob_dog(255 - self.image, min_sigma = min_s,  overlap = over)
        # Compute radii in the 3rd column.
        blob_List[:, 2] = blob_List[:, 2] * sqrt(2)


        mask = np.zeros((self.image.shape[0], self.image.shape[1], 1), dtype=np.uint8)
        # write to file
        for blob in blob_List:
            print("blos!")
            y, x, r = blob
            cv2.rectangle(mask,(int(x-r), int(y-r)), (int(x+r), int(y+r)), 255, 1)

        return 255 - mask

# Implement class advGIP here

#TASK B use MSER blob detector to override the detectBlobs method


class advGIP(grayscaleImageProcessing):

    def cleanNoise(self, params_dict):
        self.cleanNoisePreProc()

        if "it" in params_dict: it = params_dict["it"]
        else : it = 1
        if "kernel" in params_dict: kernel = params_dict["kernel"]
        else: kernel = np.ones((5,5),np.uint8)

        first = cv2.erode(cv2.bitwise_not(self.image),kernel,iterations = it)

        self.image = cv2.bitwise_not(cv2.dilate(first,kernel,iterations = it))



    def detectBlobs(self, params_dict):
        #TASK B use MSER blob detector
        img = self.getImage()
        mser = cv2.MSER_create()

        #Adjust parameters Here using params_dict,
        # try different parameters and different values for each parameter

        #detect regions in gray scale image
        regions, _ = mser.detectRegions(img)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        mask=255-mask

        for contour in hulls:
            cv2.drawContours(mask, [contour], -1, (0, 0, 0), -1)

        return mask

# https://en.wikipedia.org/wiki/Maximally_stable_extremal_regions

def main(argv):

    processing = grayscaleImageProcessing(sys.argv[1])

    params_dict1 = {"th" : 40}
    processing.cleanNoise(params_dict1)
    processing.save("denoisedBasic.jpg")

    # No need to run the process every time, you can also just load the image
    #processing.load("denoised.jpg")


    #TASK A implement the "detecBlobs function for class "grayscaleImageProcessing"
    params_dict2 = { "min_s" : 50 }
    blobsBasicImage = processing.detectBlobs(params_dict2)
    processing.save("BlobsBasic.jpg",blobsBasicImage)

    # now define advances processing class
    advProc = advGIP(sys.argv[1])

    params_dict3 = { "it" : 1 , "kernel": np.ones((3,3),np.uint8)}
    advProc.cleanNoise(params_dict3)
    advProc.save("denoisedAdvanced.jpg")

    blobsAdvImage = advProc.detectBlobs(params_dict2)
    advProc.save("BlobsADV.jpg",blobsAdvImage)





if __name__ == '__main__':
    main(sys.argv)
