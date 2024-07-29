# Standard imports
import cv2
import numpy as np
import sys

# original code : http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html
# other version : https://github.com/TheLaueLab/blob-detection

from math import sqrt
from skimage import data
from skimage.transform import rescale
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import dice

# Function to compute how many black pixels
def pixelStats(groundTruth,result):
    if groundTruth.shape!=result.shape:
        print("Images must be the same shape in pixel difference ")
        return -1

    # this step is just to make sure that both images are adequately binarised (we turn everything over 100 ihn groundTruth into 255 and everything under it to 0, for example)
    groundTruth = 255 * (groundTruth > 100)
    result = 255 * (result > 100)

    totalPixels=groundTruth.shape[0]*groundTruth.shape[1]
    totalBlack= np.count_nonzero(groundTruth == 0)# count the number of black pixels in image groundTruth

    # Find and count the pixels that are zero in both groundTruth and result -> True Positives
    bothBlack = np.count_nonzero(np.logical_and(groundTruth == 0, result == 0))
    # Find and count the pixels that are white in groundTruth but black in result -> False Positives
    whiteBlack = np.count_nonzero(np.logical_and(groundTruth == 255, result == 0))
    TPR=bothBlack/totalBlack
    FPR=whiteBlack/totalBlack
    return TPR,FPR

# receive a binary image, return how many pixels in it are white
def countWhitePixels(im):
    return cv2.countNonZero(im)

#receive two images, return how many pixels are white in both at the same time
def countBothWhite(im1,im2):
    #first, make a copy of the second image
    auxImage=im2.copy()

    #now, all black points in im1 are also painted black in the copy of im2
    auxImage[im1==0]=0

    # now all that is left is counting how many white pixels remain in auxImage (these where white in im2
    # and have not been turned black, so they where also white in im1)
    # we could use count non zero again or we can also use np.sum with a condition
    return np.sum(auxImage == 255)


def SimpleBlobDetector(argv,im):

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 1000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.2

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.25
    params.maxConvexity = 0.75

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    if len(argv)>4: params.minArea=int(argv[4])

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else :
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im)

    #now refine keypoints so overlapping blobs are merged
    overlapThreshold=0.001
    #print ("number of keypoints "+str(len(keypoints)))

    return keypoints

def main(argv):
    # Blod detector tester function
    #parameters:
    #argv[1] contains input file name
    #argv[2] contains type of blob detector: 0 for simple detector, 1 for Laplacian Of Gaussians (LoG), 2 for Difference of Gaussian (DoG), 3 for Determinant of Hessian
    # other parameters are included after、argv[3] and later, check each method

    image_gray=cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        print ('Error opening image: ' + argv[1])
        return -1

    image=image_gray
    image_black=(255-image_gray)

    maskOutputFile=argv[1].split(".")[0]+"detector"+str(argv[2])+"MASK.jpg"
    annotationFile=argv[1].split(".")[0]+"annotation.jpg"

    summaryFile=argv[1].split(".")[0]+"detector"+str(argv[2])+"summary.jpg"

    annotationPresent=True
    annotation=cv2.imread(annotationFile, 0)
    if annotation is None:
        print ('Error opening image: ' + annotationFile)
        annotationPresent=False

    print("Detecting BLOBS with image "+argv[1]+" and detector "+str(argv[2]))

    #switch between the different types of blob detector
    if int(argv[2])==0:
        keypoints=SimpleBlobDetector(argv,image)
        #print ("number of keypoints outside "+str(len(keypoints)))

    elif int(argv[2])==1:
        #parameters: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
        min_s=20
        over=.1
        #example!, other parameters exist
        if len(argv)>3: min_s=int(argv[3])
        if len(argv)>4: over=float(argv[4])
        #print("starting laplacion of gaussians detector with parameters "+str(min_s)+" "+str(over))
        blob_List = blob_log(image_black, min_sigma=min_s,  overlap=over)
        # Compute radii in the 3rd column.
        blob_List[:, 2] = blob_List[:, 2] * sqrt(2)
    elif int(argv[2])==2:
        #parameters: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_dog
        blob_List = blob_dog(image_gray,min_sigma=40,  threshold=1)
        blob_List[:, 2] = blob_List[:, 2] * sqrt(2)
    elif int(argv[2])==3:
        #parameters: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_doh
        min_s=30
        if len(argv)>3: min_s=int(argv[3])

        blob_List = blob_doh(image_gray,min_sigma=min_s, max_sigma=90, num_sigma=35, threshold=.009,overlap=0.3)
    else:
        print("Blob detector type not supported: "+argv[2])
        sys.exit(-1)

    #Now write the results to file,
    #For Scikit detectors
    if int(argv[2])==1 or int(argv[2])==2 or int(argv[2])==3:
        for blob in blob_List:
            y, x, r = blob
            cv2.rectangle(image,(int(x-r), int(y-r)), (int(x+r), int(y+r)), (0, 0, 255), 1)
            #cv2.circle(image,(int(x),int(y)),int(r),(0,0,255))
        cv2.imwrite(summaryFile,image)
    else:
    # For the opencv detector
        for kp in keypoints:
            x = kp.pt[0]
            y = kp.pt[1]
            r = kp.size
            cv2.rectangle(image,(int(x-r), int(y-r)), (int(x+r), int(y+r)), (0, 0, 255), 1)
            #cv2.circle(image,(int(x),int(y)),int(r),(0,0,255))
        plt.imshow(image, cmap = 'gray')
        plt.axis('off')
        # plt.imsave("IM6.png", img, cmap="hsv")
        plt.show()
        cv2.imwrite(summaryFile,image)

    #now make binary result image for automatic comparison
    outputMask=np.ones((image_gray.shape[0],image_gray.shape[1]),np.uint8)
    if int(argv[2])==1 or int(argv[2])==2 or int(argv[2])==3:
        i=0
        #print("Number of Blobs "+str(len(blob_List)))
        for blob in blob_List:
            #print("opencv detector painting blob "+str(i))
            y, x, r = blob
            outputMask[int(y-r):int(y+r),int(x-r):int(x+r)]=255
            i=i+1
    else:
        i=0
        #print("Number of Blobs "+str(len(keypoints)))
        for kp in keypoints:
            #print("scikit detector painting blob "+str(i))
            x = kp.pt[0]
            y = kp.pt[1]
            r = kp.size
            outputMask[int(y-r):int(y+r),int(x-r):int(x+r)]=255
            i=i+1
    invertOutputMask=255-outputMask
    plt.imshow(invertOutputMask, cmap = 'gray')
    plt.axis('off')
        # plt.imsave("IM6.png", img, cmap="hsv")
    plt.show()
    cv2.imwrite(maskOutputFile,invertOutputMask)

    #NOW COMPUTE AND PRINT DICE
    if annotationPresent:
        im1= 255 * (outputMask > 100)
        im2= 255 * ((255-annotation) > 100)

        diceCoeff=dice.dice(im1,im2)
        print("Dice coefficient :"+str(diceCoeff))

    # Finally, compute and print the pixel Difference
    if annotationPresent:
        TPR,FPR=pixelStats(annotation,invertOutputMask)
        print("True Positive Ratio :"+str(TPR))
        print("False Positive Ratio :"+str(FPR))

if __name__ == '__main__':
    main(sys.argv)
