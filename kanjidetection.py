# 標準ライブラリのインポート
import cv2
import numpy as np
import sys
from math import sqrt
from skimage import data
from skimage.transform import rescale
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import dice

# 黒いピクセルの数を計算する関数
def pixelStats(groundTruth, result):
    if groundTruth.shape != result.shape:
        print("画像はピクセル数が同じでなければなりません")
        return -1

    # groundTruthとresultを二値化する（例として100より大きいものを255にし、それ以下を0にする）
    groundTruth = 255 * (groundTruth > 100)
    result = 255 * (result > 100)

    totalPixels = groundTruth.shape[0] * groundTruth.shape[1]
    totalBlack = np.count_nonzero(groundTruth == 0)  # groundTruth内の黒いピクセルの数をカウントする

    # groundTruthとresultの両方で黒いピクセルの数をカウントする -> 真陽性
    bothBlack = np.count_nonzero(np.logical_and(groundTruth == 0, result == 0))
    # groundTruthが白く、resultが黒いピクセルの数をカウントする -> 偽陽性
    whiteBlack = np.count_nonzero(np.logical_and(groundTruth == 255, result == 0))
    TPR = bothBlack / totalBlack
    FPR = whiteBlack / totalBlack
    return TPR, FPR

# 二値画像を受け取り、その中の白いピクセルの数を返す
def countWhitePixels(im):
    return cv2.countNonZero(im)

# 2つの画像を受け取り、両方で白いピクセルの数を返す
def countBothWhite(im1, im2):
    # まず、2つ目の画像のコピーを作成
    auxImage = im2.copy()

    # 次に、im1の黒いポイントをauxImageも黒くする
    auxImage[im1 == 0] = 0

    # auxImageに残っている白いピクセルの数をカウントする
    # これはim2で白く、im1でも白かったピクセル
    return np.sum(auxImage == 255)

def SimpleBlobDetector(argv, im):
    # SimpleBlobDetectorのパラメータを設定
    params = cv2.SimpleBlobDetector_Params()

    # 閾値を変更
    params.minThreshold = 10
    params.maxThreshold = 200

    # 面積でフィルタリング
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 1000

    # 円形度でフィルタリング
    params.filterByCircularity = True
    params.minCircularity = 0.2

    # 凸性でフィルタリング
    params.filterByConvexity = False
    params.minConvexity = 0.25
    params.maxConvexity = 0.75

    # 慣性でフィルタリング
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    if len(argv) > 4:
        params.minArea = int(argv[4])

    # バージョンによってdetectorを作成
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # ブロブを検出
    keypoints = detector.detect(im)

    # 重なり合うブロブをマージするためのキー点を精査
    overlapThreshold = 0.001
    # print ("キー点の数 " + str(len(keypoints)))

    return keypoints

def main(argv):
    # ブロブ検出テスト関数
    # パラメータ:
    # argv[1] 入力ファイル名
    # argv[2] ブロブ検出器のタイプ: 0 = SimpleDetector, 1 = Laplacian Of Gaussians (LoG), 2 = Difference of Gaussian (DoG), 3 = Determinant of Hessian
    # 他のパラメータはargv[3]以降に含まれる、それぞれの手法に応じてチェック

    image_gray = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        print('画像の読み込みエラー: ' + argv[1])
        return -1

    image = image_gray
    image_black = (255 - image_gray)

    maskOutputFile = argv[1].split(".")[0] + "detector" + str(argv[2]) + "MASK.jpg"
    annotationFile = argv[1].split(".")[0] + "annotation.jpg"
    summaryFile = argv[1].split(".")[0] + "detector" + str(argv[2]) + "summary.jpg"

    annotationPresent = True
    annotation = cv2.imread(annotationFile, 0)
    if annotation is None:
        print('アノテーション画像の読み込みエラー: ' + annotationFile)
        annotationPresent = False

    print("画像 " + argv[1] + " と検出器 " + str(argv[2]) + " でBLOBSを検出中")

    # 異なるブロブ検出器の間を切り替え
    if int(argv[2]) == 0:
        keypoints = SimpleBlobDetector(argv, image)
        # print ("外部キー点の数 " + str(len(keypoints)))

    elif int(argv[2]) == 1:
        # パラメータ: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
        min_s = 25
        over = .05
        # 例、他のパラメータも存在
        if len(argv) > 3:
            min_s = int(argv[3])
        if len(argv) > 4:
            over = float(argv[4])
        blob_List = blob_log(image_black, min_sigma=min_s, overlap=over)
        blob_List[:, 2] = blob_List[:, 2] * sqrt(2)
    elif int(argv[2]) == 2:
        # パラメータ: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_dog
        blob_List = blob_dog(image_gray, min_sigma=30, threshold=0.08)
        blob_List[:, 2] = blob_List[:, 2] * sqrt(2)
    elif int(argv[2]) == 3:
        # パラメータ: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_doh
        min_s = 30
        if len(argv) > 3:
            min_s = int(argv[3])
        blob_List = blob_doh(image_gray, min_sigma=min_s, max_sigma=80, num_sigma=40, threshold=.010, overlap=0.3)
    else:
        print("サポートされていないブロブ検出器タイプ: " + argv[2])
        sys.exit(-1)

    # 結果をファイルに書き込み
    # Scikitの検出器の場合
    if int(argv[2]) == 1 or int(argv[2]) == 2 or int(argv[2]) == 3:
        for blob in blob_List:
            y, x, r = blob
            cv2.rectangle(image, (int(x - r), int(y - r)), (int(x + r), int(y + r)), (0, 0, 255), 1)
            # cv2.circle(image, (int(x), int(y)), int(r), (0, 0, 255))
        cv2.imwrite(summaryFile, image)
    else:
        # OpenCVの検出器の場合
        for kp in keypoints:
            x = kp.pt[0]
            y = kp.pt[1]
            r = kp.size
            cv2.rectangle(image, (int(x - r), int(y - r)), (int(x + r), int(y + r)), (0, 0, 255), 1)
            # cv2.circle(image, (int(x), int(y)), int(r), (0, 0, 255))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        # plt.imsave("IM6.png", img, cmap="hsv")
        plt.show()
        cv2.imwrite(summaryFile, image)

    # 自動比較用にバイナリ結果画像を作成
    outputMask = np.ones((image_gray.shape[0],image_gray.shape[1]),np.uint8)
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
