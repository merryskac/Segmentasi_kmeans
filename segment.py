import numpy as np
import cv2
import matplotlib.pyplot as plt
#read image
image = ['gbr1.jpg','gbr2.jpg','gbr3.jpg','gbr4.jpg','gbr5.jpg']
for i in image:
    img_read = cv2.imread(i)
    img_read = cv2.resize(img_read, (500, 500))
    img = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
    # preproccessing
    flatten = img.reshape((-1,3))
    flatten = np.float32(flatten)

    #parameter
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 4
    attempts = 10

    #k_means
    ret, label, center = cv2.kmeans(flatten, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    flat=center[label.flatten()]
    print(label.shape)

    A = flatten[label.ravel() == 0]
    B = flatten[label.ravel() == 1]
    C = flatten[label.ravel() == 2]
    D = flatten[label.ravel() == 3]

    #change hex to rgb
    def rgb_to_hex(rgb):
        return '%02x%02x%02x' % rgb

    c1 = tuple(center[0])
    c2 = tuple(center[1])
    c3 = tuple(center[2])
    c4 = tuple(center[3])
    col1 = (rgb_to_hex((np.uint8(c1[0]),np.uint8(c1[1]),np.uint8(c1[2]))))
    col2 = (rgb_to_hex((np.uint8(c2[0]),np.uint8(c2[1]),np.uint8(c2[2]))))
    col3 = (rgb_to_hex((np.uint8(c3[0]), np.uint8(c3[1]), np.uint8(c3[2]))))
    col4 = (rgb_to_hex((np.uint8(c4[0]), np.uint8(c4[1]), np.uint8(c4[2]))))

    # plotting
    plt.scatter(A[:, 0], A[:, 1], c='#'+col1, label='warna1', s=0.1 )
    plt.scatter(B[:, 0], B[:, 1], c='#'+col2, label='warna2', s=0.1)
    plt.scatter(C[:, 0], C[:, 1], c='#'+col3,label='warna3', s=0.1)
    plt.scatter(D[:, 0], D[:, 1], c='#'+col4,label='warna4', s=0.1)
    plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
    plt.xlabel('Red'), plt.ylabel('Green')
    plt.legend()
    plt.show()

    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    result_img = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    #clipping
    hsv_img = cv2.cvtColor(result_image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_img,(90,0,130),(128,255,255))
    result = cv2.bitwise_and(img,img,mask=mask)
    rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    cv2.imshow('ori_pic', img_read)
    cv2.imshow('color segment', result_img)
    cv2.imshow('color mask segment', rgb)
    cv2.waitKey()



