from ex1_utils import *
import matplotlib.pyplot as plt
from gamma import gammaDisplay


def histEqDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    imgeq, histOrg, histEq = hsitogramEqualize(img)

    f_name = 'histEQ_{}.png'.format('RGB' if rep == LOAD_RGB else 'GRAY')
    s_img = imgeq * 255 if rep == LOAD_GRAY_SCALE else cv2.cvtColor((255 * imgeq).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f_name, s_img)

    # Display cumsum
    cumsum = np.cumsum(histOrg)
    cumsumEq = np.cumsum(histEq)
    plt.gray()
    plt.plot(range(256), cumsum, 'r.')
    plt.plot(range(256), cumsumEq, 'g')

    # Display the images
    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(imgeq)
    plt.show()


def quantDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    st = time.time()

    img_lst, err_lst = quantizeImage(img, 4, 5, False)
    # f_name = 'quant_{}.png'.format('RGB' if rep == LOAD_RGB else 'GRAY')
    # s_img = img_lst[-1] if rep == LOAD_GRAY_SCALE else cv2.cvtColor((255*img_lst[-1]).astype(np.uint8), cv2.COLOR_RGB2BGR)
    # cv2.imwrite(f_name, s_img)

    # img_lst, err_lst = quantizeImageKmeans(img, nQuant=4, nIter=5)
    # f_name = 'quant_kmeans{}.png'.format('RGB' if rep == LOAD_RGB else 'GRAY')
    # s_img = img_lst[-1] if rep == LOAD_GRAY_SCALE else cv2.cvtColor((255 * img_lst_kmeans[-1]).astype(np.uint8),
    #                                                                 cv2.COLOR_RGB2BGR)
    # cv2.imwrite(f_name, s_img)

    print("Time:%.2f" % (time.time() - st))
    print("Error 0:\t %f" % err_lst[0])
    print("Error last:\t %f" % err_lst[-1])

    plt.gray()
    plt.imshow(img_lst[0])
    plt.figure()
    plt.imshow(img_lst[-1])

    plt.figure()
    plt.plot(err_lst, 'r')
    plt.show()


def main():
    # print("ID:", myID())
    img_path = 'beach.jpg'

    # # Basic read and display
    # imDisplay(img_path, LOAD_GRAY_SCALE)
    # imDisplay(img_path, LOAD_RGB)
    #
    # # Convert Color spaces
    # img = imReadAndConvert(img_path, LOAD_RGB)
    # yiq_img = transformRGB2YIQ(img)
    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(transformYIQ2RGB(yiq_img))
    # ax[1].imshow(img)
    # plt.show()
    #
    # # Image histEq
    # histEqDemo(img_path, LOAD_GRAY_SCALE)
    # histEqDemo(img_path, LOAD_RGB)

    # # Image Quantization
    quantDemo(img_path, LOAD_GRAY_SCALE)
    # quantDemo(img_path, LOAD_RGB)

    #
    # # Gamma
    # gammaDisplay(img_path, LOAD_GRAY_SCALE)
    # gammaDisplay(img_path, LOAD_RGB)


if __name__ == '__main__':
    main()
