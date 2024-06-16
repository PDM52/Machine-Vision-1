import cv2
import numpy as np


def count_coins(file):
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(cv2.imread(file, 0), 1)
    contours = get_contours(gray)
    coins = {'big_inside': 0, 'small_inside': 0, 'big_outside': 0, 'small_outside': 0}

    big_circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 50, param1=70, param2=25, minRadius=35, maxRadius=40)
    if big_circles is not None:
        big_circles = np.uint16(np.around(big_circles))
        for i in big_circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
            cv2.circle(img, (i[0], i[1]), 2, (0, 255, 0), 3)
            if check_point_position(contours, (i[0], i[1])):
                coins['big_inside'] += 1
            else:
                coins['big_outside'] += 1

    small_circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 50, param1=80, param2=20, minRadius=20, maxRadius=29)
    if small_circles is not None:
        small_circles = np.uint16(np.around(small_circles))
        for i in small_circles[0, :]:
            if is_outside_radius(i, big_circles, 20):
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
                if check_point_position(contours, (i[0], i[1])):
                    coins['small_inside'] += 1
                else:
                    coins['small_outside'] += 1

    cv2.imshow('COINS', img)
    print(coins)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_contours(img):
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    ret, thresh = cv2.threshold(blurred, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imax = 0
    areamax = 0
    for i in range(len(contours)):
        temp = contours[i]
        area = cv2.contourArea(temp)
        if area > areamax:
            imax = i
            areamax = area
    tray = contours[imax]
    return tray


def check_point_position(contour, point):
    result = cv2.pointPolygonTest(contour, point, False)
    if result > 0:
        return True
    elif result < 0:
        return False


def is_outside_radius(point, points, radius):
    if points is not None:
        point = np.array(point[:2])
        points = np.array(points)[0][:, :2]
        distances = np.sqrt(np.sum((points - point) ** 2, axis=1))
        return np.all(distances >= radius)
    return True

if __name__ == '__main__':
    for i in range(1,8):
        count_coins(f'tray{i}.jpg')

