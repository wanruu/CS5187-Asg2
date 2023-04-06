import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def detect_sift(img):
    sift = cv2.SIFT_create()
    kp = sift.detect(img, None)  # tuple<cv2.KeyPoint>
    kp, des = sift.compute(img, kp)
    return kp, des  # (#kp, 128)


# block matching algorithm
# psnr = 4.5619
def cal_disp_bm(img_l, img_r, numDisparities=256, blockSize=15):
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparity = stereo.compute(gray_l, gray_r)
    return disparity


# psnr = -0.5750
def cal_disp_basic(img_l, img_r):  # img: (1110, 1390, 3)
    # sift feature
    kp_l, des_l = detect_sift(img_l)
    kp_r, des_r = detect_sift(img_r)

    # feature matching
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(des_l, des_r)

    # remove matches with much diff in y-axis
    matches = tuple(filter(lambda m: abs(kp_l[m.queryIdx].pt[1] - kp_r[m.trainIdx].pt[1]) < 1, matches))
    
    # extract match points
    match_pts_l = np.asarray([kp_l[match.queryIdx].pt for match in matches])  # (1766, 2)
    match_pts_r = np.asarray([kp_r[match.trainIdx].pt for match in matches])

    # test: draw all matches
    # match_result = cv2.drawMatches(img_l, kp_l, img_r, kp_r, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(match_result)
    # plt.show()

    # fundamental mat
    f_mat, _ = cv2.findFundamentalMat(match_pts_l, match_pts_r, cv2.FM_8POINT)  # (3, 3) (1766, 1)

    # test: draw a epiline
    # pt_l, pt_r = tuple(match_pts_l[100]), tuple(match_pts_r[100])
    # cv2.circle(img_l, (int(pt_l[0]), int(pt_l[1])), 5, (0,255,0), -1)
    # cv2.imshow('img', img_l)
    # cv2.waitKey(0)
    # a, b, c = tuple(cv2.computeCorrespondEpilines(np.asarray([pt_l]), 1, f_mat)[0][0])
    # img_h, img_w = img_l.shape[0:2]
    # pt1, pt2 = (0, int(-c/b)), ((img_w-1), int((-c-a*(img_w-1))/b))
    # cv2.line(img_r, pt1, pt2, (0,0,0), 1)
    # cv2.circle(img_r, (int(pt_r[0]), int(pt_r[1])), 5, (0,255,0), -1)
    # cv2.imshow('img', img_r)
    # cv2.waitKey(0)


    # cal disp with epilines
    img_h, img_w = img_l.shape[0:2]
    xs, ys = np.arange(img_w), np.arange(img_h)
    disp = np.zeros((img_h, img_w))
    
    # points in left image (N,2)
    pts_l = np.zeros((img_h, img_w, 2))
    for y_l in ys:
        for x_l in xs:
            pts_l[y_l][x_l] = np.asarray([x_l, y_l])
    pts_l = pts_l.reshape((-1, 2))
    
    # epilines (N,3)
    lines = cv2.computeCorrespondEpilines(pts_l, 1, f_mat).reshape((-1, 3))

    # TODO

    return disp


if __name__ == "__main__":
    import os, config

    # test case
    name = "art"
    path = config.test_paths[name]

    # read image
    path_l = os.path.join(path, "view1.png")
    path_r = os.path.join(path, "view5.png")
    path_disp = os.path.join(path, "disp1.png")
    img_l = cv2.imread(path_l)
    img_r = cv2.imread(path_r)
    img_disp = cv2.imread(path_disp, cv2.IMREAD_GRAYSCALE)

    # basic
    disp = cal_disp_bm(img_l, img_r)
    
    # normalize
    max_val, min_val = disp.max(), disp.min()
    norm_disp = (disp - min_val) / (max_val - min_val) * 255

    # cal psnr
    psnr = utils.cal_psnr(img_disp, norm_disp)
    print("norm", psnr)
    psnr = utils.cal_psnr(img_disp, disp)
    print(psnr)

    # save
    path = os.path.join(config.result_path, f"norm-{name}.png")
    cv2.imwrite(path, norm_disp)
    path = os.path.join(config.result_path, f"{name}.png")
    cv2.imwrite(path, disp)
