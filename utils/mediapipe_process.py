
import os
import numpy as np
import torch
import cv2
import sys
import matplotlib.pyplot as plt

from check_image_with_plt import *
from iou import bb_intersection_over_union
root = os.path.join('..', 'MediaPipePyTorch')
sys.path.insert(0, root)

from blazebase import resize_pad, denormalize_detections
from blazepalm import BlazePalm
from blazehand_landmark import BlazeHandLandmark
from blazebase import BlazeDetector

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS

def implot(img):
    return plt.imshow(img[:,:,(2,1,0)])

def get_result(overlap_obj, iou_thres=0.5):
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    # back_detector = False
    # face_detector = BlazeFace(back_model=back_detector).to(gpu)
    # if back_detector:
    #     face_detector.load_weights("blazefaceback.pth")
    #     face_detector.load_anchors("anchors_face_back.npy")
    # else:
    #     face_detector.load_weights("blazeface.pth")
    #     face_detector.load_anchors("anchors_face.npy")

    palm_detector = BlazePalm().to(gpu)
    palm_detector.load_weights(os.path.join(root, "blazepalm.pth"))
    palm_detector.load_anchors(os.path.join(root, "anchors_palm.npy"))
    palm_detector.min_score_thresh = .75

    hand_regressor = BlazeHandLandmark().to(gpu)
    hand_regressor.load_weights(os.path.join(root, "blazehand_landmark.pth"))

    # face_regressor = BlazeFaceLandmark().to(gpu)
    # face_regressor.load_weights("blazeface_landmark.pth")


    WINDOW='test'
    cv2.namedWindow(WINDOW)
    overlap = overlap_obj
    assert overlap is not None
    # path_  ='./mycode/imgs/'
    # im_name = 'a_%s.png'%str(num).zfill(5)
    # frame = cv2.imread(os.path.join(root, path_, im_name))
    plot = 0
    assert plot == True or plot == False
    cnt = 0
    correct =0
    # for index in range(len(overlap.data)):
    for index in [11]:
        cnt += 1
        img_data = overlap.data[index]
        frame = cv2.imread(img_data.image_path)
        # print('img_data.image_path',img_data.image_path)
        mirror_img = False
        if mirror_img:
            frame = np.ascontiguousarray(frame[:,::-1,::-1])
        else:
            frame = np.ascontiguousarray(frame[:,:,::-1])

        img1, img2, scale, pad = resize_pad(frame)

        # if back_detector:
        #     normalized_face_detections = face_detector.predict_on_image(img1)
        # else:
        #     normalized_face_detections = face_detector.predict_on_image(img2)
        normalized_palm_detections, (raw_pred, nms_pred) = palm_detector.predict_on_image(img1)
        # print('raw pred=', type(raw_pred)) # list
        # print('nms pred=', type(nms_pred)) # list
        # print(raw_pred[0].shape) # torch.tensor 0,19
        # print(raw_pred[0])
        # print(nms_pred[0].shape)

        # face_detections = denormalize_detections(normalized_face_detections, scale, pad)
        palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)


        # xc, yc, scale, theta = face_detector.detection2roi(face_detections.cpu())
        # img, affine, box = face_regressor.extract_roi(frame, xc, yc, theta, scale)
        # flags, normalized_landmarks = face_regressor(img.to(gpu))
        # landmarks = face_regressor.denormalize_landmarks(normalized_landmarks.cpu(), affine)


        xc, yc, scale, theta = palm_detector.detection2roi(palm_detections.cpu())
        img, affine2, box2 = hand_regressor.extract_roi(frame, xc, yc, theta, scale)
        flags2, handed2, normalized_landmarks2 = hand_regressor(img.to(gpu))
        landmarks2 = hand_regressor.denormalize_landmarks(normalized_landmarks2.cpu(), affine2)
        

        # for i in range(len(flags)):
        #     landmark, flag = landmarks[i], flags[i]
        #     if flag>.5:
        #         draw_landmarks(frame, landmark[:,:2], FACE_CONNECTIONS, size=1)


        for i in range(len(flags2)):
            landmark, flag = landmarks2[i], flags2[i]
            if flag>.5:
                draw_landmarks(frame, landmark[:,:2], HAND_CONNECTIONS, size=2)

        # draw_roi(frame, box)
        # draw_roi(frame, box2)
        # draw_detections(frame, face_detections)
        draw_detections(frame, palm_detections)
        # print(palm_detections.shape)


        ########## plot
        if plot:
            img = frame[:,:,::-1]
            implot(img)

        all_iou = []
        for i in range(palm_detections.shape[0]):
            _ = []
            for j in range(len(img_data.hand_bboxs)):
                ymin = palm_detections[i, 0]
                xmin = palm_detections[i, 1]
                ymax = palm_detections[i, 2]
                xmax = palm_detections[i, 3]
                # plot
                if plot:
                    boxA = overlap.plot_keypoint_on_image(index=index, img = frame, plotxy=(xmin,xmax,ymin,ymax), only_plot=True)
                boxA = img_data.hand_bboxs[j].get_bbox()
                boxB = xmin, ymin, xmax, ymax
                iou = bb_intersection_over_union(boxA, boxB)
                _.append(float(iou))
            all_iou.append(_)
        # print('all iou=',all_iou)

        # select max iou
        try:
            ans_iou = []
            for i in range(len(all_iou[0])):
                a = all_iou[0][i]
                b = all_iou[1][i]
                _ = a if a>b else b
                ans_iou.append(_)
            print('[%s] iou'%index, ans_iou)
            def is_correct():
                for iou in ans_iou:
                    if iou < iou_thres:
                        return False
                return True
            ans = is_correct()
            if ans:
                correct += 1
        except:
            print('fail at index=',index)
            print('all iou=',all_iou)


        if plot:
            plt.title('%s [%d]'%(overlap.name, index), )
            plt.show()
    print(overlap.name, '%d/%d (%.2f)'%(correct, cnt, correct/cnt*100))
    return '%s %d/%d (%.2f)'%(overlap.name, correct, cnt, correct/cnt*100)

        # print(np.shape(img))
        # 1/0
        # implot(img)
        # plt.show()
        # cv2.imshow(WINDOW, frame[:,:,::-1])
        # cv2.imwrite('sample/%04d.jpg'%frame_ct, frame[:,:,::-1])
        # cv2.imwrite(str(os.path.join('./im_out/', im_name)), frame[:,:,::-1])
        # hasFrame, frame = capture.read()
        # hasFrame, frame = True, 
def main():
    # aa = get_result(read_single_hand())
    a = get_result(read_overlap0())
    # b = get_result(read_overlap5())
    # c = get_result(read_overlap20())
    # d = get_result(read_overlap50())
    # e = get_result(read_overlap80())
    # print(aa)
    # print(b)
    # print(c)
    # print(d)

if __name__ == '__main__':
    main()
    # a = BlazeDetector().min_score_thresh
    # b = BlazeDetector().min_suppression_thresh
    # print(a, b)