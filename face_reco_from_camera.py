import cv2
import os
import dlib
import logging
import numpy as np
import pandas as pd
import time
import imutils
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import distance as dist
from imutils import face_utils


# detector = dlib.cnn_face_detection_model_v1('./data/data_dlib/mmod_human_face_detector.dat')
detector = dlib.get_frontal_face_detector()
# 检测人脸的位置
shape_predictor = dlib.shape_predictor('./data/data_dlib/shape_predictor_68_face_landmarks.dat')
# 得到人脸的关键点
face_reco_model = dlib.face_recognition_model_v1('./data/data_dlib/dlib_face_recognition_resnet_model_v1.dat')
# 提取特征

total = 0
count_eye = 0
SWITCH_BLINK_DETECTION = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def blink_detect(img):
    EAR_THRESH = 0.21
    EYE_close = 2

    global total
    global count_eye
    # count_eye = 0

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # grab the frame from the threaded video file stream, resize it, and convert it to grayscale channels
    frame = img
    # frame = imutils.resize(frame, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region,
        # then convert the facial landmark (x, y)-coordinates to a Numpy array
        shape = shape_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates,
        # then use the coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EAR_THRESH:
            count_eye += 1
        else:
            if count_eye >= EYE_close:
                total += 1
            count_eye = 0

        # draw the computed eye aspect ratio on the frame to help with debugging and
        # set the correct eye aspect ratio thresholds and frame counters
        cv2.putText(frame, "When you use blink detection, only one person is allowed on camera!", (200, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(frame, "Please blink your eyes more than 5 times", (300, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Blinks: {}".format(total), (760, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, 'EAR: {:.2f}'.format(ear), (900, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

class Face_Recognizer:
    def __init__(self):
        self.face_feature_known_list = []                   # Save the features of faces in database
        self.face_name_known_list = []                      # Save the name of faces in database

        self.current_frame_face_cnt = 0
        self.current_frame_face_feature_list = []
        self.current_frame_face_name_list = []
        self.current_frame_face_name_position_list = []

        # FPS
        self.frame_cnt = 0
        self.fps = 0
        self.fps_show = 0
        self.frame_start_time = 0
        self.start_time = time.time()

        self.font = cv2.FONT_ITALIC
        self.font_chinese = ImageFont.truetype("STHeiti Light.ttc", 30, encoding="unic")

    def get_face_database(self):
        if os.path.exists("./data/features_all.csv"):
            path_features_known_csv = "./data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_feature_known_list.append(features_someone_arr)
            logging.info("Faces in Database: %d", len(self.face_feature_known_list))
            return True
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return False

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        distance = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return distance

    def update_fps(self):
        now = time.time()
        # refresh fps per second
        if str(self.start_time).split('.')[0] != str(now).split('.')[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_note(self, img_rd):
        # add some notes
        cv2.putText(img_rd, "Face Register", (20, 35), self.font, 1, (50, 132, 235), 2, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:   " + str(self.frame_cnt), (20, 150), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:   " + str(self.fps_show.__round__(2)), (20, 190), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 230), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (55, 50, 55), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "B: Blink detection", (20, 500), self.font, 0.8, (55, 50, 55), 1, cv2.LINE_AA)

    def draw_name(self, img_rd):
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.current_frame_face_cnt):
            draw.text(xy=self.current_frame_face_name_position_list[i],
                      text=self.current_frame_face_name_list[i],
                      font=self.font_chinese,
                      fill=(255, 255, 0))
            img_rd = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            # print(self.current_frame_face_name_position_list[i])
        return img_rd

    def brink_result(self, img_rd):
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.current_frame_face_cnt):
            draw.text(xy=self.current_frame_face_name_position_list[i],
                      text=self.current_frame_face_name_list[i] + '    real person',
                      font=self.font_chinese,
                      fill=(255, 255, 0))
            img_rd = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            # print(self.current_frame_face_name_position_list[i])
        return img_rd

    def transform_chinese_name(self):
        if self.current_frame_face_cnt >= 1:
            self.face_name_known_list[0] = '曹伟'.encode('utf-8').decode()
            self.face_name_known_list[1] = '刘诗诗'.encode('utf-8').decode()
            # self.face_name_known_list[2] = '韩闯'.encode('utf-8').decode()
            # self.face_name_known_list[0] = '秦老师'.encode('utf-8').decode()
            # self.face_name_known_list[1] = '黄老师'.encode('utf-8').decode()
            # self.face_name_known_list[2] = '曹老师'.encode('utf-8').decode()


    def compare_faces(self, stream):
        # 1.Reading known faces from "features_all.csv"
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame %d starts", self.frame_cnt)
                flag, img_rd = stream.read()
                img_rd = imutils.resize(img_rd, width=1000)
                faces = detector(img_rd, 0)
                kk = cv2.waitKey(1)
                if kk == ord('q'):
                    break
                else:
                    self.draw_note(img_rd)
                    self.current_frame_face_feature_list = []
                    self.current_frame_face_cnt = 0
                    self.current_frame_face_name_position_list = []
                    self.current_frame_face_name_list = []

                    # 2.Face detected in current frame
                    if len(faces) != 0:
                        # 3.Compute the face descriptors for faces in current frame / 获取当前捕获的图片的所有人脸特征
                        for i in range(len(faces)):
                            shape = shape_predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                        # 4.Traversal all the faces in the database
                        for k in range(len(faces)):
                            logging.debug("For face %d in camera:", k+1)
                            self.current_frame_face_name_list.append("unknown")
                            # Position of faces captured
                            self.current_frame_face_name_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 5.For every faces detected, compare the faces in the database
                            current_frame_e_distance_list = []
                            for i in range(len(self.face_feature_known_list)):
                                if str(self.face_feature_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(self.current_frame_face_feature_list[k],
                                                                                    self.face_feature_known_list[i])
                                    logging.debug("     With person %s , the e-distance is %f", str(i+1), e_distance_tmp)
                                    current_frame_e_distance_list.append(e_distance_tmp)
                                else:
                                    current_frame_e_distance_list.append(999999999)

                            # 6.Find the one with the minimum e-distance
                            similar_peron_num = current_frame_e_distance_list.index(min(current_frame_e_distance_list))
                            logging.debug("Minimum e-distance with %s: %f", self.face_name_known_list[similar_peron_num],
                                          min(current_frame_e_distance_list))

                            if min(current_frame_e_distance_list) < 0.4:
                                self.current_frame_face_name_list[k] = self.face_name_known_list[similar_peron_num]
                                logging.debug("Face recognition result: %s", self.face_name_known_list[similar_peron_num])
                            else:
                                logging.debug("Face recognition result: Unknown person")
                            logging.debug("\n")
                            # Draw rectangle
                            for kk, d in enumerate(faces):
                                cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                              (255, 255, 255), 2)

                        self.current_frame_face_cnt = len(faces)
                        # 7.Modify name if needed
                        self.transform_chinese_name()
                        # 8.Draw name
                        img_with_name = self.draw_name(img_rd)
                        # 9.If press key 'b', start blink detecting
                        dd = cv2.waitKey(1)
                        global SWITCH_BLINK_DETECTION
                        if dd == ord('b'):
                            SWITCH_BLINK_DETECTION = True
                        global total
                        if SWITCH_BLINK_DETECTION:
                            if self.current_frame_face_cnt == 1:
                                if total > 5:
                                    img_with_name = self.brink_result(img_rd)
                                blink_detect(img_with_name)
                            else:
                                total = 0
                    else:
                        img_with_name = img_rd

                logging.debug("Face in camera now: %s", self.current_frame_face_name_list)

                cv2.imshow("camera", img_with_name)
                # 10.Update stream FPS
                self.update_fps()
                logging.debug("Frame ends\n\n")

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 480)
        self.compare_faces(cap)

        cap.release()
        cv2.destroyAllWindows()

def main():
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()

if __name__ == '__main__':
    main()
