import cv2
import dlib
import numpy as np
import os
import shutil
import time
import logging
import imutils

detector = dlib.get_frontal_face_detector()

class Face_Register:
    def __init__(self):
        self.path_photos_from_camera = "./data/data_faces_from_camera/"
        self.font = cv2.FONT_ITALIC

        self.existing_faces_cnt = 0
        self.ss_cnt = 0
        self.current_frame_faces_cnt = 0

        self.save_flag = 1
        self.press_n_flag = 0

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

    # mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # create folders to save face images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    # delete old face folders
    def pre_work_del_old_face_folders(self):
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera + folders_rd[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")

    # 如果有之前录入的人脸，在之前person_x的序号按照person_x+1开始录入
    def check_existing_face_cnt(self):
        if os.listdir("data/data_faces_from_camera/"):
            # get the order of latest person
            person_list = os.listdir("data/data_faces_from_camera/")
            person_num_list = []
            for person in person_list:
                person_num_list.append(int(person.split('_')[-1]))
            self.existing_faces_cnt = max(person_num_list)
        # start from person_1
        else:
            self.existing_faces_cnt = 0

    # update_FPS of video stream
    def update_fps(self):
        now = time.time()
        # refresh fps per second
        if str(self.start_time).split('.')[0] != str(now).split('.')[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    # putText on cv2 window
    def draw_note(self, img_rd):
        # add some notes
        cv2.putText(img_rd, "Face Register", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:   " + str(self.fps_show.__round__(2)), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_faces_cnt), (20, 140), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "N: Create face folder", (20, 300), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "S: Save current face", (20, 350), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 400), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # Main process of face detection and saving
    def process(self, stream):
        # 1.Create folders to save photos
        self.pre_work_mkdir()
        # 2.删除"data/data_faces_from_camera"中已有的人脸图像文件
        # /Uncomment if want to delete the saved faces and start from person_1
        # if os.path.isdir(self.path_photos_from_camera):
        #     self.pre_work_del_old_face_folders()

        # 3.检查"data/data_faces_from_camera"中已有人脸文件
        self.check_existing_face_cnt()

        while stream.isOpened():
            flag, img_rd = stream.read()
            img_rd = imutils.resize(img_rd, width=850)
            kk = cv2.waitKey(1)
            faces = detector(img_rd, 0)

            # 4.按下'n'新建存储人脸的文件夹
            if kk == ord('n'):
                self.existing_faces_cnt += 1
                current_face_dir = self.path_photos_from_camera + "person_" + str(self.existing_faces_cnt)
                os.makedirs(current_face_dir)
                logging.info("\n%-40s %s", "Create folders:", current_face_dir)

                self.ss_cnt = 0             # clear the cnt of screen shots
                self.press_n_flag = 1       # pressed 'n' already

            # 5.face detected
            if len(faces) != 0:
                # show the ROI of faces
                for k, d in enumerate(faces):
                    # compute the size of rectangle box
                    height = (d.bottom() - d.top())
                    width = (d.right() - d.left())
                    hh = int(height / 2)
                    ww = int(width / 2)

                    # 6.if the size of ROI > 480*640
                    if (d.right() + ww) > 640 or (d.bottom() + hh > 480) or (d.left() - ww < 0) or (d.top() - hh < 0):
                        cv2.putText(img_rd, 'OUT OF RANGE', (20, 240), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                        color_rectangle = (0, 0, 255)
                        save_flag = 0
                        if kk == ord('s'):
                            logging.warning("Please adjust your position")
                    else:
                        color_rectangle = (255, 255, 255)
                        save_flag = 1

                    cv2.rectangle(img_rd,
                                  tuple([d.left() - ww, d.top() - hh]),
                                  tuple([d.right() + ww, d.bottom() + hh]),
                                  color_rectangle, 2)

                    # 7.create blank image according to the size of face detected
                    img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

                    if save_flag:
                        # 8.press 's' to save faces into local images
                        if kk == ord('s'):
                            # check if you have pressed 'n'
                            if self.press_n_flag:
                                self.ss_cnt += 1
                                for ii in range(height*2):
                                    for jj in range(width*2):
                                        img_blank[ii][jj] = img_rd[d.top() - hh + ii][d.left() - ww + jj]
                                cv2.imwrite(current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", img_blank)
                                logging.info("%-40s %s/img_face_%s.jpg", "Save into: ", str(current_face_dir), str(self.ss_cnt))
                            else:
                                logging.warning("Please press 'N' and press 'S'")

            self.current_frame_faces_cnt = len(faces)

            # 9.add note on cv2 window
            self.draw_note(img_rd)

            # 10.press 'q' to exit
            if kk == ord('q'):
                break

            # 11.update FPS
            self.update_fps()

            cv2.namedWindow('camera', 1)
            cv2.imshow('camera', img_rd)

    def run(self):
        # cap = cv2.VideoCapture("video.mp4")   # get video stream from video file
        cap = cv2.VideoCapture(0)               # get video stream from camera
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()

def main():
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register()
    Face_Register_con.run()

if __name__ == '__main__':
    main()
