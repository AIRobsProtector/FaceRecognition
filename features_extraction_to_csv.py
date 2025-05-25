# Extract features from images and save into "features_all.csv"
import csv
import cv2
import os
import dlib
import logging
import numpy as np

path_images_from_camera = "./data/data_faces_from_camera/"

# detector = dlib.cnn_face_detection_model_v1('./data/data_dlib/mmod_human_face_detector.dat')
detector = dlib.get_frontal_face_detector()
# 检测人脸位置
shape_predictor = dlib.shape_predictor('./data/data_dlib/shape_predictor_68_face_landmarks.dat')
# 得到人脸的关键点
face_reco_model = dlib.face_recognition_model_v1('./data/data_dlib/dlib_face_recognition_resnet_model_v1.dat')
# 提取特征

# faces from detector: <class '_dlib_pybind11.rectangles'> rectangles[[(98, 98) (253, 253)]]
# rectangles是dlib的一个类，一个矩形，表示为能够唯一表示这个人脸矩形框两个点坐标：左上角（x1,y1）、右下角(x2,y2)
# shape from shape_predictor: <class '_dlib_pybind11.full_object_detection'>
# <_dlib_pybind11.full_object_detection object at 0x131161570>
# full_object_detection是dlib中的一个类，用于表示关键点（眼睛、鼻子、嘴巴等）的完整检测结果
# face_descriptor from dlib_face_recognition_resnet_model_v1 <class '_dlib_pybind11.vector'>
# vector是dlib中的一种数据类型，表示一个128纬的向量

# Return 128D features for single image
# input:    path_img            <class 'str'>
# output:   face_descriptor     <class 'dlib.vector'>
def face_encoding(path_img):
    img_rd = cv2.imread(path_img)
    faces = detector(img_rd, 1)
    print('faces from detector:', type(faces), faces)

    logging.info("%-40s %-20s", "Image with faces detected:", path_img)

    # For photos of faces saved, we need fo make sure that we can detect faces from the cropped images
    if len(faces) != 0:
        shape = shape_predictor(img_rd, faces[0])
        print('shape from shape_predictor:', type(shape), shape)
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
        print('face_descriptor from dlib_face_recognition_resnet_model_v1', type(face_descriptor), face_descriptor)
    else:
        face_descriptor = 0
        logging.warning("no face")
    return face_descriptor

# Return the mean value of 128D face descriptor for personX
# input:    path_face_personX           <class 'str'>
# output:   features_mean_personX       <class 'numpy.ndarray'>
def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)
    if photos_list:
        for i in range(len(photos_list)):
            logging.info("%-40s %-20s", "Reading image:", path_face_personX + '/' + photos_list[i])
            features_128d = face_encoding(path_face_personX + '/' + photos_list[i])
            if features_128d == 0:
                i += 1
            else:
                features_list_personX.append(features_128d)
    else:
        logging.warning("Warning: No images in%s/", path_face_personX)

    # compute the mean / 计算128D特征的均值
    # personX的N张图像 -> 1 x 128D
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=object, order='C')
    return features_mean_personX

def main():
    logging.basicConfig(level=logging.INFO)
    # Get the order of the lasted person
    person_list = os.listdir("./data/data_faces_from_camera/")
    person_list.sort()

    with open("./data/features_all.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for person in person_list:
            # Get the average features of face/personX, it will be a list with a length of 128D
            logging.info("%sperson_%s", path_images_from_camera, person)
            features_mean_personX = return_features_mean_personX(path_images_from_camera + person)

            if len(person.split('_', 2)) == 2:
                person_name = person
            else:
                person_name = person.split('_', 2)[-1]
            features_mean_personX = np.insert(features_mean_personX, 0, person_name, axis=0)
            # features_mean_personX will be 129D, person name + 128 features
            writer.writerow(features_mean_personX)
            logging.info('\n')
        logging.info("Save all the features of faces registered into: ./data/features_all.csv")

if __name__ == '__main__':
    main()
