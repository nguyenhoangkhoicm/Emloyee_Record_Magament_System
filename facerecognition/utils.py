from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import cv2
import matplotlib.pyplot as plt
import time
import os
from . import FacialRecognition 
from django.contrib import messages
import subprocess
import tensorflow as tf
import numpy as np
from . import facenet
import os
import threading
import math
import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report

currentPythonFilePath = os.getcwd()
        
#sử dụng .replace('\\','/') để thay đổi dấu / đg dẫn.

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Load face detector and recognizer
detector = FacialRecognition.FaceDetector(
    minsize=20,
    threshold=[0.6, 0.7, 0.7],
    factor=0.709,
    gpu_memory_fraction=0.6,
    detect_face_model_path=currentPythonFilePath+'/static/align'.replace('\\','/'),
    facenet_model_path=currentPythonFilePath+'/static/Models/20180402-114759.pb'.replace('\\','/')
)
recognizer = FacialRecognition.FaceRecognition(classifier_path= currentPythonFilePath+'/static/Models/facemodel.pkl'.replace('\\','/'))
# Initialize barcode reader
barcode_reader = FacialRecognition.BarcodeReader(verbose=False)

def uploadFile(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)
    return context


def imshowx(img, title='B2DL'):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 4
    plt.rcParams["figure.figsize"] = fig_size

    plt.axis('off')
    plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def imshowgrayx(img, title='BD2L'):
    plt.axis('off')
    plt.title(title)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()

#hàm mở camera và nhận diện khuôn mặt
def face_recognition(request):
    # Create named window
    cv2.namedWindow('Nhan Dien Khuon Mat')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return HttpResponse("Khong the mo camera")
    while True:
        try:
            ret, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Read barcodes from the image
            barcodes, imgbarcode = barcode_reader.read_barcodes(rgb)      
            # Detect faces in the frame
            faces, _= detector.get_faces(rgb)
            for face in faces:
                x1, y1, x2, y2 = face[:4]

                # Get face image
                face_img = rgb[int(y1):int(y2), int(x1):int(x2), :]
                #lưu ảnh vào thư mục static
                

                # Get face embeddings
                embeddings = detector.get_embeddings(face_img)

                # Recognize face
                name, prob = recognizer.recognize_face(embeddings)
                cv2.imwrite(currentPythonFilePath+'/static/'+name+'.jpg'.replace('\\','/'),face_img)
                cv2.rectangle(imgbarcode, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(imgbarcode, "{} {:.2f}".format(name, prob), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Nhan Dien Khuon Mat', imgbarcode)
        except Exception as e:
            continue

        # Check if window is closed
        if cv2.getWindowProperty('Nhan Dien Khuon Mat', cv2.WND_PROP_VISIBLE) < 1:

            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return HttpResponse(name,barcodes)

def face_detection(request):
    
    folder_name = request.GET.get('name')
    folder_path = os.path.join('./static/data/', folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        # messages.success(request, 'Tạo thư mục thành công')
    cv2.namedWindow('Phat Hien Khuon Mat')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return HttpResponse("Khong the mo camera")
    time.sleep(2)
    count = 0
    while True:
        if count >= 10:
            messages.success(request, 'Có dữ liệu mới được thêm vào.')
            break
        try:
            ret, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect faces in the frame
            faces, _ = detector.get_faces(rgb)

            for face in faces:
                faces_found= faces.shape[0]
                if faces_found > 1:
                        cv2.putText(rgb, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                elif faces_found > 0:
                    x1, y1, x2, y2 = face[:4]
                    count += 1
                    # Get face image
                    face_img = rgb[int(y1):int(y2), int(x1):int(x2), :]

                    folder_path = os.path.join(currentPythonFilePath, 'static', 'data', folder_name)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    image_path = os.path.join(folder_path, folder_name + str(count) + '.jpg')
                    cv2.imwrite(image_path, face_img)
                    #draw bbox
                    cv2.rectangle(rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.imshow('Phat Hien Khuon Mat', rgb)
        except Exception as e:
            continue

        # Check if window is closed
        if cv2.getWindowProperty('Phat Hien Khuon Mat', cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return HttpResponse('ok')

def train_and_save_classifier(data_dir, model, classifier_filename, use_split_dataset=True,
                               min_nrof_images_per_class=20, nrof_train_images_per_class=10,
                               batch_size=90, image_size=160, seed=123, mode='TRAIN'):
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            np.random.seed(seed=seed)
            if use_split_dataset:
                dataset_tmp = facenet.get_dataset(data_dir)
                train_set, test_set = facenet.split_dataset(dataset_tmp, min_nrof_images_per_class,
                                                            nrof_train_images_per_class)
                if mode == 'TRAIN':
                    dataset = train_set
                elif mode == 'CLASSIFY':
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(data_dir)

            for cls in dataset:
                assert (len(cls.image_paths) > 0,
                        'There must be at least one image for each class in the dataset')

            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            print('Loading feature extraction model')
            facenet.load_model(model)

            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(classifier_filename)

            if mode == 'TRAIN':
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)
                class_names = [cls.name.replace('_', ' ') for cls in dataset]
                train_predictions = model.predict(emb_array)
                train_accuracy = np.mean(np.equal(train_predictions, labels))

                train_report = classification_report(labels, train_predictions, target_names=class_names)

                print('Training classification report:\n', train_report)

                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)
                print('Training accuracy: %.3f' % train_accuracy)

from datetime import datetime
def train(request):
    currentPythonFilePath = os.getcwd().replace('\\','/')
    print('Current Python File Path: ', currentPythonFilePath)
    print('data augmentation')
    cmdAug= 'python "{}"/facerecognition/Data_Augmentation.py'.format(currentPythonFilePath)

    resultAug = subprocess.run(cmdAug, stdout=subprocess.PIPE, shell=True)

    print(resultAug.stdout)
    print('align dataset')
    
    cmdAlign = 'python "{}"/facerecognition/align_dataset_mtcnn.py "{}"/static/data_process/raw/ "{}"/static/data_process/process/ "{}"/static/align --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25'.format(currentPythonFilePath, currentPythonFilePath, currentPythonFilePath, currentPythonFilePath)

    resultAlign = subprocess.run(cmdAlign, stdout=subprocess.PIPE, shell=True)

    print(resultAlign.stdout)

    print('train classifier')

    cmdClass = 'python "{}"/facerecognition/classifier.py TRAIN "{}"/static/data_process/process "{}"/static/Models/20180402-114759.pb "{}"/static/Models/facemodel.pkl --batch_size 1000'.format(currentPythonFilePath, currentPythonFilePath, currentPythonFilePath, currentPythonFilePath)

    resultClass = subprocess.run(cmdClass, stdout=subprocess.PIPE, shell=True)

    # Print the output of the subprocess
    print(resultClass.stdout)
    train_datetime = datetime.now()

# Định dạng ngày tháng năm theo định dạng Việt Nam (dd/mm/yyyy)
    date_str = train_datetime.strftime("%d/%m/%Y")

    # Định dạng giờ theo định dạng 24 giờ (HH:MM)
    time_str = train_datetime.strftime("%H:%M")

    # Ghi thông tin vào file
    with open('time.txt', 'w',encoding='utf-8') as file:
        file.write(f'Thời gian train hoàn thành vào ngày: {date_str} {time_str}\n')
    # Return a success response
    return HttpResponse('ok luon')
class Camera_feed_identified(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.is_running = True
        (self.grabbed, self.frame) = self.video.read()
        #threading dung de chay song song voi chuong trinh chinh
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.is_running = False 
        self.__del__()

    def get_frame(self):
        image = self.frame
        #chuyển về màu RGB
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while self.is_running:
            try:
                grabbed, frame = self.video.read()                   
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Read barcodes from the image
                barcodes, imgbarcode = barcode_reader.read_barcodes(rgb)      
                # Detect faces in the frame
                faces, _= detector.get_faces(rgb)
                for face in faces:
                    x1, y1, x2, y2 = face[:4]
                    
                    # Get face image
                    face_img = rgb[int(y1):int(y2), int(x1):int(x2), :]
                    #lưu ảnh vào thư mục static
                    
                    # Get face embeddings
                    embeddings = detector.get_embeddings(face_img)
                    
                    # Recognize face
                    name, prob = recognizer.recognize_face(embeddings) 
                    #lưu name vào file data.txt
                    # with open('data.txt', 'w',encoding='utf-8') as file:
                    #     file.write(name)
                    #     #close file
                    #     file.close()
                    
                    if barcodes != []:           
                        cv2.rectangle(imgbarcode, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(imgbarcode, "{} {:.2f}".format(name, prob), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        rgb_frame = cv2.cvtColor(imgbarcode, cv2.COLOR_BGR2RGB)
                        
                    else:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, "{} {:.2f}".format(name, prob), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frame = rgb_frame
                    self.grabbed = grabbed
            except:
                try:
                    grabbed, frame = self.video.read()
                    self.frame = frame
                    self.grabbed = grabbed
                except:
                    pass

def Gender_frame(camera):
    while camera.is_running:
        try:
            frame = camera.get_frame()
        
            yield (b'--frame\r\n'
                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except:
            pass
def data_feed():
    try:
        with open(currentPythonFilePath+'/data.txt', 'r',encoding='utf-8') as file:
            data = file.read()
        return data
    except:
        return 'Chưa có dữ liệu'