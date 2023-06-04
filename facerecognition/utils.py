from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from django.http import HttpResponse
import cv2
import time
import os
from . import FacialRecognition 
from django.contrib import messages
import subprocess
from django.contrib.auth.models import User
from employee.models import EmployeeDetail, Attendance
import os
import threading
from django.shortcuts import redirect
from datetime import datetime
import unidecode

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
    count = 0
    start = True
    folder_path = os.path.join(currentPythonFilePath, 'static', 'data', folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
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
                if start== True:
                    time.sleep(5)
                    start= False
                if faces_found > 1:
                        cv2.putText(rgb, "Chi Mot Khuon Mat", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                elif faces_found > 0:
                    x1, y1, x2, y2 = face[:4]
                    count += 1
                    # # Get face image
                    # face_img = rgb[int(y1):int(y2), int(x1):int(x2), :]
                    image_path = os.path.join(folder_path, folder_name +'_'+ str(count) + '.jpg')
                    cv2.imwrite(image_path, frame)
                    #draw bbox
                    show = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.imshow('Phat Hien Khuon Mat', show)
                    
        except Exception as e:
            continue

        # Check if window is closed
        if cv2.getWindowProperty('Phat Hien Khuon Mat', cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    messages.success(request, 'Có dữ liệu mới được thêm vào.')
    return HttpResponse('success')

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
    messages.success(request, 'Train dữ liệu thành công.')
    return HttpResponse('ok luon')

class Camera_feed_identified(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.is_running = True
        (self.grabbed, self.frame) = self.video.read()
        employees = EmployeeDetail.objects.all()
        self.student_dict = {}
        for student in employees:
            self.student_dict[student.emcode] = student.user.last_name + ' ' + student.user.first_name
        #threading dung de chay song song voi chuong trinh chinh
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()
    
    def get_name(self, emcode):
        return self.student_dict.get(emcode)
    
    def remove_diacritics(self, text):
        text = unidecode.unidecode(text)
        return text

    def stop(self):
        self.is_running = False 

    def get_frame(self):
        image = self.frame
        #chuyển về màu RGB
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    def perform_attendance(self,emcode, name):
        attendance = Attendance.objects.create(
            emcode=emcode,
            name=name,
            date=datetime.now().date(),
            time=datetime.now().time()
        )
        attendance.save()
    def update(self):
        start = True
        print(self.student_dict)
        while True:
            try:
                grabbed, frame = self.video.read()              
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Read barcodes from the image
                
                barcodes, x, y, w, h = barcode_reader.read_barcodes(rgb)
                
                if barcodes != []:
                    #get name
                    name = self.get_name(barcodes[-1])  
                    if name != None:
                       print (name)
                       #self.perform_attendance(barcodes[-1], name)
                       name = self.remove_diacritics(name)
                       cv2.putText(frame, name, (x[-1], y[-1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                       cv2.rectangle(frame, (x[-1], y[-1]), (x[-1] + w[-1], y[-1] + h[-1]), (0, 255, 0), 2)   
                    else:
                        name = 'Khong co trong he thong'
                        cv2.putText(frame, name, (x[-1], y[-1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x[-1], y[-1]), (x[-1] + w[-1], y[-1] + h[-1]), (0, 255, 0), 2)
                    
                       
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
                    emcode, prob = recognizer.recognize_face(embeddings)
                    if start:
                        time.sleep(3)
                        start = False
                    if emcode != 'unknown':
                        name = self.get_name(emcode)
                        #lưu vào bảng điểm danh
                        if name is not None:
                            #self.perform_attendance(emcode, name)
                            #xóa dấu tiếng việt
                            name = self.remove_diacritics(name)
                        else:
                            name = 'Khong xac dinh'
                    else:
                        name = 'Khong xac dinh'
                    # Draw rectangle and name
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    if name == 'Khong xac dinh':
                        name = 'Khong co trong he thong'
                        cv2.putText(frame, "{}".format(name), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "{} {:.2f}".format(name, prob), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    self.frame = frame
                    self.grabbed = grabbed
            except:
                try:
                    grabbed, frame = self.video.read()
                    self.frame = frame
                    self.grabbed = grabbed
                except:
                    pass
            if self.is_running == False:
                break

def Gender_frame(camera):
    while True:
        try:

            frame = camera.get_frame()
        
            yield (b'--frame\r\n'
                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            
            if camera.is_running == False:
                break
        except GeneratorExit:
            
            break
            
        except:
            pass


def data_feed():
    try:
        with open(currentPythonFilePath+'/data.txt', 'r',encoding='utf-8') as file:
            data = file.read()
        return data
    except:
        return 'Chưa có dữ liệu'