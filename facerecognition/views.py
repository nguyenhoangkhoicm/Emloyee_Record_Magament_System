from django.shortcuts import render
from .util import *
from django.http import StreamingHttpResponse

def demorecognition(request):
    return render(request, 'demorecognition.html', {})


def recognition(request):
    video_capture = cv2.VideoCapture(0)  # Mở kết nối với webcam

    while True:
    # Đọc khung hình từ webcam
        #try:
            ret, frame = video_capture.read()  
            print("1")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Read barcodes from the image
            barcodes, imgbarcode = barcode_reader.read_barcodes(rgb)      
            # Detect faces in the frame
            faces, _= detector.get_faces(rgb)
            print("2")
            for face in faces:
                x1, y1, x2, y2 = face[:4]
                
                # Get face image
                face_img = rgb[int(y1):int(y2), int(x1):int(x2), :]
                #lưu ảnh vào thư mục static
                
                # Get face embeddings
                embeddings = detector.get_embeddings(face_img)
                
                # Recognize face
                name, prob = recognizer.recognize_face(embeddings) 
                print("3")
                if barcodes != []:           
                    cv2.rectangle(imgbarcode, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(imgbarcode, "{} {:.2f}".format(name, prob), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    rgb_frame = cv2.cvtColor(imgbarcode, cv2.COLOR_BGR2RGB)
                    print("4")
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, "{} {:.2f}".format(name, prob), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    print("5")

                # Chuyển đổi khung hình thành dạng chuỗi để truyền vào template
                print("6")
                _, jpeg_frame = cv2.imencode('.jpg', rgb_frame)
                data = jpeg_frame.tobytes()
                print("7")
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n\r\n')
                print("8")
        # except Exception as e:
        #     continue
        
    #video_capture.release()

def recognition_feed(request):
    return StreamingHttpResponse(recognition(request), content_type='multipart/x-mixed-replace; boundary=frame')