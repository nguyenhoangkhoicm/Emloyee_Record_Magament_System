from django.shortcuts import render
from .utils import *
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.http import HttpResponse


def demorecognition(request):
    return render(request, 'demorecognition.html', {})


def identified(request):
    try:
        cam=Camera_feed_identified()
        gen = Gender_frame(cam)
        if request.method == 'GET':
            try:
                return StreamingHttpResponse(gen, content_type="multipart/x-mixed-replace;boundary=frame")
            except:
                pass
        elif request.method == 'POST':
            if threading.active_count() > 0:
                cam.video.release()
                cam.stop()
                time.sleep(0.2)
                gen.close()
                messages.success(request, "Dừng thành công.")
                return HttpResponse("success")
            else:
                messages.error(request, "Dừng không thành công.")
                return HttpResponse("fail")
    except:
        print("lỗi rồi")
        return HttpResponse("lõ rồi")