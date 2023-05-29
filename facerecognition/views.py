from django.shortcuts import render
from .utils import *
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.http import HttpResponse


def demorecognition(request):
    return render(request, 'demorecognition.html', {})

def identified(request):
    try:
        print(request.method)
        print("threadding 1")
        cam=Camera_feed_identified()
        gen = Gender_frame(cam)
        print("threadding 2")
        if request.method == 'GET':
            print("threadding 3")
            try:
                return StreamingHttpResponse(gen, content_type="multipart/x-mixed-replace;boundary=frame")
            except:
                pass
            print("threadding 4")
        elif request.method == 'POST':
            print("threadding 5")
            print("threadding",threading.active_count())
            if threading.active_count() > 0:
                gen.close()
                cam.stop()
                
                return HttpResponse("success")
            else:
                return HttpResponse("fail")
    except:
        print("lỗi rồi")
        return HttpResponse("lõ rồi")
