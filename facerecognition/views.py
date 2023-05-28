from django.shortcuts import render
from .utils import *
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.http import HttpResponse


def demorecognition(request):
    return render(request, 'demorecognition.html', {})

def identified(request):
    cam=Camera_feed_identified()
    if request.method == 'GET':
        try:
            return StreamingHttpResponse(Gender_frame(cam), content_type="multipart/x-mixed-replace;boundary=frame")
        except:
            pass
    elif request.method == 'POST':
        cam.stop()
        return HttpResponse("success")
