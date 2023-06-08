"""EmployeeRecordMgmt URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from employee.views import *
from facerecognition.views import *
from django.conf.urls import include

urlpatterns = [

    path('admin/', admin.site.urls),
    path('', load, name='load'),
    path('index', index, name='index'),
    path('logout', Logout, name='logout'),
    path('faceregconition/', include('facerecognition.urls')),
    path('admin_login/', admin_login, name='admin_login'),
    path('admin_home/', admin_home, name='admin_home'),
    path('change_passwordadmin/', change_passwordadmin,
         name='change_passwordadmin'),
    path('all_employee/', all_employee, name='all_employee'),
    path('delete_employee/<int:id>/', delete_employee, name='delete_employee'),
    path('create_folder/', create_folder, name='create_folder'),
    path('upload_images/', upload_images, name='upload_images'),
    path('', demorecognition),
    path('demorecognition/face_recognition',
         face_recognition, name='run_recognition'),
    path('demorecognition/face_detection/',
         face_detection, name='run_detection'),
    path('demorecognition/train/', train, name='run_train'),
    path('ifter/', ifter, name='ifter'),
    path('timetrain/', timetrain, name='timetrain'),
    path('ad_train/', ad_train, name='ad_train'),
    path('ad_attendance/', ad_attendance, name='ad_attendance'),
    path('ad_registration/', ad_registration, name='ad_registration'),
    path('save_registration/', save_registration, name='save_registration'),
    path('identified/', identified, name='identified'),
    path('attendee_list/', attendee_list, name='attendee_list'),
    path('ad_showatt/', ad_showatt, name='ad_showatt'),
    path('download_excel/', download_excel, name='download_excel'),
    path('attendance/',query_time_attendance,name='run_attendance'),
    path('query_attendance_all/',query_attendance_all,name='query_attendance_all'),

   
]
