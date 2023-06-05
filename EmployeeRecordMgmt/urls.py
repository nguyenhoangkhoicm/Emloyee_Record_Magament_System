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
    path('',load,name='load'),
    path('index',index,name='index'),
    path('registration',registration,name='registration'),
    path('emp_login',emp_login,name='emp_login'),
    path('emp_home',emp_home,name='emp_home'),
    path('profile',profile,name='profile'),
    path('logout',Logout,name='logout'),
    path('faceregconition/', include('facerecognition.urls')),
    path('registration/',registration,name='registration'),
    path('admin_login/',admin_login,name='admin_login'),
    path('my_experience/',my_experience,name='my_experience'),
    path('edit_myexperience/',edit_myexperience,name='edit_myexperience'),
    path('my_education/',my_education,name='my_education'),
    path('edit_myeducation/',edit_myeducation,name='edit_myeducation'),
    path('change_password/',change_password,name='change_password'),
    path('admin_home/',admin_home,name='admin_home'),
    path('change_passwordadmin/',change_passwordadmin,name='change_passwordadmin'),
    path('all_employee/',all_employee,name='all_employee'),
    # path('edit_employee/<int:pk>/',edit_employee,name='edit_employee'),
    path('delete_employee/<int:id>/',delete_employee,name='delete_employee'),
    path('create_folder/',create_folder,name='create_folder'),
    path('upload_images/', upload_images, name='upload_images'),
    path('', demorecognition),
    path('demorecognition/face_recognition',face_recognition,name='run_recognition'),
    path('demorecognition/face_detection/',face_detection,name='run_detection'),
    path('create_acc/',create_acc,name='create_acc'),
    path('demorecognition/train/',train,name='run_train'),
    path('ifter/',ifter,name='ifter'),
    path('timetrain/',timetrain,name='timetrain'),
    path('ad_train/',ad_train,name='ad_train'),
    path('ad_registration/',ad_registration,name='ad_registration'),
    path('save_registration/',save_registration,name='save_registration'),
    path('identified/', identified, name='identified'),
    path('attendee_list/', attendee_list, name='attendee_list'),
]
