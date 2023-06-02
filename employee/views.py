from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404, redirect
from django.shortcuts import render, redirect
from .models import *
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from django.urls import reverse
import os
import shutil
from django.http import JsonResponse, HttpResponseRedirect
import qrcode


# Create your views here.


def index(request):
    return render(request, 'index.html')


def ifter(request):
    return render(request, 'interface.html')


def ad_train(request):
    notify = request.GET.get('notify', None)
    if notify == '1':
        messages.success(request, 'Bạn vừa xóa tài khoản cần train dữ liệu lại.')

    return render(request, 'ad_train.html')


def timetrain(request):
   # Đọc nội dung từ file txt
    train_datetimes = []
    with open('time.txt', 'r', encoding='utf-8') as file:
        train_datetimes = file.readlines()

    # Xóa ký tự xuống dòng ở cuối mỗi dòng
    train_datetimes = [
        datetime.strip() + '<br>' for datetime in train_datetimes]

    # Gửi train_datetimes đến template dưới dạng JSON
    data = {'train_datetimes': train_datetimes}
    return JsonResponse(data)
#hàm tạo qr code
def create_qrcode(text):
    # Tạo đối tượng QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )

    # Thêm dữ liệu vào QR code
    path = os.getcwd()
    qr.add_data(text)
    qr.make(fit=True)

    # Tạo ảnh QR code từ đối tượng QR code
    qr_image = qr.make_image(fill_color="black", back_color="white")

    # Lưu ảnh QR code thành file path + /static/images/qrcode/<text>.png
    qr_image.save(os.path.join(path, 'static/images/qrcode', f'{text}.png'))

def create_folder(request):
    folder_name = request.GET.get('name')
    folder_path = os.path.join('./static/data/', folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return render(request, 'upload.html', {'folder_name': folder_name})
    else:
        messages.error(request, 'Tên thư mục đã tồn tại')
        return JsonResponse({'message': 'Tên thư mục đã tồn tại'})


def upload_images(request):
    if request.method == 'POST':
        folder_name = request.POST.get('folder_name')
        folder_path = os.path.join('./static/data/', folder_name)
        # Lưu hình ảnh vào thư mục
        for i in range(10):
            image_file = request.FILES.get(f'image_{i+1}')
            if image_file:
                with open(os.path.join(folder_path, image_file.name), 'wb') as f:
                    f.write(image_file.read())
    messages.success(request, 'Có dữ liệu mới được thêm vào.')
    return redirect('ad_registration')


def registration(request):

    error = ""
    if request.method == "POST":
        fn = request.POST['firstname']
        ln = request.POST['lastname']
        ec = request.POST['empcode']
        em = request.POST['email']

        try:
            user = User.objects.create_user(
                first_name=fn, last_name=ln, username=em)
            EmployeeDetail.objects.create(user=user, emcode=ec)
            EmployeeExperience.objects.create(user=user)
            EmployeeEducation.objects.create(user=user)
            error = "no"
        except:
            error = "yes"

    return render(request, 'registration.html', locals())


def ad_registration(request):
    error = ""

    if request.method == "POST":
        fn = request.POST['firstname']
        ln = request.POST['lastname']
        ec = request.POST['empcode']
        em = request.POST['email']

        try:
            user = User.objects.create_user(
                first_name=fn, last_name=ln, username=em)
            EmployeeDetail.objects.create(user=user, emcode=ec)
            EmployeeExperience.objects.create(user=user)
            EmployeeEducation.objects.create(user=user)
            error = "no"
        except:
            error = "yes"

    return render(request, 'admin_rg.html', locals())


def save_registration(request):
    error = ""

    if request.method == "POST":
        fn = request.POST['empcode']
        ln = request.POST['lastname']
        ec = request.POST['firstname']
        em = request.POST['email']

        try:
            user = User.objects.create_user(
                first_name=fn, last_name=ln, username=em)
            EmployeeDetail.objects.create(user=user, emcode=ec)
            EmployeeExperience.objects.create(user=user)
            EmployeeEducation.objects.create(user=user)
            error = "no"
            create_qrcode(fn)
        except:
            error = "yes"

    return JsonResponse({'error': error})


def create_acc(request):
    error = ""

    if request.method == "POST":
        fn = request.POST['firstname']
        ln = request.POST['lastname']
        ec = request.POST['empcode']
        em = request.POST['email']
        pwd = request.POST['pwd']
        try:
            user = User.objects.create_user(
                first_name=fn, last_name=ln, username=em, password=pwd)
            EmployeeDetail.objects.create(user=user, emcode=ec)
            EmployeeExperience.objects.create(user=user)
            EmployeeEducation.objects.create(user=user)
            error = "no"
        except:
            error = "yes"
    return render(request, 'create_acc.html', locals())


def Logout(request):
    logout(request)
    return redirect('index')


def emp_login(request):
    error = ""
    if request.method == "POST":
        u = request.POST['emailid']
        p = request.POST['password']
        user = authenticate(username=u, password=p)
        if user:
            login(request, user)
            error = "no"
        else:
            error = "yes"
    return render(request, 'emp_login.html', locals())


def emp_home(request):
    if not request.user.is_authenticated:
        return redirect('emp_login')
    return render(request, 'emp_home.html')


def admin_login(request):

    return render(request, 'admin_login.html')


def profile(request):
    if not request.user.is_authenticated:
        return redirect('emp_login')
    error = ""
    user = request.user
    employee = EmployeeDetail.objects.get(user=user)
    if request.method == "POST":
        fn = request.POST['firstname']
        ln = request.POST['lastname']
        ec = request.POST['empcode']
        dept = request.POST['department']
        designation = request.POST['designation']
        contact = request.POST['contact']
        jdate = request.POST['jdate']
        gender = request.POST['gender']

        employee.user.first_name = fn
        employee.user.last_name = ln
        employee.emcode = ec
        employee.emdept = dept
        employee.designation = designation
        employee.contact = contact
        employee.gender = gender
        if jdate:
            employee.joiningdate = jdate

        try:
            employee.save()
            employee.user.save()
            error = "no"
        except:
            error = "yes"
    return render(request, 'profile.html', locals())


def my_experience(request):
    if not request.user.is_authenticated:
        return redirect('emp_login')

    user = request.user
    experience = EmployeeExperience.objects.get(user=user)

    return render(request, 'my_experience.html', locals())


def edit_myexperience(request):
    if not request.user.is_authenticated:
        return redirect('emp_login')
    error = ""
    user = request.user
    experience = EmployeeExperience.objects.get(user=user)
    if request.method == "POST":
        company1name = request.POST['company1name']
        company1desig = request.POST['company1desig']
        company1salary = request.POST['company1salary']
        company1duration = request.POST['company1duration']

        company2name = request.POST['company2name']
        company2desig = request.POST['company2desig']
        company2salary = request.POST['company2salary']
        company2duration = request.POST['company2duration']

        company3name = request.POST['company3name']
        company3desig = request.POST['company3desig']
        company3salary = request.POST['company3salary']
        company3duration = request.POST['company3duration']

        experience.company1name = company1name
        experience.company1desig = company1desig
        experience.company1salary = company1salary
        experience. company1duration = company1duration

        experience.company2name = company2name
        experience.company2desig = company2desig
        experience.company2salary = company2salary
        experience. company2duration = company2duration

        experience.company3name = company3name
        experience.company3desig = company3desig
        experience.company3salary = company3salary
        experience. company3duration = company3duration

        try:
            experience.save()

            error = "no"
        except:
            error = "yes"
    return render(request, 'edit_myexperience.html', locals())


def my_education(request):
    if not request.user.is_authenticated:
        return redirect('emp_login')

    user = request.user
    education = EmployeeEducation.objects.get(user=user)

    return render(request, 'my_education.html', locals())


def edit_myeducation(request):
    if not request.user.is_authenticated:
        return redirect('emp_login')
    error = ""
    user = request.user
    education = EmployeeEducation.objects.get(user=user)
    if request.method == "POST":
        coursepg = request.POST['coursepg']
        shoolclgpg = request.POST['shoolclgpg']
        yearofpassingpg = request.POST['yearofpassingpg']
        percentagepg = request.POST['percentagepg']

        coursepgra = request.POST['coursepgra']
        shoolclgpggra = request.POST['shoolclgpggra']
        yearofpassingra = request.POST['yearofpassingra']
        percentagegra = request.POST['percentagegra']

        coursessc = request.POST['coursessc']
        shoolclgssc = request.POST['shoolclgssc']
        yearofpassingssc = request.POST['yearofpassingssc']
        percentagessc = request.POST['percentagessc']

        coursehsc = request.POST['coursehsc']
        shoolclghsc = request.POST['shoolclghsc']
        yearofpassighsc = request.POST['yearofpassighsc']
        percentagehsc = request.POST['percentagehsc']

        education.coursepg = coursepg
        education.shoolclgpg = shoolclgpg
        education.yearofpassingpg = yearofpassingpg
        education. percentagepg = percentagepg

        education.coursepgra = coursepgra
        education.shoolclgpggra = shoolclgpggra
        education.yearofpassingra = yearofpassingra
        education. percentagegra = percentagegra

        education.coursessc = coursessc
        education.shoolclgssc = shoolclgssc
        education.yearofpassingssc = yearofpassingssc
        education. percentagessc = percentagessc

        education.coursehsc = coursehsc
        education.shoolclghsc = shoolclghsc
        education.yearofpassighsc = yearofpassighsc
        education. percentagehsc = percentagehsc

        try:
            education.save()

            error = "no"
        except:
            error = "yes"
    return render(request, 'edit_myeducation.html', locals())


def change_password(request):
    if not request.user.is_authenticated:
        return redirect('emp_login')
    error = ""
    user = request.user

    if request.method == "POST":
        c = request.POST['currentpassword']
        n = request.POST['newpassword']

        try:
            if user.check_password(c):
                user.set_password(n)
                user.save()
                error = "no"
            else:
                error = "not"
        except:
            error = "yes"
    return render(request, 'change_password.html', locals())


def admin_login(request):
    error = ""
    if request.method == "POST":
        u = request.POST['username']
        p = request.POST['pwd']
        user = authenticate(username=u, password=p)
        try:
            if user.is_staff:
                login(request, user)
                error = "no"
            else:
                error = "yes"
        except:
            error = "yes"
    return render(request, 'admin_login.html', locals())


def admin_home(request):
    if not request.user.is_authenticated:
        return redirect('admin_login')
    return render(request, 'admin_home.html')


def change_passwordadmin(request):
    if not request.user.is_authenticated:
        return redirect('admin_login')
    error = ""
    user = request.user

    if request.method == "POST":
        c = request.POST['currentpassword']
        n = request.POST['newpassword']

        try:
            if user.check_password(c):
                user.set_password(n)
                user.save()
                error = "no"
            else:
                error = "not"
        except:
            error = "yes"
    return render(request, 'change_passwordadmin.html', locals())


def all_employee(request):
    if not request.user.is_authenticated:
        return redirect('admin_login')
    employee = EmployeeDetail.objects.all()
    return render(request, 'all_employee.html', locals())


def edit_profile(request):
    if not request.user.is_authenticated:
        return redirect('admin_login')
    error = ""
    user = request.user
    # education= EmployeeDetail.objects.get(user=user)
    if request.method == "POST":
        coursepg = request.POST['coursepg']
        shoolclgpg = request.POST['shoolclgpg']
        yearofpassingpg = request.POST['yearofpassingpg']
        percentagepg = request.POST['percentagepg']

        coursepgra = request.POST['coursepgra']
        shoolclgpggra = request.POST['shoolclgpggra']
        yearofpassingra = request.POST['yearofpassingra']
        percentagegra = request.POST['percentagegra']

        coursessc = request.POST['coursessc']
        shoolclgssc = request.POST['shoolclgssc']
        yearofpassingssc = request.POST['yearofpassingssc']
        percentagessc = request.POST['percentagessc']

        coursehsc = request.POST['coursehsc']
        shoolclghsc = request.POST['shoolclghsc']
        yearofpassighsc = request.POST['yearofpassighsc']
        percentagehsc = request.POST['percentagehsc']

        user.coursepg = coursepg
        user.shoolclgpg = shoolclgpg
        user.yearofpassingpg = yearofpassingpg
        user. percentagepg = percentagepg

        user.coursepgra = coursepgra
        user.shoolclgpggra = shoolclgpggra
        user.yearofpassingra = yearofpassingra
        user. percentagegra = percentagegra

        user.coursessc = coursessc
        user.shoolclgssc = shoolclgssc
        user.yearofpassingssc = yearofpassingssc
        user. percentagessc = percentagessc

        user.coursehsc = coursehsc
        user.shoolclghsc = shoolclghsc
        user.yearofpassighsc = yearofpassighsc
        user. percentagehsc = percentagehsc

        try:
            user.save()

            error = "no"
        except:
            error = "yes"
    return render(request, 'edit_profile.html', locals())

# def delete_employee(request, id):
#     if not request.user.is_authenticated:
#         return redirect('admin_login')

#     employee = get_object_or_404(EmployeeDetail, pk=id)
#     user_id = employee.user_id

#     try:
#         user = User.objects.get(id=user_id)
#         user.delete()
#         employee.delete()
#         return redirect('all_employee')
#     except User.DoesNotExist:
#         # Xử lý lỗi khi người dùng không tồn tại
#         # ...
#         return redirect('all_employee')

def delete_employee(request, id):
    if not request.user.is_authenticated:
        return redirect('admin_login')

    employee = get_object_or_404(EmployeeDetail, pk=id)
    employee_code = employee.emcode
    user_id = employee.user_id

    try:
        user = User.objects.get(id=user_id)

        # Xóa thư mục theo mã nhân viên tương ứng
        folder_path = os.path.join('static', 'data', str(employee_code))
        full_path = os.path.abspath(folder_path)
        shutil.rmtree(full_path)
        try:
            folder_path = os.path.join('static', 'data_process','process', str(employee_code))
            full_path = os.path.abspath(folder_path)
            shutil.rmtree(full_path)

            folder_path = os.path.join('static', 'data_process','raw', str(employee_code))
            full_path = os.path.abspath(folder_path)
            shutil.rmtree(full_path)
        except:
            pass
        user.delete()
        employee.delete()
        messages.success(request, 'Bạn vừa xóa dữ liệu của nhân viên.')
        return redirect('all_employee')
    except User.DoesNotExist:
        
        return redirect('all_employee')