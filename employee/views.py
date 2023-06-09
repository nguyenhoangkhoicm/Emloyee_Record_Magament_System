from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404, redirect
from django.shortcuts import render, redirect
from .models import *
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from django.http import JsonResponse,HttpResponse
from django.conf import settings

import os
import shutil
import qrcode
import json
import datetime
import pandas as pd
# Create your views here.

def load(request):
    return render(request, 'loading.html')

def index(request):
   
    return render(request, 'index.html')

def leave(request):
    if request.method == 'POST':
        empcode = request.POST['empcode']
        name = request.POST['name']
        reason = request.POST['reason']
        date = request.POST['date']
        leave = Leave(emcode = empcode, name=name, reason=reason, date=date)
        leave.save()
        messages.success(request, 'Bạn đã gửi đơn xin nghỉ thành công.')
        return redirect('leave')
    else:
        leave = Leave.objects.all()
        return render(request, 'leave.html', {'leaves': leave})

def attendee_list(request):
    attendance = Attendance.objects.filter(date=datetime.date.today())
    name_list = []
    for i in attendance:
        name_list.append(i.name)
    
    return HttpResponse(json.dumps(name_list), content_type='application/json')

def ifter(request):
    return render(request, 'interface.html')

def ad_attendance(request):
    attendances = Attendance.objects.all()
    return render(request, 'ad_attendance.html', {'attendances': attendances})

def download_excel(request):
    file_path = os.path.join(settings.BASE_DIR, './attendance.xlsx')  # Đường dẫn tới file Excel 
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as excel_file:
            response = HttpResponse(excel_file.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            response['Content-Disposition'] = 'attachment; filename="attendance.xlsx"'
            return response
    else:
        return HttpResponse("File not found.")

def ad_showatt(request):
    # Đường dẫn đến tệp Excel
    excel_file = './attendance.xlsx'
    
    # Đọc tệp Excel và chuyển đổi thành HTML
    df = pd.read_excel(excel_file)
    df.fillna('', inplace=True)
    html_table = df.to_html(classes='table table-striped table-bordered')
    
    # Truyền HTML vào template
    context = {'html_table': html_table}
   
    return render(request, 'ad_showatt.html', context)


def ad_train(request):
    notify = request.GET.get('notify', None)
    if notify == '1':
        messages.success(
            request, 'Bạn vừa xóa tài khoản cần train dữ liệu lại.')

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
 
# hàm tạo qr code
def create_qrcode(text):
    # Tạo đối tượng QR code
    qr = qrcode.QRCode(
        version=4,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    # Thêm dữ liệu vào QR code
    qr.add_data(text)
    qr.make(fit=True)

    # Tạo ảnh QR code từ đối tượng QR code
    qr_image = qr.make_image(fill_color="black", back_color="white")

    # Lưu ảnh QR code thành file tại thư mục 'static/images/qrcode'
    save_path = os.path.join('static/images/qrcode', f'{text}.png')
    save_path = save_path.replace('\\', '/')  # Thay thế ký tự '\' bằng '/'
    file_path = os.path.join(settings.BASE_DIR, 'qrcode_path.txt')
    with open(file_path, 'w') as file:
        file.write('/'.join(save_path.split('/')[3:]))  # Ghi đường dẫn sau từ khóa "static" vào file
    qr_image.save(save_path)

    return save_path


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


def ad_registration(request):
    qrcode_path = []
    # Đọc nội dung của tệp tin txt
    with open('qrcode_path.txt', 'r') as file:
        content = [line.strip() for line in file]
    return render(request, 'admin_rg.html', {'content': content})


def save_registration(request):
    error = ""

    if request.method == "POST":
        fn = request.POST['firstname']
        ln = request.POST['lastname']
        ec = request.POST['empcode']
        em = request.POST['email']
        ad = request.POST['address']
        dp = request.POST['department']
        ge = request.POST['gender']

        try:
            user = User.objects.create_user(
                first_name=fn, last_name=ln, username=em)
            EmployeeDetail.objects.create(
                user=user, emcode=ec, address=ad, department=dp, gender=ge)

            error = "no"
            create_qrcode(ec)
        except:
            error = "yes"

    return JsonResponse({'error': error})


def Logout(request):
    logout(request)
    return redirect('index')


def admin_login(request):

    return render(request, 'admin_login.html')


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
            folder_path = os.path.join(
                'static', 'data_process', 'process', str(employee_code))
            full_path = os.path.abspath(folder_path)
            shutil.rmtree(full_path)

            folder_path = os.path.join(
                'static', 'data_process', 'raw', str(employee_code))
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
