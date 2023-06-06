from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class EmployeeDetail(models.Model):
    user= models.ForeignKey(User,on_delete=models.CASCADE)
    emcode=models.CharField(max_length=50)
    address=models.CharField(max_length=50)
    department=models.CharField(max_length=50)
    gender=models.CharField(max_length=10)
    def __str__(self): 
        return self.user.username


class Attendance(models.Model):
    emcode = models.CharField(max_length=50)
    name = models.CharField(max_length=100)
    date = models.DateField()
    time = models.TimeField()
    counter = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"Attendance for {self.name} ({self.emcode})"
