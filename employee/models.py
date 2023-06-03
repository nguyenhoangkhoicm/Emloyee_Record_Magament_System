from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class EmployeeDetail(models.Model):
    user= models.ForeignKey(User,on_delete=models.CASCADE)
    emcode=models.CharField(max_length=50)
    emdept=models.CharField(max_length=100,null=True)
    designation=models.CharField(max_length=100,null=True)
    contact=models.CharField(max_length=15,null=True)
    gender=models.CharField(max_length=50,null=True)
    joiningdate=models.DateField(null=True)
    def __str__(self): 
        return self.user.username
    
# class EmployeeEducation(models.Model):
#     user= models.ForeignKey(User,on_delete=models.CASCADE)
#     coursepg=models.CharField(max_length=100,null=True)
#     shoolclgpg=models.CharField(max_length=200,null=True)
#     yearofpassingpg=models.CharField(max_length=20,null=True)
#     percentagepg=models.CharField(max_length=30,null=True)

#     coursepgra=models.CharField(max_length=100,null=True)
#     shoolclgpggra=models.CharField(max_length=200,null=True)
#     yearofpassingra=models.CharField(max_length=20,null=True)
#     percentagegra=models.CharField(max_length=30,null=True)

#     coursessc=models.CharField(max_length=100,null=True)
#     shoolclgssc=models.CharField(max_length=200,null=True)
#     yearofpassingssc=models.CharField(max_length=20,null=True)
#     percentagessc=models.CharField(max_length=30,null=True)
    
#     coursehsc=models.CharField(max_length=100,null=True)
#     shoolclghsc=models.CharField(max_length=200,null=True)
#     yearofpassighsc=models.CharField(max_length=20,null=True)
#     percentagehsc=models.CharField(max_length=30,null=True)
    
#     def __str__(self): 
#         return self.user.username

# class EmployeeExperience(models.Model):
#     user= models.ForeignKey(User,on_delete=models.CASCADE)
#     companylname=models.CharField(max_length=100,null=True)
#     companyldesig=models.CharField(max_length=100,null=True)
#     companylsalary=models.CharField(max_length=100,null=True)
#     companylduration=models.CharField(max_length=100,null=True)
#     company1name=models.CharField(max_length=100,null=True)
#     company1desig=models.CharField(max_length=100,null=True)
#     company1salary=models.CharField(max_length=100,null=True)
#     company1duration=models.CharField(max_length=100,null=True)
#     company2name=models.CharField(max_length=100,null=True)
#     company2desig=models.CharField(max_length=100,null=True)
#     company2salary=models.CharField(max_length=100,null=True)
#     company2duration=models.CharField(max_length=100,null=True)
#     company3name=models.CharField(max_length=100,null=True)
#     company3desig=models.CharField(max_length=100,null=True)
#     company3salary=models.CharField(max_length=100,null=True)
#     company3duration=models.CharField(max_length=100,null=True)
    
#     def __str__(self): 
#         return self.user.username
    
class Attendance(models.Model):
    emcode = models.CharField(max_length=50)
    name = models.CharField(max_length=100)
    date = models.DateField()
    time = models.TimeField()

    def __str__(self):
        return f"Attendance for {self.name} ({self.emcode})"
