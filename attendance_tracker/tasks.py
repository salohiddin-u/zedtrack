from celery import shared_task
from django.utils.timezone import now
from django.contrib.auth.models import User
from .models import *

@shared_task
def create_daily_rate():
    for i in User.objects.all():
        attendance_rate = (Attendance.objects.filter(center=i, status=True, time__date=now().date()).count() / Student.objects.filter(
            center=i).count()) * 100 if Student.objects.filter(center=i).count() else 0
        AttendanceRate.objects.create(center=i, rate=attendance_rate)