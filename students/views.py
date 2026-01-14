from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView

from .models import Student
from attendance_tracker.models import *

class StudentsListView(LoginRequiredMixin, ListView):
    model = Student
    login_url = 'account_login'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        students_list = Student.objects.filter(center=self.request.user)
        students = {}
        for student in students_list:
            attendances = len(Attendance.objects.filter(student=student))
            attendance_rate = (len(Attendance.objects.filter(student=student, status=True))/attendances)*100
            students[student] = attendance_rate
        context['students'] = students
        return context