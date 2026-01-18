from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import *

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
            attendance_rate = (len(Attendance.objects.filter(student=student, status=True))/attendances)*100 if attendances else 0
            students[student] = attendance_rate
        context['students'] = students
        return context
    
class StudentCreateView(LoginRequiredMixin, CreateView):
    model = Student
    fields = ['first_name', 'last_name', 'phone_number', 'course', 'gender']
    login_url = 'account_login'
    success_url = reverse_lazy('students')


    def form_valid(self, form):
        form.instance.center = self.request.user
        print(form)
        return super().form_valid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['courses'] = Course.objects.filter(center=self.request.user)
        return context
