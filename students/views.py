from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView

from .models import Student


class StudentsListView(LoginRequiredMixin, ListView):
    model = Student
    login_url = 'account_login'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        students = Student.objects.filter(center=self.request.user)
        context["students"] = students
        
        print(students)
        return context