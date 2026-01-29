from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, CreateView

from .models import Teacher
from students.models import *

class TeachersListView(LoginRequiredMixin, ListView):
    model = Teacher
    login_url = 'account_login'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["teachers"] = Teacher.objects.filter(user=self.request.user)

        return context
    
class TeacherCreateView(LoginRequiredMixin, CreateView):
    model = Student
    fields = ['']
    login_url = "account_login"