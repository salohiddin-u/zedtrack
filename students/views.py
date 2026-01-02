from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView

from .models import Student


class StudentsListView(LoginRequiredMixin, ListView):
    model = Student
    login_url = 'account_login'