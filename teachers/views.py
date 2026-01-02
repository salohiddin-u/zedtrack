from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView

from .models import Teacher

class TeachersListView(LoginRequiredMixin, ListView):
    model = Teacher
    login_url = 'account_login'