from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, CreateView
from django.urls import reverse_lazy

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
    fields = ['first_name', 'last_name', 'phone_number']
    login_url = "account_login"
    success_url = reverse_lazy('teachers')
    template_name = "teachers/teacher_form.html"

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)
    