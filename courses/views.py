from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import ListView, CreateView

from .models import Course
from teachers.models import *

class CoursesListView(LoginRequiredMixin, ListView):
    model = Course
    login_url = 'account_login'
    context_object_name = 'courses'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["courses"] = Course.objects.filter(user=self.request.user)
        return context
        

class CourseCreateView(LoginRequiredMixin, CreateView):
    model = Course
    fields = ['name', 'teacher', 'time', 'days', 'description', 'user']
    login_url = 'account_login'
    success_url = reverse_lazy('courses')

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["teachers"] = Teacher.objects.filter(user=self.request.user)
        return context
    
    