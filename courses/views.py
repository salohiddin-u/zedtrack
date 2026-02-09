from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import ListView, CreateView

from .models import Course
from teachers.models import *
from students.models import *

class CoursesListView(LoginRequiredMixin, ListView):
    model = Course
    login_url = 'account_login'
    context_object_name = 'courses'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        courses = {}
        for course in Course.objects.filter(user=self.request.user):
            courses[course] = len(Student.objects.filter(course=course, user=self.request.user))
        context["courses"] = courses
        return context
        

class CourseCreateView(LoginRequiredMixin, CreateView):
    model = Course
    fields = ['name', 'teacher', 'time', 'days']
    login_url = 'account_login'
    success_url = reverse_lazy('courses')
    template_name = 'courses/course_form.html'

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["teachers"] = Teacher.objects.filter(user=self.request.user)
        return context
    
    