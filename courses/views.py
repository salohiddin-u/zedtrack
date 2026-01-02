from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView

from .models import Course

class CoursesListView(LoginRequiredMixin, ListView):
    model = Course
    login_url = 'account_login'
    context_object_name = 'courses'

    def get_queryset(self):
        return Course.objects.filter(center=self.request.user)