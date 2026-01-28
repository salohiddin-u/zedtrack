from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView

from .models import Teacher

class TeachersListView(LoginRequiredMixin, ListView):
    model = Teacher
    login_url = 'account_login'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["teachers"] = Teacher.objects.filter(user=self.request.user)

        return context