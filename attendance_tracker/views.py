from django.views.generic import TemplateView, View

from students.models import Student


class DashboardView(TemplateView):
    template_name = 'dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        user = self.request.user
        students = Student.objects.filter(user=user)
        context['user'] = user
        context['students'] = students
        return context