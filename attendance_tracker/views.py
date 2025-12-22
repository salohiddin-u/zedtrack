from datetime import datetime, timedelta

from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, View

from attendance_tracker.models import Attendance
from students.models import Student


class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard.html'
    login_url = 'account_login'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        user = self.request.user
        students = Student.objects.filter(user=user)
        present_yesterday = Attendance.objects.filter(user=user, status=True, time__date=datetime.today()-timedelta(days=1)).count()
        absent_yesterday = Attendance.objects.filter(user=user, status=False, time__date=datetime.today()-timedelta(days=1)).count()
        context['user'] = user
        context['students'] = students
        context['present_today'] = Attendance.objects.filter(user=user, status=True, time__date=datetime.today()).count()
        context['absent_today'] = Attendance.objects.filter(user=user, status=False, time__date=datetime.today()).count()
        context['present_yest_vs_tod'] = ((context['present_today'] - present_yesterday)/present_yesterday)*100 if present_yesterday else 0
        context['absent_yest_vs_tod'] = ((context['absent_today'] - absent_yesterday)/absent_yesterday)*100 if absent_yesterday else 0
        context['attendance_rate'] = (Attendance.objects.filter(user=user, status=True, time__date=datetime.today()).count()/Student.objects.filter(user=user).count())*100 if Student.objects.filter(user=user).count() else 0
        context['attendance_rate_yesterday'] = (Attendance.objects.filter(user=user, status=True, time__date=datetime.today()-timedelta(days=1)).count()/Student.objects.filter(user=user).count())*100 if Student.objects.filter(user=user).count() else 0
        context['diffrence_rate'] = context['attendance_rate'] - context['attendance_rate_yesterday']
        return context