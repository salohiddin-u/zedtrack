from datetime import datetime, timedelta

from allauth.core.internal.httpkit import redirect
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, View, DetailView, CreateView
from django.db.models import Count, Q, F, FloatField, ExpressionWrapper, Case, When, Value
from django.utils import timezone

from attendance_tracker.models import Attendance
from courses.models import Course
from students.models import Student


class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard.html'
    login_url = 'account_login'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        since = timezone.now() - timedelta(days=30)

        user = self.request.user
        students = Student.objects.filter(center=user)
        present_yesterday = Attendance.objects.filter(center=user, status=True, time__date=datetime.today()-timedelta(days=1)).count()
        absent_yesterday = Attendance.objects.filter(center=user, status=False, time__date=datetime.today()-timedelta(days=1)).count()
        context['user'] = user
        context['students'] = students
        context['present_today'] = Attendance.objects.filter(center=user, status=True, time__date=datetime.today()).count()
        context['absent_today'] = Attendance.objects.filter(center=user, status=False, time__date=datetime.today()).count()
        context['present_yest_vs_tod'] = ((context['present_today'] - present_yesterday)/present_yesterday)*100 if present_yesterday else 0
        context['absent_yest_vs_tod'] = ((context['absent_today'] - absent_yesterday)/absent_yesterday)*100 if absent_yesterday else 0
        context['attendance_rate'] = (Attendance.objects.filter(center=user, status=True, time__date=datetime.today()).count()/Student.objects.filter(center=user).count())*100 if Student.objects.filter(center=user).count() else 0
        context['attendance_rate_yesterday'] = (Attendance.objects.filter(center=user, status=True, time__date=datetime.today()-timedelta(days=1)).count()/Student.objects.filter(center=user).count())*100 if Student.objects.filter(center=user).count() else 0
        context['diffrence_rate'] = context['attendance_rate'] - context['attendance_rate_yesterday']

        line_chart_labels = []
        line_chart_data = []


        for i in range(30):
            line_chart_data.append((Attendance.objects.filter(center=user, status=True, time__date=datetime.today() - timedelta(
                days=i)).count() / Student.objects.filter(center=user).count()) * 100 if Student.objects.filter(
                center=user).count() else 0)

            line_chart_labels.append(datetime.today().date() - timedelta(days=i))

        context['line_chart_data'] = line_chart_data
        context['line_chart_labels'] = line_chart_labels
        context['recents_attendance_records'] = Attendance.objects.filter(center=user, ).order_by("-time")[:5]

        since = timezone.now() - timedelta(days=30)

        context['high_attendance'] = (
            Student.objects.filter(center=user)
            .annotate(
                total=Count('attendance', filter=Q(attendance__time__gte=since)),
                present=Count('attendance', filter=Q(attendance__time__gte=since, attendance__status=True)),
            )
            .annotate(
                rate=Case(
                    When(total=0, then=Value(0.0)),
                    default=ExpressionWrapper(F('present') * 100.0 / F('total'), output_field=FloatField()),
                )
            )
            .filter(rate__gte=70)
            .order_by('-rate')[:10]
        )

        context['low_attendance'] = (
            Student.objects.filter(center=user)
            .annotate(
                total=Count('attendance', filter=Q(attendance__time__gte=since)),
                present=Count('attendance', filter=Q(attendance__time__gte=since, attendance__status=True)),
            )
            .annotate(
                rate=Case(
                    When(total=0, then=Value(0.0)),
                    default=ExpressionWrapper(F('present') * 100.0 / F('total'), output_field=FloatField()),
                )
            )
            .filter(rate__lte=60)
            .order_by('-rate')[:10]
        )

        return context

class MarkingAttendanceView(LoginRequiredMixin, TemplateView):
    template_name = 'marking_attendance.html'
    login_url = 'account_login'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['courses'] = Course.objects.filter(center=self.request.user)

        return context

class MarkAttendanceView(LoginRequiredMixin, TemplateView):
    template_name = 'mark_attendance.html'
    login_url = 'account_login'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        course_id = self.kwargs['a']
        context['students'] = Student.objects.filter(center=self.request.user, course__id=course_id)
        context['course_id'] = course_id

        return context

def attendance_create(request, course_id):
    if request.method == 'POST':
        students = Student.objects.filter(center=request.user, course__id=course_id)
        for student in students:
            if request.POST.get(f"status-{student.id}") != None:
                status = request.POST.get(f"status-{student.id}") == "present"
                print(status)
                course = Course.objects.get(id=course_id)
                attendance = Attendance.objects.create(student=student, time=timezone.now(), course=course,
                                                       status=status, center=request.user, marked_by=request.user)
        return redirect("/mark-attendance/")
