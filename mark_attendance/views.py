from datetime import datetime, timedelta

from allauth.core.internal.httpkit import redirect
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, View, DetailView, CreateView
from django.db.models import Count, Q, F, FloatField, ExpressionWrapper, Case, When, Value
from django.utils import timezone

from .models import *
from courses.models import Course
from students.models import Student
from teachers.models import Teacher

import json


class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard.html'
    login_url = 'account_login'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        since = timezone.now() - timedelta(days=30)

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
        context['attendance_rate_yesterday'] = (AttendanceRate.objects.filter(user=user, date__date=datetime.today()-timedelta(days=0)).values_list('rate', flat=True).first() or 0)
        context['diffrence_rate'] = context['attendance_rate'] - context['attendance_rate_yesterday']
        line_chart_labels = []
        line_chart_data = []

        line_chart_data.append((Attendance.objects.filter(user=self.request.user, status=True, time__date=datetime.today().date()).count() / 
                               Student.objects.filter(user=self.request.user).count()
                               ) * 100 if Student.objects.filter(user=self.request.user).count() else 0)
        line_chart_labels.append(datetime.today().date())
        for i in range(29):
            date = ((datetime.today() - timedelta(days=1)) - timedelta(days=i)).date()
            
            rate = AttendanceRate.objects.filter(
                user=self.request.user,
                date__date=date
            ).values_list('rate', flat=True).first() or 0
            line_chart_data.append(rate)
            line_chart_labels.append(date)
        
        
        context['line_chart_data'] = list(reversed(line_chart_data))
        context['line_chart_labels'] = list(reversed(line_chart_labels))
        context['recents_attendance_records'] = Attendance.objects.filter(user=user, ).order_by("-time")[:5]

        since = timezone.now() - timedelta(days=30)

        context['high_attendances'] = (
            Student.objects.filter(user=user)
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
            .order_by('-rate')[:7]
        )

        context['low_attendances'] = (
            Student.objects.filter(user=user)
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
            .filter(rate__lte=70)
            .order_by('-rate')[:10]
        )

        return context

class SelectCourse(LoginRequiredMixin, TemplateView):
    template_name = 'mark_attendance/select_course.html'
    login_url = 'account_login'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        courses = {}

        for course in Course.objects.filter(user=self.request.user):
            if Attendance.objects.filter(course__id=course.id, time__date=timezone.localdate()).exists():
                status = "Marked"
            else:
                status = "Not Marked"
            courses[course] = status
        context['courses'] = courses
        return context

class MarkAttendanceView(LoginRequiredMixin, TemplateView):
    template_name = 'mark_attendance/mark_attendance.html'
    login_url = 'account_login'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        course_id = self.kwargs['a']

        context['students'] = Student.objects.filter(user=self.request.user, course__id=course_id)
        context['course_id'] = course_id
        status = False
        attendances = Attendance.objects.filter(user=self.request.user, time__date=datetime.today().date(), course__id=course_id)
        if attendances.exists():
            status = True
            context["attendances"] = attendances
        context["status"] = status

        return context

def attendance_create(request, course_id):
    if request.method == 'POST':
        students = Student.objects.filter(user=request.user, course__id=course_id)
        for student in students:
            
            status = request.POST.get(f"status-{student.id}")
            if status == "present":
                status = True
            elif status == None:
                status = False
            course = Course.objects.get(id=course_id)
            attendance = Attendance.objects.create(student=student, time=timezone.now(), course=course,
                                                       status=status, user=request.user)
        return redirect("/mark-attendance/") 


class HistoryView(LoginRequiredMixin, TemplateView):
    template_name = "history.html"
    login_url = "account_login"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['courses'] = Course.objects.filter(user=self.request.user)
        context['teachers'] = Teacher.objects.filter(user=self.request.user)
        context["attendances"] = Attendance.objects.filter(user=self.request.user)

        start_date = self.request.GET.get('start_date')
        end_date = self.request.GET.get('end_date')
        course = self.request.GET.get('course')
        teacher = self.request.GET.get('teacher')

        queryset = Attendance.objects.all()

        if start_date and end_date:
            if start_date>end_date:
                context['error_message'] = "Start date cannot be after end date."
            context['attendances'] = queryset.filter(time__date__range=(start_date, end_date))
            context['start_date'] = start_date
            context['end_date'] = end_date
        elif start_date or end_date:
            context["error_message"] = "Provide both start date and end date."
        if course:
            if course and teacher and Teacher.objects.get(id=teacher).id != Course.objects.get(id=course).teacher_id:
                context["error_message"] = "Teacher mismatch"
            else:
                context['attendances'] = queryset.filter(course__id=course)
                context['course'] = Course.objects.get(id=course)
        if teacher:
            if course and teacher and Teacher.objects.get(id=teacher).id != Course.objects.get(id=course).teacher_id:
                context["error_message"] = "Teacher mismatch!"
            else:
                context['attendances'] = queryset.filter(course__teacher__id=teacher)
                context['teacher'] = Teacher.objects.get(id=teacher)

        return context
    


class ZedAI(LoginRequiredMixin, TemplateView):
    template_name = "zedai.html"
    login_url = "accounts/login"