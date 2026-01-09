from django.urls import path

from .views import *

urlpatterns = [
    path('', DashboardView.as_view(), name="dashboard"),
    path('mark-attendance/', MarkingAttendanceView.as_view(), name="marking-attendance"),
    path('mark-attendance/<int:a>/', MarkAttendanceView.as_view(), name="mark-attendance"),
    path('mark-attendance/create/<int:course_id>/', attendance_create, name="attendance-create"),
]