from django.urls import path

from .views import *

urlpatterns = [
    path('', DashboardView.as_view(), name="dashboard"),
    path('mark-attendance/', SelectCourse.as_view(), name="select-course"),
    path('mark-attendance/<int:a>/', MarkAttendanceView.as_view(), name="mark-attendance"),
    path('mark-attendance/create/<int:course_id>/', attendance_create, name="attendance-create"),
    path('history/', HistoryView.as_view(), name="history"),
    path('history/<str:start_date>/<str:end_date>/', HistoryView.as_view(), name="history"),
]