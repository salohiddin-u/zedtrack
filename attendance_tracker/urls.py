from django.urls import path

from .views import *

urlpatterns = [
    path('', DashboardView.as_view(), name="dashboard"),
    path('marking-attendance/', MarkingAttendance.as_view(), name="marking-attendance"),
]