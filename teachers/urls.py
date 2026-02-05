from django.urls import path

from .views import *

urlpatterns = [
    path('', TeachersListView.as_view(), name="teachers"),
    path('create/', TeacherCreateView.as_view(), name="create_teacher")
]