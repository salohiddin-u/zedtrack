from django.urls import path

from .views import *

urlpatterns = [
    path('', CoursesListView.as_view(), name="courses"),
]