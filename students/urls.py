from django.urls import path

from .views import *

urlpatterns = [
    path('', StudentsListView.as_view(), name="students"),
]