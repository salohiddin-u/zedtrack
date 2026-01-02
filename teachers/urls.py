from django.urls import path

from .views import *

urlpatterns = [
    path('', TeachersListView.as_view(), name="teachers"),
]