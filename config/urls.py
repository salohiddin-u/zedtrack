from django.contrib import admin
from django.urls import path, include

from mark_attendance.views import *


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('mark_attendance.urls')),
    path('students/', include('students.urls')),
    path('teachers/', include('teachers.urls')),
    path('courses/', include('courses.urls')),
    path('accounts/', include('allauth.urls')),

]
