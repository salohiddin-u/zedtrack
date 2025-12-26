from django.contrib import admin
from django.urls import path, include

from attendance_tracker.views import *


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('attendance_tracker.urls')),
    path('accounts/', include('allauth.urls')),

]
