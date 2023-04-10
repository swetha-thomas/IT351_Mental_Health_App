"""attendance_system_facial_recognition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from recognition import views as recog_views
from users import views as users_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', recog_views.home, name='home'),
    
    path('dashboard/', recog_views.dashboard, name='dashboard'),
    path('login/',auth_views.LoginView.as_view(template_name='login.html'),name='login'),
    path('logout/',auth_views.LogoutView.as_view(template_name='home.html'),name='logout'),
    path('register/', users_views.register, name='register'),
    path('question1/', recog_views.question1, name='question1'),
    path('answer1/', recog_views.answer1, name='answer1'),
    path('question2/', recog_views.question2, name='question2'),
    path('question3/', recog_views.question3, name='question3'),
    path('question4/', recog_views.question4, name='question4'),
    path('question5/', recog_views.question5, name='question5'),
    path('clickPicture1/', recog_views.clickPicture1, name='clickPicture1'),
    path('clickPicture2/', recog_views.clickPicture2, name='clickPicture2'),
    path('clickPicture3/', recog_views.clickPicture3, name='clickPicture3'),
    path('clickPicture4/', recog_views.clickPicture4, name='clickPicture4'),
    path('clickPicture5/', recog_views.clickPicture5, name='clickPicture5'),   
    path('end/', recog_views.end, name='end'),
    path('calculating/', recog_views.calculating, name='calculating'),    

    path('mark_your_attendance', recog_views.mark_your_attendance ,name='mark-your-attendance'),
    path('mark_your_attendance_out/', recog_views.mark_your_attendance_out,name='mark_your_attendance_out'),
    path('view_attendance_home', recog_views.view_attendance_home ,name='view-attendance-home'),
    path('view_attendance_in', recog_views.view_attendance_in, name= 'view-attendance-in'),
    
     

]
