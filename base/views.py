from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.http import HttpResponse
from .models import User
from .forms import  UserForm, MyUserCreationForm
import joblib
from django.http import JsonResponse
import numpy as np
from .forms import PredictionForm
import os
# Create your views here.

def loginPage(request):
    page='login'
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        email = request.POST.get('email').lower()
        password = request.POST.get('password')

        try:
            user = User.objects.get(email=email)
        except:
            messages.error(request, 'User does not exist')
        user = authenticate(request, email=email, password=password)

        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Username Or Password does not exist')
    context = {'page': page}
    return render(request, 'base/login_register.html', context)

def logoutUser(request):
    logout(request)
    return redirect('home')

def registerUser(request):
    form = MyUserCreationForm()
    if request.method == 'POST':
        form = MyUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.username = user.username.lower()
            user.save()
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'An error occured during registration')
    return render(request, 'base/login_register.html', {'form': form})

def home(request):
    return render(request, 'base/home.html')



def predict_view(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # get the input data from the form
            X = [
                [
                    form.cleaned_data['Number_of_establishments'],
                    form.cleaned_data['Number_of_rooms'],
                    form.cleaned_data['Number_of_bed_place'],
                    form.cleaned_data['Occupancy_rate_rooms'],
                    form.cleaned_data['Occupancy_rate_bed'],
                ]
            ]

            # load the model and use it to make a prediction
            model_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model.pkl')

# load the model into memory
            pipeline = joblib.load(model_file)
            predicted_value = pipeline.predict(X)[0]

            # render the response with the predicted value
            return render(request, 'base/predicted_value.html', {'predicted_value': predicted_value})
    else:
        form = PredictionForm()

    return render(request, 'base/predict_form.html', {'form': form})
