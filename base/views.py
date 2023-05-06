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
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

def registerPage(request):
    form = MyUserCreationForm()

    if request.method == 'POST':
        form = MyUserCreationForm(request.POST)
        if form.is_valid():
            print(form)
            user = form.save(commit=False)
            user.username = user.username.lower()
            user.save()
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'An error occurred during registration')

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
            print(joblib.__version__)

# load the model into memory
            pipeline = joblib.load(model_file)
            predicted_value = pipeline.predict(X)[0]

            # render the response with the predicted value
            return render(request, 'base/predicted_value.html', {'predicted_value': predicted_value})
    else:
        form = PredictionForm()

    return render(request, 'base/predict_form.html', {'form': form})



hotel = pd.read_csv(r'C:\Users\medGhaith\Desktop\pi-web\pi\base\hotel.csv')

def requirementbased(city,number,features):
    hotel['city']=hotel['city'].str.lower()
    hotel['roomamenities']=hotel['roomamenities'].str.lower()
    features=features.lower()
    features_tokens=word_tokenize(features)  
    sw = stopwords.words('english')
    lemm = WordNetLemmatizer()
    f1_set = {w for w in features_tokens if not w in sw}
    f_set=set()
    for se in f1_set:
        f_set.add(lemm.lemmatize(se))
    reqbased=hotel[hotel['city']==city.lower()]
    reqbased=reqbased[reqbased['guests_no']==number]
    reqbased=reqbased.set_index(np.arange(reqbased.shape[0]))
    l1 =[];l2 =[];cos=[];
    #print(reqbased['roomamenities'])
    for i in range(reqbased.shape[0]):
        temp_tokens=word_tokenize(reqbased['roomamenities'][i])
        temp1_set={w for w in temp_tokens if not w in sw}
        temp_set=set()
        for se in temp1_set:
            temp_set.add(lemm.lemmatize(se))
        rvector = temp_set.intersection(f_set)
        #print(rvector)
        cos.append(len(rvector))
    reqbased['similarity']=cos
    reqbased=reqbased.sort_values(by='similarity',ascending=False)
    reqbased.drop_duplicates(subset='hotelcode',keep='first',inplace=True)
    return reqbased[['hotelname','roomtype','guests_no','starrating','address','roomamenities','ratedescription','similarity']].head(10)


def recommendation_view(request):
    if request.method == 'POST':
        # Get the user inputs from the form
        city = request.POST['city']
        number = int(request.POST['number'])
        features = request.POST['features']
        
        # Call the recommendation function to generate recommendations based on the inputs
        recommendations = requirementbased(city, number, features)
        recommendations_list = []
        for index, row in recommendations.iterrows():
            recommendations_list.append({
                'hotelname': row['hotelname'],
                'address': row['address']
            })
        print(recommendations)
        # Pass the recommendations to the template
        return render(request, 'base/recommendations.html', {'recommendations': recommendations_list})
    else:
        # Render the form
        return render(request, 'base/input_form.html')

