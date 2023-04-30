from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from .models import User
from django import forms

class MyUserCreationForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['name', 'username', 'email', 'password1', 'password2']

class UserForm(ModelForm):
    class Meta:
        model = User
        fields = ['avatar', 'name', 'username', 'email', 'bio']

class PredictionForm(forms.Form):
    Number_of_establishments = forms.IntegerField(label='Number of establishments')
    Number_of_rooms = forms.IntegerField(label='Number of rooms')
    Number_of_bed_place = forms.IntegerField(label='Number of bed places')
    Occupancy_rate_rooms = forms.FloatField(label='Occupancy rate of rooms')
    Occupancy_rate_bed = forms.FloatField(label='Occupancy rate of beds')