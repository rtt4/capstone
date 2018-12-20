from django import forms
from .models import MetaSurvey
from django.forms import ImageField


class SurveyForm(forms.Form):
    title = forms.CharField(max_length=50)
    survey = ImageField()
    data = forms.FileField(label='Select a data')
    # resized_survey = forms.ImageField()

    # class Meta:
    #     model = MetaSurvey
    #     fields = ['title', 'survey', 'data']
    #     exclude = ['upload_date']
