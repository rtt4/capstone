from django import forms

class SurveyForm(forms.Form):
    survey = forms.ImageField(label='Select a meta')
    data = forms.FileField(label='Select a data')
    # survey = forms.FileField(
    #     label='Select a file'
    #     # help_text='max. 42 megabytes'
    # )

class FileForm(forms.Form):
    name = forms.CharField()

class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()
