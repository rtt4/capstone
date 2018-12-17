from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import *

# Create your views here.
def get_name(request):
    if request.method == 'POST':
        form = NameForm(request.POST)
        if form.is_valid():
            # 폼 데이터가 유효하면, 데이터는 cleaned_data로 복사
            new_name = form.cleaned_data['name']

            # 로직에 따라 추가적인 처리

            # 새로운 URL로 리다이렉션
            return HttpResponseRedirect('/thanks/')

    else:
        form = NameForm()

    return render(request, 'formtest/name.html', {'form': form})

def your_name(request):
    form = NameForm(request.POST)
    return render(request, 'formtest/your_name.html', {'form': form})