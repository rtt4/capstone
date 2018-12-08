from django.contrib import admin
from .models import MetaSurvey, Question, Answer

# Register your models here.
admin.site.register(MetaSurvey)
admin.site.register(Question)
admin.site.register(Answer)