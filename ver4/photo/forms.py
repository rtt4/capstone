from .models import Album, Photo
from django.forms.models import inlineformset_factory

# 폼셋: 동일한 폼 여러 개로 구성된 폼.
# 인라인 폼셋, 메인 폼에 딸려 있는 하위 폼셋으로,
# 1:N 관계인 테이블 관계에서 N 테이블의 레코드 여러 개를 한꺼번에 입력 받기 위한 폼.

PhotoInlineFormSet = inlineformset_factory(Album, Photo,
    fields = ['image', 'title', 'description'],
    extra = 2)
