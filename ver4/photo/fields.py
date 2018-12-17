import os

from django.db.models.fields.files import ImageField, ImageFieldFile
from PIL import Image

# 212페이지
def _add_thumb(s):  # 기존 파일명을 기준으로 썸네일 이미지 파일명 만든다.
    parts = s.split(".")
    parts.insert(-1, "thumb")
    if parts[-1].lower() not in ['jpeg', 'jpg']:
        parts[-1] = 'jpg'
    return ".".join(parts)


class ThumbnailImageFieldFile(ImageFieldFile):  # 파일 시스템에 직접 파일을 쓰고 지우는 작업을 한다.

    # 이미지 처리하는 필드는 path와  url 속성을 제공해야한다.
    def _get_thumb_path(self):
        return _add_thumb(self.path)
    thumb_path = property(_get_thumb_path)
    
    def _get_thumb_url(self):
        return _add_thumb(self.url)
    thumb_url = property(_get_thumb_url)

    # 파일 시스템에 파일을 저장하고 생성하는 메소드.
    def save(self, name, content, save=True):
        super(ThumbnailImageFieldFile, self).save(name, content, save)  # 부모 ImageFieldFile 클래스의 save() 호출해서 원본 이미지 저장.
        img = Image.open(self.path)

        size = (128, 128)
        img.thumbnail(size, Image.ANTIALIAS)
        # background = Image.new('RGBA', size, (255, 255, 255, 0))
        # background.paste(
        #     img, ( int((size[0] - img.size[0]) / 2), int((size[1] - img.size[1]) / 2) ) )
        # background.save(self.thumb_path, 'JPEG')    # 합쳐진 최종 이미지를 JPEG 형식으로 thumb_path에 저장.
        img.save(self.thumb_path, 'JPEG')

    # 원본, 썸네일 이미지 같이 삭제되도록 함.
    def delete(self, save=True):
        if os.path.exists(self.thumb_path):
            os.remove(self.thumb_path)
        super(ThumbnailImageFieldFile, self).delete(save)


class ThumbnailImageField(ImageField):  # 장고 모델 정의에 사용되는 필드 역할.
    attr_class = ThumbnailImageFieldFile    # 새로운 FileField 클래스를 정의할 때 그에 상응하는 File 처리 클래스를 attr_class 속성에 지정하는 것이 필수.(ThumbnailImageFieldFile)

    def __init__(self, thumb_width=128, thumb_height=128, *args, **kwargs):
        self.thumb_width = thumb_width
        self.thumb_height = thumb_height
        super(ThumbnailImageField, self).__init__(*args, **kwargs)  # 부모 ImageField 클래스의 생성자 호출, 관련 속성 초기화.
