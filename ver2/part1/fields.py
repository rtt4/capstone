import os

from django.core.files import File
from django.db.models.fields.files import ImageField, ImageFieldFile

from .my_module.Preprocessor import Preprocessor
import cv2

# https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
import scipy.misc

def _add_thumb(s):
    parts = s.split(".")
    parts.insert(-1, "resize")
    if parts[-1].lower() not in ['jpeg', 'jpg']:
        parts[-1] = 'jpg'
    return ".".join(parts)

class ThumbnailImageFieldFile(ImageFieldFile):
    def _get_thumb_path(self):
        return _add_thumb(self.path)
    thumb_path = property(_get_thumb_path)

    def _get_thumb_url(self):
        return _add_thumb(self.url)
    thumb_url = property(_get_thumb_url)

    def save(self, name, content, save=True):
        super(ThumbnailImageFieldFile, self).save(name, content, save)
        print('save in fields')

        # resize
        sp = Preprocessor()
        sp.load_original(self.path)
        resized_img = sp.survey_original
        cv2.imwrite(self.thumb_path, resized_img)
        print('type: ', type(resized_img))  # <class 'numpy.ndarray'>

        scipy.misc.imsave(self.thumb_path, resized_img)

    def delete(self, save=True):
        if os.path.exists(self.thumb_path):
            os.remove(self.thumb_path)
        super(ThumbnailImageFieldFile, self).delete(save)


class ThumbnailImageField(ImageField):
    attr_class = ThumbnailImageFieldFile

    # 모델 필드 정의 시, 옵션 지정.
    def __init__(self, *args, **kwargs):
        super(ThumbnailImageField, self).__init__(*args, **kwargs)
