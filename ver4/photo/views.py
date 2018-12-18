from django.urls import reverse_lazy
from django.views.generic import ListView, DetailView
from django.views.generic import CreateView, UpdateView, DeleteView

from .models import Album, Photo


class AlbumLV(ListView):
    model = Album


class AlbumDV(DetailView):
    model = Album


class PhotoDV(DetailView):
    model = Photo


#--- Add/Change/Update/Delete for Photo
class PhotoCreateView(CreateView):
    model = Photo
    fields = ['album', 'title', 'image', 'data']
    success_url = reverse_lazy('photo:index')

    def form_valid(self, form):
        return super(PhotoCreateView, self).form_valid(form)


class PhotoChangeLV(ListView):
    template_name = 'photo/photo_change_list.html'

    def get_queryset(self):
        return Photo.objects.filter(owner=self.request.user)


class PhotoUpdateView(UpdateView) :
    model = Photo
    fields = ['album', 'title', 'image', 'data']
    success_url = reverse_lazy('photo:index')


class PhotoDeleteView(DeleteView) :
    model = Photo
    success_url = reverse_lazy('photo:index')


#--- Add/Change/Update/Delete for Album
#--- Change/Delete for Album
class AlbumChangeLV(ListView):
    template_name = 'photo/album_change_list.html'

    def get_queryset(self):
        return Album.objects.filter(owner=self.request.user)


class AlbumDeleteView(DeleteView) :
    model = Album
    success_url = reverse_lazy('photo:index')


#--- InlineFormSet View
#--- Add/Update for Album
from django.shortcuts import redirect, render
from .forms import PhotoInlineFormSet


class AlbumPhotoCV(CreateView):
    model = Album
    fields = ['name', 'description']

    def get_context_data(self, **kwargs):
        context = super(AlbumPhotoCV, self).get_context_data(**kwargs)
        if self.request.POST:
            context['formset'] = PhotoInlineFormSet(self.request.POST, self.request.FILES)
        else:
            context['formset'] = PhotoInlineFormSet()
        return context

    def form_valid(self, form):
        context = self.get_context_data()
        formset = context['formset']

        if formset.is_valid():
            self.object = form.save()
            formset.instance = self.object
            formset.save()
            return redirect('photo:album_detail', pk=self.object.id)
        else:
            return self.render_to_response(self.get_context_data(form=form))


class AlbumPhotoUV(UpdateView):
    model = Album
    fields = ['name', 'description']

    def get_context_data(self, **kwargs):
        context = super(AlbumPhotoUV, self).get_context_data(**kwargs)
        if self.request.POST:
            print('self.request.POST')
            context['formset'] = PhotoInlineFormSet(self.request.POST, self.request.FILES, instance=self.object)
            return context
        else:
            # context['formset'] = PhotoInlineFormSet(instance=self.object)
            myDict = dict(self.request.GET)
            print('self.request.GET')
            print('myDict:', myDict)
            active_list = ['img_id', 'img_target', 'img_alt', 'img_coords']
            f = open("tmp.txt", 'w')
            for i in range(len(myDict['img_id']) - 1):
                tmp_str = ""
                tmp_str += (myDict['img_id'][i] + " ")
                tmp_str += (myDict['img_target'][i] + " ")
                tmp_str += (myDict['img_alt'][i] + " ")
                tmp_str += (myDict['img_coords'][i] + " ")
                tmp_str += "\n"
                print(tmp_str)
                f.writelines(tmp_str)
            f.close()
            return redirect('p5')


    def form_valid(self, form):
        context = self.get_context_data()
        formset = context['formset']
        if formset.is_valid():
            print('form_valid')
            self.object = form.save()
            formset.instance = self.object
            formset.save()
            return redirect(self.object.get_absolute_url())
        else:
            print('form_invalid')
            return self.render_to_response(self.get_context_data(form=form))

# def p1(request):
#     if request.method == "POST":
#
#         # resize
#         # sp = Preprocessor()
#         # sp.load_original(request.FILES['survey'])   # 현재 디비에 저장되지 않은 상태의 이미지. 경로가 아니라 오류날 것 같음.
#         # resized_img = sp.survey_original
#         # newFile = MetaSurvey(survey=request.FILES['survey'], data=request.FILES['data'], resized_survey=resized_img)
#
#         newFile = MetaSurvey(survey=request.FILES['survey'], data=request.FILES['data'])
#         newFile.save()  # 데이터베이스에 저장. table 이름 MetaSurvey
#
#         my_file = newFile
#
#         # unzip된 파일들을 meta survey와 분석작업을 진행.
#
#         # list 'test_file' 출력해서 확인
#         # print('type: ', type(test_list))
#         # for test_file in test_list:
#         #     print(type(test_file), ', test_file: ', test_file)
#
#         # sp = Preprocessor(origianl_survey, [test_list])
#
#         # sp = Preprocessor()
#         # sp.load_original(origianl_survey)
#         # resized_img = sp.survey_original
#
#         # MetaSurvey.objects
#         # Model.objects.filter(id=param_dict['id']).update(**param_dict['Model'])
#
#         return render(request, 'part1/p1.html', {'my_file': my_file})
#     else:
#         myDict = dict(request.GET)
#         active_list = ['img_id', 'img_target', 'img_alt', 'img_coords']
#         f = open("tmp.txt", 'w')
#         for i in range(len(myDict['img_id']) - 1):
#             tmp_str = ""
#             tmp_str += (myDict['img_id'][i] + " ")
#             tmp_str += (myDict['img_target'][i] + " ")
#             tmp_str += (myDict['img_alt'][i] + " ")
#             tmp_str += (myDict['img_coords'][i] + " ")
#             tmp_str += "\n"
#             print(tmp_str)
#             f.writelines(tmp_str)
#         f.close()
#         return redirect('p5')
#
def p5(request):
    # app_root_path = r"C:\pyproject\capstone\ver2"

    # app_root_path = "../ver2"
    # meta = os.path.join(app_root_path, "tmp.txt")
    # temp = MetaSurvey.objects.order_by('-id')[0]
    # origianl_survey = temp.survey.path
    # test_survey = temp.data.path
    #
    # print('original_survey: ', origianl_survey)
    # print('test_survey: ', test_survey)

    # unzip

    # test_list = []
    # with zipfile.ZipFile(test_survey, 'r') as zip_ref:
    #     new_dir = test_survey + temp.data.__str__()
    #     new_dir = new_dir.split(".")[0]
    #     print('new_dir:', new_dir)
    #     if not os.path.exists(new_dir):
    #         os.makedirs(new_dir)
    #
    #     zip_ref.extractall(new_dir)
    #
    #     filenames = os.listdir(new_dir)
    #     for filename in filenames:
    #         filename = os.path.join(new_dir, filename)
    #         test_list.append(filename)
    #
    # # list 'test_file' 출력해서 확인
    # print('type: ', type(test_list))
    # for test_file in test_list:
    #     print(type(test_file), ', test_file: ', test_file)

    # sp = Preprocessor(meta, origianl_survey, [test_survey])

    # sp = Preprocessor(meta, origianl_survey, [test_list])
    # sp = Preprocessor(meta, origianl_survey, test_list)

    # ocr_json_path = os.path.join(app_root_path, "OCR_result.json")
    # ocr_ori = os.path.join(app_root_path, "ocr_ori.jpg")
    # detect_text(ocr_ori, ocr_json_path)
    # #sp.debug()
    # sp.load_original_survey_ocr(ocr_json_path)
    #
    # sp.displacement_fix()
    # csv_filename = os.path.join(app_root_path, "result_csv.csv")
    # sp.print_answers(0)
    # sp.make_csv(csv_filename)

    return render(request, 'photo/p5.html', {})