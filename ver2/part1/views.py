import os
import zipfile
import cv2

from django.shortcuts import render, get_object_or_404, redirect
from .models import MetaSurvey
from .forms import SurveyForm
from django.views.generic import View

from django.http import HttpResponseRedirect
from django.urls import reverse
from .my_module.Preprocessor import Preprocessor
from .my_module.my_ocr_module import detect_text

import cv2

def resizing_image(datapath):   #컴퓨터의 경로, 반환값은 이미지 데이터
    raise NotImplemented


# Create your views here.

# class MyFormView(View):
#     form_class = MyFormView
#     initial = {'key':'value'}
#     template_name = 'form_template.html'
#     # success_url = '/thanks/'
#
#     def get(self, request, *args, **kwargs):
#         form = self.form_class(initital=self.initial)
#         return render(request, self.template_name, {'form': form})
#
#     def post(self, request, *args, **kwargs):
#         form = self.form_class(request.POST)
#         if form.is_valid():
#             return HttpResponseRedirect('/success/')
#
#         return render(request, self.template_name, {'form':form})




def p0(request):
    form = SurveyForm()
    return render(request, 'part1/p0.html', {'form': form})

def p1(request):
    if request.method == "POST":
        newFile = MetaSurvey(title= request.POST['title'], survey=request.FILES['survey'], data=request.FILES['data'], resized_survey=request.FILES['survey'])
        newFile.save()  # 데이터베이스에 저장. table 이름 MetaSurvey

        my_file = newFile
        # print(my_file.pk)

        # if form.is_valid():
        #     print("form: ", form.cleaned_data)
        #     # form.save()
        #     my_file = form
        # else:
        #     print('invalid')

        # resize는 이제 fields.py에서
        # sp = Preprocessor()
        # sp.load_original(request.FILES['survey'])   # 현재 디비에 저장되지 않은 상태의 이미지. 경로가 아니라 오류날 것 같음.
        # resized_img = sp.survey_original
        # newFile = MetaSurvey(survey=request.FILES['survey'], data=request.FILES['data'], resized_survey=resized_img)

        # newFile = MetaSurvey(survey=request.FILES['survey'], data=request.FILES['data'])
        # newFile.save()  # 데이터베이스에 저장. table 이름 MetaSurvey

        # my_file = newFile

        # unzip된 파일들을 meta survey와 분석작업을 진행.

        # list 'test_file' 출력해서 확인
        # print('type: ', type(test_list))
        # for test_file in test_list:
        #     print(type(test_file), ', test_file: ', test_file)

        # sp = Preprocessor(origianl_survey, [test_list])

        # sp = Preprocessor()
        # sp.load_original(origianl_survey)
        # resized_img = sp.survey_original

        # MetaSurvey.objects
        # Model.objects.filter(id=param_dict['id']).update(**param_dict['Model'])

        return render(request, 'part1/p1.html', {'my_file': my_file})

    else:
        myDict = dict(request.GET)
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



# def p2(request, pk):
def p2(request):
    #     meta = get_object_or_404(MetaSurvey, pk)
    if request.method == 'POST':
        form = SurveyForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = DataSurvey(docfile = request.FILES['survey'])
            newdoc.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('p2'))
    else:
        form = SurveyForm() # A empty, unbound form

    # Load documents for the list page
    documents = DataSurvey.objects.all()

    # Render list page with the documents and the form
    # return render(request, 'part1/p2.html', {'meta': meta, 'documents': documents, 'form': form})
    return render(request, 'part1/p2.html', {'documents': documents, 'form': form})
    # docs = MetaSurvey.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    # doc = get_object_or_404(docs, pk=pk)
    # return render(request, 'part1/p2_upload.html', {'docs': docs})
    # return render(request, 'part1/p2.html')

def p4(request):
    newFile = MetaSurvey(survey=request.FILES['survey'])
    newFile.save()

    # Load documents for the list page
    img = newFile
    # img = MetaSurvey.objects.all()[0]
    # img = MetaSurvey.objects.order_by('pk')[0]

    return render(request, 'part1/p4.html', {'img': img })

# def p5(request, pk):
#     post = get_object_or_404(MetaSurvey, pk=pk)
#     # return redirect(reverse('part1/p5.html', kwargs={'pk': post.id}))
#     return render(request, 'part1/p5.html', { 'post': post })

def p5(request):
    # app_root_path = r"C:\pyproject\capstone\ver2"
    app_root_path = "../ver2"
    meta = os.path.join(app_root_path, "tmp.txt")
    temp = MetaSurvey.objects.order_by('-id')[0]
    origianl_survey = temp.survey.path
    test_survey = temp.data.path

    print('original_survey: ', origianl_survey)
    print('test_survey: ', test_survey)

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

    test_img = cv2.imread(test_survey)
    cv2.imshow('test', test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    sp = Preprocessor(origianl_survey, [test_survey], meta)
    # sp = Preprocessor(meta, origianl_survey, [test_list])
    # sp = Preprocessor(meta, origianl_survey, test_list)

    ocr_json_path = os.path.join(app_root_path, "OCR_result.json")
    ocr_ori = os.path.join(app_root_path, "ocr_ori.jpg")
    detect_text(ocr_ori, ocr_json_path)
    #sp.debug()
    sp.load_original_survey_ocr(ocr_json_path)

    sp.displacement_fix()
    csv_filename = os.path.join(app_root_path, "result_csv.csv")
    sp.print_answers(0)
    sp.make_csv(csv_filename)
    return render(request, 'part1/p5.html', {})
