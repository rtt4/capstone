import zipfile
import os
from django.conf import settings
from django.core.files import File
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .models import MetaSurvey
from .forms import SurveyForm
from django.http import Http404
from .my_module.Preprocessor import Preprocessor
from .my_module.my_ocr_module import detect_text

# Create your views here.
def p0(request):
    form = SurveyForm()
    return render(request, 'part1/p0.html', {'form': form})

def p1(request):
    if request.method == "POST":
        # newFile = MetaSurvey(title= request.POST['title'], survey=request.FILES['survey'], data=request.FILES['data'], resized_survey=request.FILES['survey'])

        #1 save in-memory file
        survey_path = os.path.join(settings.MEDIA_ROOT, "survey")
        f = request.FILES['survey']
        print('type(f): ', type(f))

        #2 after saving, parsing the ~resized.jpeg file
        form = MetaSurvey(title=request.POST['title'], survey=request.FILES['survey'], data=request.FILES['data'])
        form.save() # 데이터베이스에 저장한 뒤에야 ~resized.jpeg 파일이 생성된다. => 디비에 따로 저장해야 p1에서 사용.
        print('pk: ', form.pk)

        print(form.survey.name.split("/"))
        survey_file_name = form.survey.name.split("/")[-1]

        file_path = survey_file_name.split(".")
        file_path.insert(-1, "resize")
        file_path = ".".join(file_path)

        print('survey_path: ', survey_path)
        # print('file_path, before: ', file_path)
        file_path = os.path.join(survey_path, file_path)
        print('file_path, after: ', file_path)

        f = open(file_path, 'rb')
        print('size of f: ', os.path.getsize(file_path))
        djangofile = File(f)
        form.resized_survey.save('resized_survey.jpeg', djangofile)
        f.close()
        print('form type: ', type(form))

        context = {'my_file': form,}
        return render(request, 'part1/p1.html', context)
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

def p5(request):
    # app_root_path = r"C:\pyproject\capstone\ver2"
    app_root_path = "../ver2"
    meta = os.path.join(app_root_path, "tmp.txt")
    temp = MetaSurvey.objects.order_by('-id')[0]
    origianl_survey = temp.survey.path
    test_survey = temp.data.path

    # unzip
    test_list = []
    with zipfile.ZipFile(test_survey, 'r') as zip_ref:
        new_dir = test_survey + temp.data.__str__()
        new_dir = new_dir.split(".")[0]
        print('new_dir:', new_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        zip_ref.extractall(new_dir)

        filenames = os.listdir(new_dir)
        for filename in filenames:
            filename = os.path.join(new_dir, filename)
            test_list.append(filename)

    sp = Preprocessor(origianl_survey, test_list, meta)

    ocr_json_path = os.path.join(app_root_path, "OCR_result.json")
    ocr_ori = os.path.join(app_root_path, "ocr_ori.jpg")
    detect_text(ocr_ori, ocr_json_path)
    sp.load_original_survey_ocr(ocr_json_path)

    sp.displacement_fix()
    csv_filename = os.path.join(app_root_path, "result_csv.csv")
    # sp.print_answers(0)
    sp.make_csv(csv_filename)

    return render(request, 'part1/p5.html', {'csv_path':csv_filename})

def download(request):
    path = "../ver2/result_csv.csv"
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(path)
            return response
    raise Http404