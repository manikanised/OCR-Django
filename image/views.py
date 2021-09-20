from django.shortcuts import render
from .form import ImageForm
from .models import Image
from django.conf import settings
from .OCR import conv
# Create your views here.

def index(request):
    if request.method=='POST':
        form=ImageForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            obj=form.instance
            print(str(settings.BASE_DIR)+obj.image.url)
            stri=conv(str(settings.BASE_DIR)+obj.image.url, str(settings.BASE_DIR)+'\image\lstm-weights-epoch24-val_loss0.242.h5')
            return render(request,'index.html',{"obj":obj,'stri':stri})
    else:
        form=ImageForm()
        img=Image.objects.all()
    return render(request,"index.html",{'img':img,'form':form})