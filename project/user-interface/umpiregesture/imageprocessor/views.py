from django.shortcuts import render, redirect
from .models import Image
from .forms import ImageForm
from .predict import *
from django.urls import reverse


def image_upload_view(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()
            # predictions = {
            #     'Model1': predict_image_model1(image_instance.picture.path),
            #     'Model2': predict_image_model2(image_instance.picture.path),
            #     'Model3': predict_image_model3(image_instance.picture.path),
            #     'Model4': predict_image_model4(image_instance.picture.path),
            #     'Model5': predict_image_model5(image_instance.picture.path)
            # }
            # predictions = {
            #     'Model1': "leg byes",
            #     'Model2': "byes",
            #     'Model3': "six",
            #     'Model4': "four"
            # }
            return redirect(reverse('display_predictions', args=[image_instance.id]))
    else:
        form = ImageForm()
    return render(request, 'image_form.html', {'form': form})

# def F(request):
#     form = ImageForm()
#     return render(request,  'image_form.html', {'form': form})

def display_predictions(request, image_id):
    # from .models import ImageModel
    # predictions = {
    #     'Model1': predict_image_model1(image_instance.picture.path),
    #     'Model2': predict_image_model2(image_instance.picture.path),
    #     'Model3': predict_image_model3(image_instance.picture.path),
    #     'Model4': predict_image_model4(image_instance.picture.path),
    #     'Model5': predict_image_model5(image_instance.picture.path)
    # }
    image_instance = Image.objects.get(id=image_id)
    predictions = {
        'AlexNet': predict_image_model1(image_instance.picture.path),
        'KNN': predict_image_model2(image_instance.picture.path),
        'RF': predict_image_model3(image_instance.picture.path),
        'Model4': "NA",
    }
    # predictions = request.session.get('predictions', {})
    return render(request, 'display_predictions.html', {
        'image_instance': image_instance,
        'predictions': predictions
    })