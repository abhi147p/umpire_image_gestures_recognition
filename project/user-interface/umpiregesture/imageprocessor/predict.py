# from imageprocessor.models import *
import pickle
import cv2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
class BaseModel:
    def _init_(self, model_path):
        # Load the model from the specified path
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        # Load a model from a file. Replace this with the actual loading code.
        # For instance, if you use TensorFlow or PyTorch, you would initialize your model here.
        pass

    def predict(self, image_path):
        # Load the image and make a prediction.
        # This method should be overridden by subclasses if the prediction logic differs.
        pass

    def process_image(self, image_path):
        # Process the image so it can be inputted to the model.
        # This might include resizing, normalizing, etc.
        pass
    

class Model2(BaseModel):
    def predict(self, image_path):
        image = self.process_image(image_path)
        return "Model2 predicts: " + str(self.model.predict(image))


class Model3(BaseModel):
    def predict(self, image_path):
        image = self.process_image(image_path)
        return "Model3 predicts: " + str(self.model.predict(image))


class Model4(BaseModel):
    def predict(self, image_path):
        image = self.process_image(image_path)
        return "Model4 predicts: " + str(self.model.predict(image))


class Model5(BaseModel):
    def predict(self, image_path):
        image = self.process_image(image_path)
        return "Model5 predicts: " + str(self.model.predict(image))

class Model1(BaseModel):
    def predict(self, image_path):
        image = self.process_image(image_path)
        # Assuming self.model.predict() is the method to make a prediction.
        return "Model1 predicts: " + str(self.model.predict(image))

ROLES = {
    0: 'wide',
    1: 'six',
    2: 'out',
    3: 'four',
    4: 'no_ball',
    5: 'byes',
    6: 'leg_byes',
    7: 'no_action',
    8: 'not_out'
}

classification_labels = {
    0 : "byes",
    1 : "four",
    2 : "leg byes",
    3 : "no action",
    4 : "no ball",
    5 : "not out",
    6 : "out",
    7 : "six",
    8 : "wide"
}

def predict_image_model1(image_path):
    model1 = get_alexNet_model_predict(image_path)
    return ROLES.get(model1, "no predict")

def predict_image_model2(image_path):
    model2 = get_KNN_model_predict(image_path)
    return classification_labels.get(model2, "no predict")

def predict_image_model3(image_path):
    model3 = get_RF_model_predict(image_path)
    return classification_labels.get(model3, "no predict")

def predict_image_model4(image_path):
    # model4 = Model4("/path/to/model4/file")
    return "NA"

# def predict_image_model5(image_path):
#     model5 = Model5("/path/to/model5/file")
#     return model5.predict(image_path)

import joblib
# from azureml.core import Workspace, Model
import sklearn
import keras
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Connect to your Azure ML workspace
# ws = Workspace.from_config()  # Assumes you have a config.json file in your working directory

def get_alexNet_model_predict(X_new):

    # Get the path to the registered model
    # model_path = Model.get_model_path('AlexNet', _workspace=ws)

    # Load the model
    model_path = "C:\\Users\\Kusha\\Desktop\\masters\\final_sem\\project\\user-interface\\models\\alexnetcrossmodel.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)    
    print(model)

    # Make predictions (replace X_new with your new data)
    # X_new = [[...], [...], ...]  # Your new data for prediction
    # img_array = cv2.imread(X_new, cv2.IMREAD_COLOR)
    # img_array = cv2.resize(img_array,(50,50))
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # print(np.array(img_array).shape)
    img_array = load_img(X_new, target_size=(50, 50))
    img_array = img_to_array(img_array) 
    # Add the batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 50, 50, 3)
    #print(f"Img: {img_array}")
    predictions = model.predict(img_array)

    # print(f"pred: {predictions}")
    return np.argmax(predictions)

# get_alexNet_model_predict("")

from skimage.feature import hog

def extract_hog_features(image):
    features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return features

def get_KNN_model_predict(X_new):

    # Get the path to the registered model
    # model_path = Model.get_model_path('AlexNet', _workspace=ws)

    # Load the model
    model_path = "C:\\Users\\Kusha\\Desktop\\masters\\final_sem\\project\\user-interface\\models\\grid_search_knn_1.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)    
    print(model)

    # Make predictions (replace X_new with your new data)
    # X_new = [[...], [...], ...]  # Your new data for prediction
    # img_array = cv2.imread(X_new, cv2.IMREAD_COLOR)
    # img_array = cv2.resize(img_array,(50,50))
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # print(np.array(img_array).shape)
    # img_array = load_img(X_new, target_size=(50, 50))
    # img_array = img_to_array(img_array) 
    # # Add the batch dimension
    # img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 50, 50, 3)
    # #print(f"Img: {img_array}")
    # print(img_array[0].shape)
    img = cv2.imread(X_new)
    img = cv2.resize(img, (256, 256))
    img_features = extract_hog_features(img)
    print(f"img features: {len(img_features)} ----> {img_features}")
    predictions = model.predict([img_features])
    print(f"pred: {predictions}")

    # print(f"pred: {predictions}")
    return predictions[0]

# get_KNN_model_predict("")

def get_RF_model_predict(X_new):

    # Get the path to the registered model
    # model_path = Model.get_model_path('AlexNet', _workspace=ws)

    # Load the model
    model_path = "C:\\Users\\Kusha\\Desktop\\masters\\final_sem\\project\\user-interface\\models\\grid_search_rf_1.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)    
    print(model)

    # Make predictions (replace X_new with your new data)
    # X_new = [[...], [...], ...]  # Your new data for prediction
    # img_array = cv2.imread(X_new, cv2.IMREAD_COLOR)
    # img_array = cv2.resize(img_array,(50,50))
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # print(np.array(img_array).shape)
    # img_array = load_img(X_new, target_size=(50, 50))
    # img_array = img_to_array(img_array) 
    # # Add the batch dimension
    # img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 50, 50, 3)
    # #print(f"Img: {img_array}")
    # predictions = model.predict(img_array)

    # # print(f"pred: {predictions}")
    # return np.argmax(predictions)

    img = cv2.imread(X_new)
    img = cv2.resize(img, (256, 256))
    img_features = extract_hog_features(img)
    print(f"img features: {len(img_features)} ----> {img_features}")
    predictions = model.predict([img_features])
    print(f"pred: {predictions}")

    # print(f"pred: {predictions}")
    return predictions[0]

# get_RF_model_predict("")

# from tensorflow.keras.models import load_model
def get_ResNet_model_predict(X_new):
    pass

    # Get the path to the registered model
    # model_path = Model.get_model_path('AlexNet', _workspace=ws)

    # Load the model
    # model_path = "C:\\Users\\Kusha\\Desktop\\masters\\final_sem\\project\\user-interface\\models\\resnet_model1.pkl"
    # # with open(model_path, 'rb') as file:
    # #     model = pickle.load(file)    
    # # print(model)
    # custom_objects = {'ResNet': ResNet1}
    # model = load_model(model_path, custom_objects=custom_objects)


    # Make predictions (replace X_new with your new data)
    # X_new = [[...], [...], ...]  # Your new data for prediction
    # img_array = cv2.imread(X_new, cv2.IMREAD_COLOR)
    # img_array = cv2.resize(img_array,(50,50))
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # print(np.array(img_array).shape)

    # img_array = load_img(X_new, target_size=(50, 50))
    # img_array = img_to_array(img_array) 
    # # Add the batch dimension
    # img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 50, 50, 3)
    # #print(f"Img: {img_array}")
    # predictions = model.predict(img_array)

    # # print(f"pred: {predictions}")
    # return np.argmax(predictions)

# get_ResNet_model_predict("")