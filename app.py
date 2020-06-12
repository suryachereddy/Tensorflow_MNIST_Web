 

from flask import Flask, render_template, request,url_for
from PIL import Image
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import cv2
import base64
import sys 
import os
from werkzeug.utils import secure_filename
import re
from io import StringIO,BytesIO 
#import matplotlib.pyplot as plt


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded / image'
model = tf.keras.models.load_model('model') 

def pred(x): 
#x=tf.cast(x, tf.float32)
   # x=cv2.imread(x,cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, dsize=(28, 28), interpolation=cv2.INTER_CUBIC).astype(float)
    x  = x[:, :, 0]
    x/=255

    
    x =1-x
    #print(x)
    #print(x.shape)
    #plt.imshow(x, cmap=plt.cm.binary)
    #plt.show()
    x = x.reshape((1,28,28,1))  
  
    pred = model.predict(x) 
    print(np.argmax(pred)) 
    return np.argmax(pred)


@app.route("/",methods=["POST","GET"])
def index():
    val=None
    if request.method=="POST":
        r = request.values["imageBase64"]
        
        img_bytes = base64.b64decode(r)
        img = Image.open(BytesIO(img_bytes))
        img  = np.array(img)
        

        #img=np.fromstring(request.values["imageBase64"], np.uint8)
        #print(img.shape)
        #nparr = np.fromstring(r.data["imageBase64"], np.uint8)
        #print(nparr.shape)
        

        val=pred(img)
        #print(val)
        return str(val)
    return render_template("index.html")
