import os
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'C:\Users\AKSHAYA\Pictures\static\images'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('recognise.html')


model = tf.keras.models.load_model("digit_classifier.h5")

@app.route('/predict', methods = ['GET','POST'])
def upload_image_file():
    if request.method == 'POST':
        imagefile = request.files['image']
        filename = secure_filename(imagefile.filename)
        imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        path_img = os.path.join(UPLOAD_FOLDER, filename)
        img = Image.open(path_img).convert("L")
        img = img.resize((28,28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,1)
        y_pred = model.predict(im2arr)
        
        return render_template('predict.html', num = str(y_pred))

if __name__ == '__main__' :
    app.run(host='0.0.0.0', port=8000, debug=True)
