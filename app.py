import os
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'memes'
img_path = None
cap = None
pred = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    global img_path
    global cap
    if request.method == 'POST':
        f = request.files['image']
        img_path = os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'], f.filename)
        f.save(img_path)
        cap = request.form.get('caption')

        return redirect('/prediction')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    global pred
    # pred = get_pred(img_path, cap)
    return render_template('prediction.html', pred=1)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    fname = request.form.get('fname')
    lname = request.form.get('lname')
    email = request.form.get('email')
    reason = request.form.get('reason')

    ## save these in database with ~pred and img_path

    return redirect('/')


if __name__ == '__main__':
    #from werkzeug.serving import run_simple
    #run_simple('localhost', 5000, app)
    app.run(debug=True)