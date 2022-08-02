from flask import Flask, render_template,request, render_template, Response
from flask_bootstrap import Bootstrap
import os ,cv2
app = Flask(__name__)
Bootstrap(app)

@app.route("/recon",methods=['POST', 'GET'])
def recon():
    if request.method == 'POST':
        upload_img = request.files['recon_file']
        upload_img.save(os.path.join('./static/img', upload_img.filename))
        saved_path = os.path.join('./static/img', upload_img.filename)

        # response = Response(image, mimetype="image/jpeg")
        
        return render_template('index.html',image=saved_path)
    if request.method == 'GET':
        saved_path= ''
        return render_template('index.html',image=saved_path)

@app.route("/",methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        return  render_template('index.html')
    else:
        return  render_template('index.html')
        # user = request.args.get('user')
        # print("get : user => ", user)
        # return redirect(url_for('success', name=user, action="get"))

if __name__ == '__main__':
    app.run()