from distutils.log import error
from tkinter import E
from flask import Flask, render_template,request, render_template, Response ,redirect , url_for
from flask_bootstrap import Bootstrap
import os ,cv2
import subprocess
app = Flask(__name__)
Bootstrap(app)
INPUT_DIR="C:/Users/ITM_Student_11/Desktop/yolov5/static/img"
WEIGHTS = "C:/Users/ITM_Student_11/Desktop/yolov5/runs/train/exp12/weights/last.pt"
FILE_TYPES = set(['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp','asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'])
output_path = "C:/Users/ITM_Student_11/Desktop/yolov5/static/"
@app.route("/recon",methods=['POST', 'GET'])
def recon():
    try:
        if request.method == 'POST':
            print(request.files.getlist)
            try:
                if (request.files['recon_file'] != None):
                    upload_img = request.files['recon_file']
                    img_exts = f"{upload_img.filename.rsplit('.', 1)[1]}"
                    newfilename = f'1.{img_exts}'
                    if '.' in upload_img.filename and upload_img.filename.rsplit('.', 1)[1] in FILE_TYPES:
                        upload_path = os.path.join(INPUT_DIR, newfilename)
                        upload_img.save(os.path.join(INPUT_DIR, newfilename))
                        saved_path = os.path.join("./static/img", newfilename)
                        print(saved_path)
                        # print(subprocess.run("ls"))
                        subprocess.run(f"python detect.py --weights {WEIGHTS} --source {upload_path} --conf-thres 0.5 --project {output_path} --name output --exist-ok")
                        output_file = './static/output/'+newfilename
                        # response = Response(image, mimetype="image/jpeg")
                        print(output_file)
                        # return redirect(url_for('recon', image=saved_path,output_img=output_file))
                        return render_template('index.html',image=saved_path,output_img=output_file)
                    else:
                        return render_template('index.html',result='file type not support')
                else:
                    return render_template('index.html',image='')
            except Exception as e:
                print(e)    
                return render_template('index.html',image='')
        if request.method == 'GET':
            saved_path= ''
            return render_template('index.html',image=saved_path)
    except Exception as e:
        print(e)

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