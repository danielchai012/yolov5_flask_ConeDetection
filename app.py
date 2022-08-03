from distutils.log import error
from tkinter import E
from flask import Flask, render_template,request, render_template, Response ,redirect , url_for
from flask_bootstrap import Bootstrap
import os ,cv2
from importlib import import_module
import subprocess
camera = cv2.VideoCapture(0)
app = Flask(__name__)
Bootstrap(app)
INPUT_DIR="C:/Users/ITM_Student_11/Desktop/yolov5/static/img"
WEIGHTS = "C:/Users/ITM_Student_11/Desktop/yolov5/runs/train/exp12/weights/last.pt"
FILE_TYPES = set(['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp','asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'])
VIDEO_TYPES = set(['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'])
output_path = "C:/Users/ITM_Student_11/Desktop/yolov5/static/"
@app.route("/recon",methods=['POST', 'GET'])
def recon():
    try:
        if request.method == 'POST':
            print(request.files.getlist)
            try:
            
                upload_img = request.files['recon_file']
                img_exts = f"{upload_img.filename.rsplit('.', 1)[1]}"
                newfilename = f'1.{img_exts}'
                if '.' in upload_img.filename and upload_img.filename.rsplit('.', 1)[1] in FILE_TYPES:
                    upload_path = os.path.join(INPUT_DIR, newfilename)
                    upload_img.save(os.path.join(INPUT_DIR, newfilename))
                    saved_path = os.path.join("./static/img", newfilename)
                    print(saved_path)
                    # response = Response(image, mimetype="image/jpeg")
                    
                    # return redirect(url_for('recon', image=saved_path,output_img=output_file))
                    if '.' in newfilename and newfilename.rsplit('.', 1)[1] in VIDEO_TYPES:
                        subprocess.run(f"python detect.py --weights {WEIGHTS} --source {upload_path} --conf-thres 0.5 --project {output_path} --name output --exist-ok")
                        return render_template('video.html')
                    else:
                        subprocess.run(f"python detect.py --weights {WEIGHTS} --source {upload_path} --conf-thres 0.5 --project {output_path} --name output --exist-ok")
                        output_file = './static/output/'+newfilename
                        return render_template('index.html',image=saved_path,output_img=output_file,result='Success',output_video='./static/output/1.mp4')
                else:
                    return render_template('index.html',result='file type not support')

            except Exception as e:
                print(e)    
                return render_template('index.html',image='')
        if request.method == 'GET':
            saved_path= ''
            return render_template('index.html',image=saved_path,output_video='./static/2.mp4')
    except Exception as e:
        print(e)

@app.route("/webcam",methods=['GET'])
def open_webcam():
    try:
        subprocess.run(f"python detect.py --weights {WEIGHTS} --source 0 --conf-thres 0.5 --project {output_path} --name output --exist-ok")
        return redirect(url_for('recon'))
    except:
        return redirect(url_for('recon'))
@app.route("/",methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        return  render_template('index.html')
    else:
        return  render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    """Video streaming generator function."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == '__main__':
    app.run()