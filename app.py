from distutils.log import error
from tkinter import E
from flask import Flask, render_template,request, render_template, Response ,redirect , url_for
import os ,cv2
from importlib import import_module
import subprocess
from yolov5_flask import Camera
camera = cv2.VideoCapture(0)
app = Flask(__name__)
INPUT_DIR="./static/img"
WEIGHTS = "./last.pt"
FILE_TYPES = set(['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp','asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'])
VIDEO_TYPES = set(['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'])
output_path = "./static/"
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
                        return render_template('video.html',image=saved_path,result='Success',output_video='./static/output/1.mp4')
                    else:
                        output=subprocess.check_output(f"python detect.py --weights {WEIGHTS} --source {upload_path} --conf-thres 0.5 --project {output_path} --name output --exist-ok")
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
        subprocess.run(f"python detect.py --weights {WEIGHTS} --source http://127.0.0.1:5000/local_feed --conf-thres 0.5 --project {output_path} --name output --exist-ok")
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
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/local_feed')
def local_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(local_gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def local_gen(camera):
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
def gen(objcamera):
    """Video streaming generator function."""
    while True:
        frame = objcamera.get_frame()
        # a = camera.people_appeal()
        # print('a:{}0'.format(a))
        # for i in a:
        #     if i =='people':
        #         print('是people：{}}')
        #         people_appeal()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run()