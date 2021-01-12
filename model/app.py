import os
from flask import Flask, request, jsonify
from model.main import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import uuid

UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/face-detect', methods=['POST'])
def face_detect():
    if 'file_selfie' not in request.files or 'file_id' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    file_selfie = request.files['file_selfie']
    file_id = request.files['file_id']
    threshold = request.form.get("threshold", 0.4)

    if file_selfie.filename == '' or file_id.filename == '':
        resp = jsonify({'message': 'No file selected for uploading'})
        resp.status_code = 400
        return resp

    if allowed_file(file_selfie.filename) and allowed_file(file_id.filename):
        fn_selfie = str(uuid.uuid4()) + file_selfie.filename
        fn_id = str(uuid.uuid4()) + file_id.filename
        path_selfie = os.path.join(app.config["UPLOAD_FOLDER"], fn_selfie)
        path_id = os.path.join(app.config["UPLOAD_FOLDER"], fn_id)

        path_selfie2 = os.path.join(app.config["OUTPUT_FOLDER"], fn_selfie)
        path_id2 = os.path.join(app.config["OUTPUT_FOLDER"], fn_id)
        file_selfie.save(path_selfie)
        file_id.save(path_id)
        face_alignment(path_selfie, path_selfie2, 1)
        face_alignment(path_id, path_id2, 0)

        output, preds = compare_face(path_id2, path_selfie2, float(threshold))

        os.remove(path_selfie)
        os.remove(path_id)
        os.remove(path_selfie2)
        os.remove(path_id2)
        return jsonify({'message': 'Success', 'data': {'output': str(output), 'preds': str(preds)}})
    else:
        resp = jsonify({'message': 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)
