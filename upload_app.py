import glob
from flask import Flask, flash, request, redirect, render_template
from config import *

app = Flask(__name__,
            static_url_path='',  # removes path prefix requirement */templates/static
            static_folder='templates/static',  # static file location
            template_folder='templates'  # template file location
            )
app.secret_key = app_key
app.config['MAX_CONTENT_LENGTH'] = file_mb_max * 1024 * 1024


# Check that the upload folder exists
def makedir(dest):
    fullpath = '%s/%s' % (upload_dest, dest)
    if not os.path.isdir(fullpath):
        os.mkdir(fullpath)


makedir('')  # make uploads folder


# on page load display the upload file
@app.route('/upload')
def upload_form():
    flash('Drag files to upload here.')
    return render_template('upload.html')


# on a POST request of data
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        emp_code = str(request.form['code'])
        emp_name = str(request.form['name'])
        id_file = str(request.form['id_file'])
        allfiles = request.files

        if 'files[]' not in allfiles:
            flash('No files found, try again.')
            return redirect(request.url)

        files = allfiles.getlist('files[]')
        print(len(files))
        for file in files:
            print(file)
            if file.mimetype in extensions:
                if emp_code is None or emp_code == '':
                    filename = "temp_%s.%s" % (id_file, file.filename.split(".")[-1])
                else:
                    filename = "%s_%s.%s" % (emp_code, id_file, file.filename.split(".")[-1])
                file.save(os.path.join(upload_dest, filename))
            else:
                print('Not allowed', file)

        flash('File(s) uploaded')
        return redirect('/upload')


# what have I updated? Return a list of updated files
@app.route('/uploaded/<emp_code>', methods=['GET', 'POST'])
def data_get(emp_code):
    if request.method == 'POST':  # POST request
        print(request.get_text())  # parse as text
        return 'OK', 200

    else:  # GET request
        files = glob.glob('%s/%s*' % (upload_dest, emp_code))
        return ','.join([i.rsplit('/', 1)[1] for i in files])


if __name__ == "__main__":
    print('to upload files navigate to http://127.0.0.1:4000/upload')
    # lets run this on localhost port 4000
    app.run(host='127.0.0.1', port=4000, debug=True, threaded=True)
