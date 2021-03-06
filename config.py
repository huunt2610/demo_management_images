import os 
'''
Global arguments
'''
# maximum filesize in megabytes
file_mb_max = 100
# encryption key
app_key = 'test'
# full path destination for our upload files
upload_dest = os.path.join(os.getcwd(), 'uploads_folder')
# list of allowed allowed extensions
extensions = set(['txt', 'pdf', 'image/png', 'image/jpg', 'image/tiff','image/gtiff', 'image/jpeg'])
#text/html 
db_loc = os.path.join(os.getcwd(), 'user_cred.db')
# full path destination for extracted feature
feature_storage = os.path.join(os.getcwd(), "features_folder")
