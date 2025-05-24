import json
import numpy as np
from flask import Blueprint, jsonify, request, current_app, \
    send_from_directory
from werkzeug.utils import secure_filename
import os
from PIL import Image

from core.orthodontist import Orthodontist

api = Blueprint('api', __name__)

@api.route('/process-image', methods=['POST'])
def process_image():
    image_path = ""
    if 'image' in request.files:
        image = request.files['image']
        if image:
            filename = secure_filename(image.filename)
            folder = os.path.splitext(filename)[0]  # folder is name of the image for now
            image_path = save_image(image, filename, folder)

    elif 'filename' in request.form:
        filename = request.form['filename']
        filename = filename.split('?')[0] # remove timestamp
        filename = filename.split('processed_')[-1] # remove prefix to get original image
        folder = os.path.splitext(filename)[0]
        image_path = os.path.join(current_app.root_path,
                                  current_app.config['UPLOAD_FOLDER'],
                                  folder,
                                  filename)
    else:
        return jsonify({'status': 'error',
                        'message': "No image provided"}), 400

    if os.path.isfile(image_path):
        new_config = False
        if 'config' in request.form:
            new_config = request.form['config']
            new_config = json.loads(new_config)

        # here is where the magic happens:
        processed_image, used_config, segmentation_interims = do_braces_on_image(image_path, new_config)
        # todo: handle when something goes wrong! If we don't get image back or so.

        # handle processed image
        folder = os.path.splitext(filename)[0]
        processed_filename = 'processed_' + filename
        save_image(processed_image, processed_filename, folder)
        processed_image_url = f"/api/uploads/{folder}/{processed_filename}"  # endpoint to serve images

        # handle interims / intermediate results, e.g. 'teeth_area', 'clahe'
        file_urls = {}
        for key in segmentation_interims:
            interim_filename = key + '_' + filename
            save_image(segmentation_interims[key], interim_filename, folder)
            interim_image_url = f"api/uploads/{folder}/{interim_filename}"
            file_urls[key] = interim_image_url

        return jsonify({'status': 'success',
                        'message': 'Image from user was processed',
                        'processedImageUrl': processed_image_url,
                        'segmentationConfig': used_config,
                        'segmentationInterims': file_urls})
    return jsonify({'status': 'error',
                    'message': "Something went wrong"}), 400

def save_image(image, filename, folder):
    path = os.path.join(current_app.root_path,
                        current_app.config['UPLOAD_FOLDER'],
                        folder)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path,
                        filename)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if isinstance(image, Image.Image) and image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(path)
    return path

def do_braces_on_image(image_path, new_config):
    ortho = Orthodontist()
    if new_config:
        ortho.change_config(new_config)

    processed_image, used_config, segmentation_interims = ortho.fit_braces(image_path)
    return processed_image, used_config, segmentation_interims

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # Security note: verify file header to confirm file type, not just file extension
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# serve images
@api.route('/uploads/<folder>/<filename>')
def uploaded_file(folder, filename):
    return send_from_directory(os.path.join(current_app.config['UPLOAD_FOLDER'], folder), filename)

# error handling
@api.errorhandler(413)
def too_large(error):
    response = jsonify({'status': 'error',
                        'message': "Image is too large. Max. 8 MB allowed."})
    response.status_code = 413
    return response

@api.errorhandler(500)
def handle_500(error):
    response = jsonify({'status': 'error', 'message': 'Server Error'})
    response.status_code = 500
    return response