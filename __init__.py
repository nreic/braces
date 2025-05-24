# initialize the flask app and register blueprints

from flask import Flask


def create_app():
    app = Flask(__name__, template_folder='ui/templates', static_folder='ui/static')
    app.config['UPLOAD_FOLDER'] = 'ui/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # max 8 MB

    from ui import ui as ui_blueprint
    app.register_blueprint(ui_blueprint)

    from core import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')
    return app
