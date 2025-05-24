from flask import Blueprint, render_template


ui = Blueprint("ui", __name__)

@ui.route('/')
def index():
    return render_template("index.html")
