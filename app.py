
from flask import Flask, request, render_template, url_for
import os, uuid
from flask import jsonify
import numpy as np
import cv2



app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file!", 400

        ext = file.filename.rsplit(".",1)[-1]
        filename = f"{uuid.uuid4()}.{ext}"
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img_url = url_for("static", filename=f"uploads/{filename}")
        return render_template("upload.html", img_url=img_url)

    return render_template("upload.html", img_url=None)

@app.route("/api/upload", methods=["POST"])
def api_upload():

    # debug info
    print("Received request to /api/upload")
    print("content-type:", request.content_type)
    print("content type",request.headers.get("Content-Type"))
    print("content length:", request.content_length)
    print("form keys:", request.form.keys())
    print("files keys:", request.files.keys())

    file = request.files.get("file") or request.files.get("image")
    if not file:
        # try to fetch from any file present
        files_list = list(request.files.values())
        file = files_list[0] if files_list else None

    if not file:
        # last resort - print first 200 bytes of the request data
        data = request.data[:200]
        print("Raw request start (turnicated):", data)
        return jsonify({"error": "No file received on server", "debug_files": list(request.files.keys())}), 400

    ext = file.filename.rsplit(".", 1)[-1] if '.' in file.filename else 'jpg'
    filename = f"{uuid.uuid4()}.{ext}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # Return the URL of the uploaded image so it can be displayed (web page can link to it)
    img_url = url_for("static", filename=f"uploads/{filename}", _external=True)
    return jsonify({"img_url": img_url}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
