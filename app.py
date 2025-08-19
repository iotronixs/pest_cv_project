
from flask import Flask, request, render_template, url_for
import os, uuid
from flask import jsonify
import numpy as np
import cv2
import base64




app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
TMP_FOLDER = "tmp_parts"
os.makedirs(TMP_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_bytes_and_url(data_bytes, ext="jpg"):
    """
    Save bytes to a file and return the URL and saved path
    """
    fname = f"{uuid.uuid4()}.{ext}"
    fpath = os.path.join(UPLOAD_FOLDER, fname)
    with open(fpath, "wb") as f:
        f.write(data_bytes)
    
    return url_for("static", filename=f"uploads/{fname}", _external=True), fpath
    

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

@app.route("/api/upload_b64_chunk", methods=["POST"])
def upload_b64_chunk():
    """
    Query params required:
      id   -> unique upload id (string)
      idx  -> 1-based chunk index (int)
      total-> total number of chunks (int)
    POST body: the base64 text chunk (raw text) -- Web1.PostText body
    """
    upload_id = request.args.get("id", None)
    try:
        idx = int(request.args.get("idx", "0"))
        total = int(request.args.get("total", "0"))
    except ValueError:
        return jsonify({"error": "bad idx/total"}), 400

    if not upload_id or idx <= 0 or total <= 0:
        return jsonify({"error": "missing id/idx/total"}), 400

    # read raw body as text (works regardless of content-type)
    chunk_text = request.get_data(as_text=True)
    if chunk_text is None:
        return jsonify({"error": "empty chunk"}), 400

    part_path = os.path.join(TMP_FOLDER, f"{upload_id}.part")
    # append the text chunk (we assume client sends chunks in correct order)
    try:
        with open(part_path, "a", encoding="utf-8") as f:
            f.write(chunk_text)
    except Exception as e:
        return jsonify({"error": "failed to save chunk", "err": str(e)}), 500

    # If last chunk, assemble and decode
    if idx == total:
        try:
            with open(part_path, "r", encoding="utf-8") as f:
                whole_b64 = f.read()
            # optional: remove whitespace/newlines
            whole_b64 = "".join(whole_b64.split())
            data = base64.b64decode(whole_b64)
        except Exception as e:
            # remove part file to avoid reuse of broken data
            try:
                os.remove(part_path)
            except:
                pass
            return jsonify({"error": "base64 decode failed", "err": str(e)}), 400

        # save final image
        url, saved_path = save_bytes_and_url(data, ext="jpg")
        # cleanup
        try:
            os.remove(part_path)
        except:
            pass
        return jsonify({"img_url": url, "saved": saved_path}), 200

    # not last chunk yet
    return jsonify({"status": "ok", "received": idx, "total": total}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)