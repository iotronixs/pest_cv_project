from flask import Flask, request, render_template, url_for
import os, uuid

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
