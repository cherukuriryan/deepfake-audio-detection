import os
import tempfile

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.predict_audio import predict_audio


SITE_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "Deepfake Detection Site Prototype"
)

app = Flask(
    __name__,
    template_folder=SITE_FOLDER,
    static_folder=SITE_FOLDER,
    static_url_path="/static"
)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


ALLOWED_EXTENSIONS = {
    "wav",
    "mp3",
    "m4a",
    "aiff",
    "aif",
    "flac"
}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template(
        "home.html"
    )


@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    if "audioFile" not in request.files:
        return "No audio file was uploaded.", 400

    audio_file = request.files["audioFile"]

    if audio_file.filename == "":
        return "No audio file was selected.", 400

    if not allowed_file(
        audio_file.filename
    ):
        return "Unsupported audio file type.", 400

    filename = secure_filename(
        audio_file.filename
    )

    extension = os.path.splitext(
        filename
    )[1]

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_path = temp_file.name

            audio_file.save(
                temp_path
            )

        prediction, confidence = predict_audio(
            temp_path
        )

        prediction_text = str(
            prediction
        ).lower()

        if prediction_text in {
            "spoofed",
            "spoof",
            "fake",
            "1"
        }:

            result_title = "Likely Synthetic"

            result_description = (
                "The model detected acoustic patterns "
                "that are more consistent with synthetic "
                "or manipulated speech."
            )

            classification = "Spoofed"

        else:

            result_title = "Likely Real"

            result_description = (
                "The model detected acoustic patterns "
                "that are more consistent with genuine "
                "human speech."
            )

            classification = "Real"

        return render_template(
            "results.html",
            result_title=result_title,
            result_description=result_description,
            classification=classification,
            confidence=round(
                float(confidence),
                2
            ),
            filename=filename
        )

    except Exception as error:

        print(
            "Prediction error:",
            error
        )

        return (
            "The audio could not be analyzed. "
            "Please try another file.",
            500
        )

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            os.remove(
                temp_path
            )


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
