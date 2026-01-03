"""
Flask Web Application for Real-Time Football Player Tracking
Provides MJPEG streaming with live jersey number recognition.
"""

import os
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from werkzeug.utils import secure_filename

from config import FlaskConfig, ALLOWED_EXTENSIONS, UPLOAD_FOLDER
from logger import get_logger, info, warning, error
from cleanup import run_auto_cleanup, get_upload_folder_size_mb
from camera import VideoCamera

# Initialize logger
logger = get_logger("app")

# Create Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = FlaskConfig.SECRET_KEY
app.config["UPLOAD_FOLDER"] = FlaskConfig.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = FlaskConfig.MAX_CONTENT_LENGTH

# Global camera instance
camera = None


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_camera():
    """Get the current camera instance."""
    global camera
    return camera


def set_camera(video_path: str):
    """Initialize camera with a new video."""
    global camera
    if camera:
        logger.info("Releasing previous camera instance")
        camera.release()
    logger.info(f"Initializing camera with: {video_path}")
    camera = VideoCamera(video_path)
    return camera


def generate_frames():
    """Generator function for MJPEG streaming."""
    cam = get_camera()
    if cam is None:
        logger.warning("No camera instance available for streaming")
        return

    frame_count = 0
    while True:
        frame = cam.get_frame()

        if frame is None:
            logger.info("Video stream ended")
            break

        frame_count += 1
        if frame_count % 100 == 0:
            logger.debug(f"Streamed {frame_count} frames")

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    """Upload page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    """Handle video upload."""
    if "video" not in request.files:
        logger.warning("Upload attempt with no video file")
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]

    if file.filename == "":
        logger.warning("Upload attempt with empty filename")
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({"error": "Invalid file type. Allowed: mp4, avi, mov, mkv, webm"}), 400

    # Save file
    filename = secure_filename(file.filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    logger.info(f"Saving uploaded file: {filename}")
    file.save(filepath)

    try:
        # Initialize camera with uploaded video
        set_camera(filepath)
        logger.info(f"Video loaded successfully: {filename}")
        return redirect(url_for("dashboard"))
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard")
def dashboard():
    """Dashboard page with video stream and stats."""
    if get_camera() is None:
        logger.warning("Dashboard accessed without loaded video")
        return redirect(url_for("index"))
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    if get_camera() is None:
        return "No video loaded", 404

    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    """JSON endpoint for current frame count and player data."""
    cam = get_camera()
    if cam is None:
        return jsonify(
            {
                "current_frame": 0,
                "total_frames": 0,
                "fps": 0,
                "is_playing": False,
                "players": [],
            }
        )

    return jsonify(cam.get_stats())


@app.route("/player_crops")
def player_crops():
    """JSON endpoint for player crop images (base64 encoded)."""
    cam = get_camera()
    if cam is None:
        return jsonify({})

    return jsonify(cam.get_player_crops())


@app.route("/toggle_play", methods=["POST"])
def toggle_play():
    """Toggle play/pause state."""
    cam = get_camera()
    if cam is None:
        return jsonify({"error": "No video loaded"}), 404

    is_playing = cam.toggle_play()
    logger.info(f"Playback toggled: {'playing' if is_playing else 'paused'}")
    return jsonify({"is_playing": is_playing})


@app.route("/seek", methods=["POST"])
def seek():
    """Seek to a specific frame."""
    cam = get_camera()
    if cam is None:
        return jsonify({"error": "No video loaded"}), 404

    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "Frame number required"}), 400

    frame_number = int(data["frame"])
    success = cam.seek_to_frame(frame_number)
    logger.info(f"Seek to frame {frame_number}: {'success' if success else 'failed'}")

    return jsonify({
        "success": success,
        "current_frame": cam.current_frame
    })


@app.route("/get_frame")
def get_frame():
    """Get a single frame (used when paused/seeking)."""
    cam = get_camera()
    if cam is None:
        return "No video loaded", 404

    frame_bytes = cam.get_single_frame()
    if frame_bytes is None:
        return "Could not get frame", 500

    return Response(frame_bytes, mimetype="image/jpeg")


@app.route("/reset", methods=["POST"])
def reset():
    """Reset and go back to upload page."""
    global camera
    if camera:
        logger.info("Resetting camera instance")
        camera.release()
        camera = None
    return jsonify({"success": True})


@app.route("/health")
def health():
    """Health check endpoint."""
    upload_size = get_upload_folder_size_mb()
    return jsonify(
        {
            "status": "healthy",
            "camera_active": camera is not None,
            "upload_folder_size_mb": round(upload_size, 1),
        }
    )


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(str(UPLOAD_FOLDER), exist_ok=True)

    # Run cleanup on startup
    run_auto_cleanup()

    # Startup banner
    logger.info("=" * 60)
    logger.info("Football Player Tracking Application")
    logger.info("=" * 60)
    logger.info(f"Starting server on http://{FlaskConfig.HOST}:{FlaskConfig.PORT}")
    logger.info("Upload a video to begin tracking...")
    logger.info("=" * 60)

    app.run(
        host=FlaskConfig.HOST,
        port=FlaskConfig.PORT,
        debug=FlaskConfig.DEBUG,
        threaded=True,
    )
