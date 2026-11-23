import base64
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, render_template_string


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]

MODEL_CANDIDATES = [
    BASE_DIR / "best.onnx",
    REPO_ROOT / "ai-model" / "weights" / "best.onnx",
]

MODEL_PATH = next(
    (candidate for candidate in MODEL_CANDIDATES if candidate.is_file()),
    MODEL_CANDIDATES[0],
)

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
DEBUG = True

CLASS_NAMES = [
    "open",
    "short",
    "mousebite",
    "spur",
    "copper",
    "pin-hole",
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE


@app.after_request
def disable_cache(response):
    """Prevent browsers/proxies from reusing stale inspection pages."""
    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ============================================================
# LOAD ONNX MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("\n========================================")
print("       LOADING YOLO11n ONNX MODEL")
print("========================================")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

model_input = session.get_inputs()[0]
model_output = session.get_outputs()[0]
input_name = model_input.name

print("Model:", MODEL_PATH)
print("Input name:", input_name)
print("Input shape:", model_input.shape)
print("Output name:", model_output.name)
print("Output shape:", model_output.shape)
print("Execution providers:", session.get_providers())
print("Number of classes:", len(CLASS_NAMES))
print("========================================\n")


# ============================================================
# PREPROCESSING
# ============================================================

def letterbox(image, new_size=IMAGE_SIZE):
    """Resize with aspect ratio preserved and pad to a square."""
    height, width = image.shape[:2]
    scale = min(new_size / height, new_size / width)

    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    dw = new_size - resized_width
    dh = new_size - resized_height

    left = int(round(dw / 2 - 0.1))
    right = int(round(dw / 2 + 0.1))
    top = int(round(dh / 2 - 0.1))
    bottom = int(round(dh / 2 + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return padded, scale, left, top


def preprocess(image):
    padded, scale, pad_x, pad_y = letterbox(image, IMAGE_SIZE)

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    return tensor, scale, pad_x, pad_y


# ============================================================
# OUTPUT DECODING AND NMS
# ============================================================

def decode_output(raw_output):
    prediction = np.asarray(raw_output)

    if prediction.ndim == 3:
        if prediction.shape[0] != 1:
            raise ValueError(
                f"Unexpected batch size: {prediction.shape}"
            )
        prediction = prediction[0]

    if prediction.ndim != 2:
        raise ValueError(
            f"Unexpected YOLO output dimensions: {prediction.shape}"
        )

    expected_features = 4 + len(CLASS_NAMES)

    if prediction.shape[0] == expected_features:
        prediction = prediction.T
    elif prediction.shape[1] == expected_features:
        pass
    else:
        raise ValueError(
            f"Cannot interpret output shape {prediction.shape}; "
            f"expected one dimension to equal {expected_features}."
        )

    return prediction


def class_aware_nms(boxes, scores, class_ids):
    final_indices = []

    for class_id in range(len(CLASS_NAMES)):
        indices = [
            i for i, cid in enumerate(class_ids)
            if cid == class_id
        ]

        if not indices:
            continue

        class_boxes = [boxes[i] for i in indices]
        class_scores = [scores[i] for i in indices]

        kept = cv2.dnn.NMSBoxes(
            class_boxes,
            class_scores,
            CONF_THRESHOLD,
            IOU_THRESHOLD,
        )

        if len(kept) > 0:
            for kept_index in np.asarray(kept).flatten():
                final_indices.append(indices[int(kept_index)])

    return final_indices


# ============================================================
# INFERENCE + DIAGNOSTICS
# ============================================================

def run_inference(image):
    tensor, scale, pad_x, pad_y = preprocess(image)

    if DEBUG:
        print("\n========== INPUT DEBUG ==========")
        print("Original image:", image.shape)
        print("Model input:", tensor.shape)
        print("Tensor dtype:", tensor.dtype)
        print("Tensor min/max:", float(tensor.min()), float(tensor.max()))
        print("Letterbox scale:", scale)
        print("Padding:", pad_x, pad_y)
        print("=================================")

    start_time = time.perf_counter()
    outputs = session.run(None, {input_name: tensor})
    inference_ms = (time.perf_counter() - start_time) * 1000

    prediction = decode_output(outputs[0])
    class_scores = prediction[:, 4:]

    best_scores = np.max(class_scores, axis=1)
    best_class_ids = np.argmax(class_scores, axis=1)

    max_index = int(np.argmax(best_scores))
    max_confidence = float(best_scores[max_index])
    max_class_id = int(best_class_ids[max_index])

    threshold_counts = {
        str(threshold): int(np.sum(best_scores >= threshold))
        for threshold in (0.05, 0.10, 0.25)
    }

    top_indices = np.argsort(best_scores)[-10:][::-1]
    top_predictions = []

    for index in top_indices:
        top_predictions.append({
            "class_name": CLASS_NAMES[int(best_class_ids[index])],
            "confidence": float(best_scores[index]),
            "raw_box": [float(v) for v in prediction[index, :4]],
        })

    debug_info = {
        "output_shape": str(outputs[0].shape),
        "decoded_shape": str(prediction.shape),
        "candidate_count": int(len(prediction)),
        "max_confidence": max_confidence,
        "max_class": CLASS_NAMES[max_class_id],
        "threshold_counts": threshold_counts,
        "top_predictions": top_predictions,
        "inference_ms": inference_ms,
    }

    if DEBUG:
        print("\n========== YOLO OUTPUT DEBUG ==========")
        print("Raw output shape:", outputs[0].shape)
        print("Decoded output shape:", prediction.shape)
        print("Total candidates:", len(prediction))
        print("Maximum confidence:", max_confidence)
        print("Highest-scoring class:", CLASS_NAMES[max_class_id])
        print("Candidates above thresholds:", threshold_counts)
        print("Top 10 predictions:")
        for i, item in enumerate(top_predictions, start=1):
            print(
                f"{i}. {item['class_name']} "
                f"{item['confidence']:.6f} "
                f"box={item['raw_box']}"
            )

    boxes = []
    scores = []
    class_ids = []

    original_height, original_width = image.shape[:2]

    for row in prediction:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if not np.isfinite(confidence) or confidence < CONF_THRESHOLD:
            continue

        cx, cy, width, height = [float(v) for v in row[:4]]

        if not all(np.isfinite(v) for v in (cx, cy, width, height)):
            continue

        # Convert model-space center XYWH to original-image XYXY.
        x1 = (cx - width / 2 - pad_x) / scale
        y1 = (cy - height / 2 - pad_y) / scale
        x2 = (cx + width / 2 - pad_x) / scale
        y2 = (cy + height / 2 - pad_y) / scale

        x1 = int(np.clip(round(x1), 0, original_width - 1))
        y1 = int(np.clip(round(y1), 0, original_height - 1))
        x2 = int(np.clip(round(x2), 0, original_width - 1))
        y2 = int(np.clip(round(y2), 0, original_height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(confidence)
        class_ids.append(class_id)

    final_indices = class_aware_nms(boxes, scores, class_ids)

    detections = []
    for index in final_indices:
        detections.append({
            "class_id": class_ids[index],
            "class_name": CLASS_NAMES[class_ids[index]],
            "confidence": scores[index],
            "box": boxes[index],
        })

    debug_info["pre_nms_count"] = len(boxes)
    debug_info["post_nms_count"] = len(detections)

    if DEBUG:
        print("Candidates after confidence filter:", len(boxes))
        print("Detections after NMS:", len(detections))
        print("Inference time:", round(inference_ms, 2), "ms")
        print("=======================================\n")

    return detections, inference_ms, debug_info


# ============================================================
# ANNOTATION AND ENCODING
# ============================================================

def annotate_image(image, detections):
    output = image.copy()

    for detection in detections:
        x, y, w, h = detection["box"]
        label = (
            f"{detection['class_name']}: "
            f"{detection['confidence'] * 100:.1f}%"
        )
        color = (0, 220, 100)

        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        label_y = max(y, text_height + 8)

        cv2.rectangle(
            output,
            (x, label_y - text_height - 8),
            (x + text_width + 8, label_y + baseline),
            color,
            -1,
        )

        cv2.putText(
            output,
            label,
            (x + 4, label_y - 4),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return output


def image_to_base64(image):
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Failed to encode image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


# ============================================================
# WEBSITE
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PCB Inspection System</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0; background: #10151f; color: #edf2f7;
    font-family: Arial, sans-serif;
}
header {
    padding: 24px 5%; background: #171f2d;
    border-bottom: 1px solid #303b4d;
}
header h1 { margin: 0; font-size: 25px; }
header p { color: #9eacbf; margin-bottom: 0; }
main { max-width: 1250px; margin: 30px auto; padding: 0 20px; }
.panel {
    background: #192333; border: 1px solid #303d50;
    border-radius: 12px; padding: 22px; margin-bottom: 22px;
}
.upload-form {
    display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
}
input[type=file] {
    flex: 1; min-width: 200px; padding: 12px;
    background: #101824; border-radius: 7px; color: white;
}
button {
    padding: 13px 24px; border: none; border-radius: 7px;
    background: #18b77b; color: #061c13; font-weight: bold;
    cursor: pointer;
}
button:hover { background: #27d995; }
button:disabled { opacity: 0.65; cursor: wait; }
.grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
}
.image-box {
    background: #101824; border: 1px solid #303d50;
    border-radius: 9px; padding: 12px; min-width: 0;
}
.image-box h3 { font-size: 15px; }
.image-box img {
    width: 100%; height: auto; display: block; border-radius: 5px;
}
.stats {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
}
.stat { padding: 18px; background: #101824; border-radius: 9px; }
.stat small { display: block; color: #9eacbf; margin-bottom: 10px; }
.stat strong { font-size: 23px; }
.status {
    padding: 15px; border-radius: 8px; margin-bottom: 18px;
    font-size: 19px; font-weight: bold;
}
.fail { background: #48252d; color: #ff929f; }
.pass { background: #153d32; color: #63e6ae; }
.warning { background: #44351c; color: #ffd37b; }
table { width: 100%; border-collapse: collapse; margin-top: 15px; }
th, td {
    padding: 12px 8px; text-align: left;
    border-bottom: 1px solid #303d50;
}
th { color: #9eacbf; font-size: 12px; }
.error { color: #ff929f; }
.muted { color: #9eacbf; }
.debug {
    background: #101824; border-radius: 9px;
    padding: 16px; overflow-x: auto;
}
footer {
    text-align: center; padding: 25px;
    color: #8291a5; font-size: 12px;
}
@media(max-width: 700px) {
    .grid, .stats { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
    <h1>PCB INSPECTION SYSTEM</h1>
    <p>YOLO11n | ONNX Runtime | Raspberry Pi Edge AI</p>
</header>

<main>
    <section class="panel">
        <h2>Image Inspection</h2>
        <p>Upload a PCB image to detect manufacturing defects.</p>

        <form id="inspection-form" class="upload-form"
              action="/inspect" method="POST"
              enctype="multipart/form-data">
            <input id="pcb-image" type="file" name="pcb_image"
                   accept=".jpg,.jpeg,.png" required>
            <button id="inspect-button" type="submit">INSPECT PCB</button>
        </form>

        <p id="selected-file" class="muted"></p>
        <p id="loading-message" style="display:none;color:#63e6ae;font-weight:bold;">
            Processing image on Raspberry Pi... Please wait.
        </p>

        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}
    </section>

    {% if result %}
    <div id="inspection-result">
        <section class="panel">
            <h2>Inspection Result</h2>

            {% if result.count > 0 %}
                <div class="status fail">FAIL - DEFECTS DETECTED</div>
            {% elif result.debug.max_confidence >= 0.25 %}
                <div class="status warning">
                    No final detections after NMS - inspect diagnostics
                </div>
            {% else %}
                <div class="status pass">
                    NO DEFECTS DETECTED ABOVE THRESHOLD
                </div>
            {% endif %}

            <div class="stats">
                <div class="stat">
                    <small>TOTAL DEFECTS</small>
                    <strong>{{ result.count }}</strong>
                </div>
                <div class="stat">
                    <small>INFERENCE TIME</small>
                    <strong>{{ "%.1f"|format(result.inference_ms) }} ms</strong>
                </div>
                <div class="stat">
                    <small>MODEL</small>
                    <strong>YOLO11n</strong>
                </div>
            </div>
        </section>

        <section class="panel">
            <div class="grid">
                <div class="image-box">
                    <h3>Original PCB Image</h3>
                    <img src="data:image/jpeg;base64,{{ result.original }}"
                         alt="Original PCB">
                </div>
                <div class="image-box">
                    <h3>AI Detection Output</h3>
                    <img src="data:image/jpeg;base64,{{ result.annotated }}"
                         alt="Annotated PCB">
                </div>
            </div>
        </section>

        <section class="panel">
            <h2>Detected Defects</h2>
            {% if result.detections %}
                <table>
                    <thead>
                    <tr>
                        <th>#</th><th>DEFECT CLASS</th>
                        <th>CONFIDENCE</th><th>BOUNDING BOX</th>
                    </tr>
                    </thead>
                    <tbody>
                    {% for item in result.detections %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td>{{ item.class_name }}</td>
                        <td>{{ "%.2f"|format(item.confidence * 100) }}%</td>
                        <td>{{ item.box }}</td>
                    </tr>
                    {% endfor %}
                    </tbody>
                </table>
            {% else %}
                <p>No detections survived the confidence threshold and NMS.</p>
            {% endif %}
        </section>

        <section class="panel">
            <h2>Model Diagnostics</h2>
            <div class="debug">
                <p><strong>Raw ONNX output:</strong> {{ result.debug.output_shape }}</p>
                <p><strong>Decoded output:</strong> {{ result.debug.decoded_shape }}</p>
                <p><strong>Total candidates:</strong> {{ result.debug.candidate_count }}</p>
                <p><strong>Maximum confidence:</strong>
                   {{ "%.6f"|format(result.debug.max_confidence) }}</p>
                <p><strong>Highest-scoring class:</strong> {{ result.debug.max_class }}</p>
                <p><strong>Pre-NMS candidates:</strong> {{ result.debug.pre_nms_count }}</p>
                <p><strong>Post-NMS detections:</strong> {{ result.debug.post_nms_count }}</p>

                <h3>Candidate counts at different thresholds</h3>
                <table>
                    <tr><th>Threshold</th><th>Candidate count</th></tr>
                    {% for threshold, count in result.debug.threshold_counts.items() %}
                    <tr><td>{{ threshold }}</td><td>{{ count }}</td></tr>
                    {% endfor %}
                </table>

                <h3>Top 10 raw predictions</h3>
                <table>
                    <tr><th>#</th><th>Class</th><th>Confidence</th><th>Raw XYWH</th></tr>
                    {% for item in result.debug.top_predictions %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td>{{ item.class_name }}</td>
                        <td>{{ "%.6f"|format(item.confidence) }}</td>
                        <td>{{ item.raw_box }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </section>
    </div>
    {% endif %}
</main>

<footer>PCB Quality Inspection | On-device AI Demonstration</footer>

<script>
(() => {
    const imageInput = document.getElementById("pcb-image");
    const form = document.getElementById("inspection-form");
    const selectedFile = document.getElementById("selected-file");
    const loadingMessage = document.getElementById("loading-message");
    const inspectButton = document.getElementById("inspect-button");

    // Selecting a different image immediately removes the previous result.
    imageInput.addEventListener("change", () => {
        const previousResult = document.getElementById("inspection-result");
        if (previousResult) {
            previousResult.remove();
        }

        loadingMessage.style.display = "none";
        inspectButton.disabled = false;
        inspectButton.textContent = "INSPECT PCB";

        if (imageInput.files.length > 0) {
            selectedFile.textContent =
                "Selected image: " + imageInput.files[0].name;
        } else {
            selectedFile.textContent = "";
        }
    });

    // Prevent accidental empty submissions and show processing state.
    form.addEventListener("submit", (event) => {
        if (!imageInput.files.length) {
            event.preventDefault();
            return;
        }

        loadingMessage.style.display = "block";
        inspectButton.disabled = true;
        inspectButton.textContent = "PROCESSING...";
    });
})();
</script>
</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():
    response = app.make_response(render_template_string(
        HTML,
        result=None,
        error=None,
    ))
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/inspect", methods=["POST"])
def inspect():
    if "pcb_image" not in request.files:
        return render_template_string(
            HTML, result=None, error="Please select an image."
        ), 400

    uploaded_file = request.files["pcb_image"]

    if not uploaded_file.filename:
        return render_template_string(
            HTML, result=None, error="No image selected."
        ), 400

    filename = uploaded_file.filename.lower()

    if not filename.endswith((".jpg", ".jpeg", ".png")):
        return render_template_string(
            HTML, result=None, error="Please upload a JPG or PNG image."
        ), 400

    try:
        # The uploaded bytes and decoded image stay in RAM only.
        image_bytes = uploaded_file.read()

        if not image_bytes:
            raise ValueError("Uploaded file is empty.")

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("OpenCV could not decode this image.")

        print("\n========================================")
        print("NEW PCB IMAGE INSPECTION")
        print("Image:", filename)
        print("Dimensions:", image.shape)
        print("========================================")

        detections, inference_ms, debug_info = run_inference(image)
        annotated = annotate_image(image, detections)

        result = {
            "count": len(detections),
            "inference_ms": inference_ms,
            "original": image_to_base64(image),
            "annotated": image_to_base64(annotated),
            "detections": detections,
            "debug": debug_info,
        }

        print("Final defect count:", len(detections))
        print("Inspection completed.")
        print("========================================\n")

        return render_template_string(
            HTML,
            result=result,
            error=None,
        )

    except Exception as error:
        print("\nINSPECTION ERROR:", repr(error))
        return render_template_string(
            HTML,
            result=None,
            error=f"Inspection failed: {error}",
        ), 500


@app.errorhandler(413)
def file_too_large(error):
    return render_template_string(
        HTML,
        result=None,
        error="Image exceeds the 10 MB upload limit.",
    ), 413


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print("\n========================================")
    print("       PCB IMAGE INSPECTION SERVER")
    print("========================================")
    print("Model:", MODEL_PATH)
    print("Runtime: ONNX Runtime CPU")
    print("Input size:", IMAGE_SIZE)
    print("Confidence threshold:", CONF_THRESHOLD)
    print("IoU threshold:", IOU_THRESHOLD)
    print("Port: 5001")
    print("Image storage: DISABLED")
    print("Debug diagnostics: ENABLED")
    print("========================================\n")

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,
        threaded=True,
    )
