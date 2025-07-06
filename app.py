# app.py

import os
import sys
import pathlib
import base64
import re

import numpy as np
import torch
import cv2
import pytesseract
from PIL import Image
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS

# ─── Fix Windows pathlib pickling issue for YOLOv5 ─────────────────────────
pathlib.PurePosixPath = pathlib.WindowsPath
pathlib.PosixPath     = pathlib.WindowsPath

# ─── Add YOLOv5 repo to path ──────────────────────────────────────────────
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), 'yolov5')
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

# ─── YOLOv5 imports & load model once ─────────────────────────────────────
from yolov5.models.common       import DetectMultiBackend
from yolov5.utils.general       import non_max_suppression
from yolov5.utils.augmentations import letterbox

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
yolo        = DetectMultiBackend(MODEL_PATH, device='cpu')
stride      = yolo.stride

# ─── Allergen detection imports & dict ────────────────────────────────────
from allergen_detector import detect_allergens, get_allergen_dict
allergen_dict = get_allergen_dict()

# ─── Flask setup ──────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder='html',
    static_url_path=''
)
CORS(app)

# ─── Serve your HTML pages ─────────────────────────────────────────────────
@app.route('/')
def serve_home():
    return send_from_directory('html', 'Homepage.html')

@app.route('/scanner')
def serve_scanner():
    return send_from_directory('html', 'scanner.html')

@app.route('/ingredients')
def serve_ingredients():
    return send_from_directory('html', 'ingredientlist.html')

@app.route('/history')
def serve_history():
    return send_from_directory('html', 'history.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('html', filename)

# ─── Utility: preprocess for OCR ─────────────────────────────────────────
def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh

# ─── The API endpoint ─────────────────────────────────────────────────────
@app.route('/api/scan', methods=['POST'])
def api_scan():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # 1) Load image
    file    = request.files['image']
    img_p   = Image.open(file.stream).convert('RGB')
    img_np  = np.array(img_p)  # HWC, RGB

    # 2) YOLOv5 detect
    img_rsz, ratio, pad = letterbox(img_np, new_shape=640, stride=stride)
    ratio_x, ratio_y    = ratio if isinstance(ratio, (tuple, list)) else (ratio, ratio)
    pad_x, pad_y        = pad

    img_t = img_rsz.transpose(2, 0, 1)           # HWC → CHW
    img_t = np.ascontiguousarray(img_t)
    img_t = torch.from_numpy(img_t).float() / 255.0
    if img_t.ndim == 3:
        img_t = img_t.unsqueeze(0)

    pred = yolo(img_t, augment=False)[0]
    det  = non_max_suppression(pred, conf_thres=0.5)[0]

    # 3) Crop best box or fallback
    if det is not None and len(det):
        best = det[torch.argmax(det[:, 4])]
        x1, y1, x2, y2 = map(float, best[:4])

        x1 = int((x1 - pad_x) / ratio_x)
        y1 = int((y1 - pad_y) / ratio_y)
        x2 = int((x2 - pad_x) / ratio_x)
        y2 = int((y2 - pad_y) / ratio_y)

        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(img_np.shape[1], x2)
        y2 = min(img_np.shape[0], y2)

        crop = img_np[y1:y2, x1:x2]
    else:
        crop = img_np  # fallback

    # 4) OCR
    ocr_ready = preprocess_for_ocr(crop)
    raw_text  = pytesseract.image_to_string(ocr_ready, lang='eng').strip()

    # 5) Drop everything up to (and including) first "Ingredients" header
    m = re.search(r'(?i)ingredients?\s*:?', raw_text)
    if m:
        raw_text = raw_text[m.end():].strip()

    # 6) Extract full ingredient list
    tokens = re.split(r',|\band\b', raw_text)
    all_ingredients = [t.strip() for t in tokens if t.strip()]

    # 7) Allergen detection
    result  = detect_allergens(raw_text)
    details = [
        {'name': alg, 'risk': allergen_dict.get(alg, {}).get('risk', 'None')}
        for alg in result.get('matched_allergens', [])
    ]

    # 8) Encode crop as base64
    _, crop_buf = cv2.imencode('.jpg', crop)
    crop_b64    = base64.b64encode(crop_buf.tobytes()).decode('utf-8')
    crop_data   = f"data:image/jpeg;base64,{crop_b64}"

    # 9) Return combined JSON
    return jsonify({
        'text':               raw_text,
        'all_ingredients':    all_ingredients,
        'risk_level':         result.get('risk_level', 'N/A'),
        'matched_allergens':  result.get('matched_allergens', []),
        'allergen_details':   details,
        'detected_chemicals': result.get('detected_chemicals', []),
        'fuzzy_matches':      result.get('fuzzy_matches', []),
        'crop_image':         crop_data
    })

# ─── Run the app ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)