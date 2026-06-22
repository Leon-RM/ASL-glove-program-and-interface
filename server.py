import sys
sys.coinit_flags = 0  # Force COM Multi-Threaded Apartment (MTA) model to avoid Bleak conflict
import asyncio
import csv
import json
import os
import pickle
import queue
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager

import numpy as np
import pyttsx3
from bleak import BleakClient, BleakScanner
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLE_DEVICE_NAME = "Leon_Glove_BLE"
BLE_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
BLE_CHAR_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "training_data.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")

FEATURE_COUNT = 11
FEATURE_NAMES = [
    "Thumb", "Index", "Middle", "Ring", "Pinky",
    "AccelX", "AccelY", "AccelZ",
    "GyroX", "GyroY", "GyroZ",
]

NUM_ESTIMATORS = 100
STABILITY_WINDOW = 10
STABILITY_THRESHOLD = 7
CONFIDENCE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# TTS Engine (runs in a background thread)
# ---------------------------------------------------------------------------
_tts_queue: queue.Queue = queue.Queue()


def _tts_worker():
    import comtypes
    try:
        comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
    except Exception:
        try:
            comtypes.CoInitialize()
        except Exception:
            pass
    try:
        engine = pyttsx3.init()
        rate = engine.getProperty("rate")
        engine.setProperty("rate", rate - 20)
        while True:
            text = _tts_queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
    finally:
        comtypes.CoUninitialize()


tts_thread = threading.Thread(target=_tts_worker, daemon=True)
tts_thread.start()


def speak(text: str) -> None:
    _tts_queue.put(text)


# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.ble_connected = False
        self.recording = False
        self.current_label = ""
        self.model: Pipeline | None = None
        self.classes: list[str] = []
        self.trained = False
        self.history_text = ""
        self.stability_window = STABILITY_WINDOW
        self.stability_threshold = STABILITY_THRESHOLD
        self.prediction_stability: deque[str] = deque(maxlen=STABILITY_WINDOW)
        self.last_spoken = ""
        self.cooldown_seconds = 2.0
        self._last_trigger_time = 0.0
        self._latest_features: list[float] | None = None
        self._ws_connections: list[WebSocket] = []
        self.training = False
        # Cached training results for download endpoints
        self.last_cm: list | None = None
        self.last_cm_classes: list[str] = []
        self.last_training_loss: list[float] = []
        self.last_validation_scores: list[float] = []

    @property
    def latest_features(self) -> list[float] | None:
        return self._latest_features

    @latest_features.setter
    def latest_features(self, value: list[float] | None) -> None:
        self._latest_features = value

    def add_ws(self, ws: WebSocket) -> None:
        self._ws_connections.append(ws)

    def remove_ws(self, ws: WebSocket) -> None:
        if ws in self._ws_connections:
            self._ws_connections.remove(ws)

    async def broadcast(self, message: dict) -> None:
        body = json.dumps(message)
        dead: list[WebSocket] = []
        for ws in self._ws_connections:
            try:
                await ws.send_text(body)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.remove_ws(ws)

    def can_trigger(self) -> bool:
        return time.time() - self._last_trigger_time >= self.cooldown_seconds

    def mark_triggered(self) -> None:
        self._last_trigger_time = time.time()

    def update_stability_settings(self, window: int, threshold: int):
        self.stability_window = max(1, min(50, window))
        self.stability_threshold = max(1, min(self.stability_window, threshold))
        old_elems = list(self.prediction_stability)
        self.prediction_stability = deque(old_elems, maxlen=self.stability_window)


state = AppState()

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
csv_file_lock = threading.RLock()


def _ensure_csv_headers():
    with csv_file_lock:
        if not os.path.exists(CSV_PATH):
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(CSV_PATH, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "thumb", "index", "middle", "ring", "pinky",
                    "accX", "accY", "accZ",
                    "gyroX", "gyroY", "gyroZ",
                    "label",
                ])


def load_csv_data():
    if not os.path.exists(CSV_PATH):
        return [], []
    with csv_file_lock:
        X, y = [], []
        with open(CSV_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    feats = [float(row[k]) for k in [
                        "thumb", "index", "middle", "ring", "pinky",
                        "accX", "accY", "accZ",
                        "gyroX", "gyroY", "gyroZ",
                    ]]
                    X.append(feats)
                    y.append(row["label"].strip())
                except (ValueError, KeyError):
                    continue
        return np.array(X), np.array(y)


_csv_buffer: list[list[str | float]] = []
_csv_buffer_lock = threading.Lock()

def append_csv_row(features: list[float], label: str) -> None:
    with _csv_buffer_lock:
        _csv_buffer.append(list(features) + [label])


def _flush_csv():
    with _csv_buffer_lock:
        if not _csv_buffer:
            return
        batch = _csv_buffer[:]
        _csv_buffer.clear()
    with csv_file_lock:
        _ensure_csv_headers()
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            w.writerows(batch)


def save_model():
    data = {
        "model": state.model,
        "classes": state.classes,
        "last_cm": state.last_cm,
        "last_cm_classes": state.last_cm_classes,
        "last_training_loss": state.last_training_loss,
        "last_validation_scores": state.last_validation_scores,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(data, f)


def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            loaded_model = data["model"]
            if hasattr(loaded_model, "named_steps") and "mlp" in loaded_model.named_steps:
                state.model = loaded_model
                state.classes = data["classes"]
                state.trained = True
                state.last_cm = data.get("last_cm")
                state.last_cm_classes = data.get("last_cm_classes", [])
                state.last_training_loss = data.get("last_training_loss", [])
                state.last_validation_scores = data.get("last_validation_scores", [])
        except Exception:
            pass


def delete_csv_class(label: str) -> bool:
    if not os.path.exists(CSV_PATH):
        return False
    with csv_file_lock:
        rows = []
        header = None
        with open(CSV_PATH, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) > 11 and row[11].strip() != label:
                    rows.append(row)
        
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            if header:
                w.writerow(header)
            w.writerows(rows)
    return True


def delete_csv_sample(index: int) -> bool:
    if not os.path.exists(CSV_PATH):
        return False
    with csv_file_lock:
        rows = []
        header = None
        with open(CSV_PATH, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for i, row in enumerate(reader):
                if i != index:
                    rows.append(row)
                    
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            if header:
                w.writerow(header)
            w.writerows(rows)
    return True


def rename_csv_class(old_label: str, new_label: str) -> bool:
    if not os.path.exists(CSV_PATH):
        return False
    with csv_file_lock:
        rows = []
        header = None
        with open(CSV_PATH, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) > 11:
                    if row[11].strip() == old_label:
                        row[11] = new_label
                rows.append(row)
        
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            if header:
                w.writerow(header)
            w.writerows(rows)
    return True


def relabel_csv_sample(index: int, new_label: str) -> bool:
    if not os.path.exists(CSV_PATH):
        return False
    with csv_file_lock:
        rows = []
        header = None
        with open(CSV_PATH, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for i, row in enumerate(reader):
                if i == index:
                    if len(row) > 11:
                        row[11] = new_label
                rows.append(row)
                
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            if header:
                w.writerow(header)
            w.writerows(rows)
    return True


def get_csv_samples_mapping():
    if not os.path.exists(CSV_PATH):
        return {}
    with csv_file_lock:
        mapping = {}
        with open(CSV_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                label = row.get("label", "").strip()
                if not label:
                    continue
                if label not in mapping:
                    mapping[label] = []
                
                thumb = row.get("thumb", "0")
                index_val = row.get("index", "0")
                try:
                    summary = f"T:{int(float(thumb))} I:{int(float(index_val))}"
                except Exception:
                    summary = f"T:{thumb} I:{index_val}"
                    
                mapping[label].append({
                    "index": i,
                    "summary": summary
                })
        return mapping


async def broadcast_updated_status():
    X, y = load_csv_data()
    sample_count = len(y)
    unique_labels = sorted(set(y)) if sample_count else []
    from collections import Counter
    label_counts = dict(Counter(y)) if sample_count else {}
    samples_map = get_csv_samples_mapping()
    
    await state.broadcast({
        "type": "status",
        "recording": state.recording,
        "current_label": state.current_label,
        "samples": sample_count,
        "unique_labels": unique_labels,
        "label_counts": label_counts,
        "samples_map": samples_map
    })


# ---------------------------------------------------------------------------
# BLE
# ---------------------------------------------------------------------------
async def ble_loop():
    while True:
        if state.ble_connected:
            await asyncio.sleep(2)
            continue

        await state.broadcast({
            "type": "status",
            "ble": False,
            "ble_status": "Scanning for glove...",
        })

        try:
            device = await BleakScanner.find_device_by_name(BLE_DEVICE_NAME)
        except Exception as e:
            await state.broadcast({
                "type": "status",
                "ble": False,
                "ble_status": f"Scan error: {e}",
            })
            await asyncio.sleep(5)
            continue

        if not device:
            await state.broadcast({
                "type": "status",
                "ble": False,
                "ble_status": "Glove not found",
            })
            await asyncio.sleep(5)
            continue

        await state.broadcast({
            "type": "status",
            "ble": False,
            "ble_status": f"Found! Connecting...",
        })

        try:
            async with BleakClient(device) as client:
                state.ble_connected = True
                await state.broadcast({"type": "status", "ble": True, "ble_status": "Connected"})
                loop = asyncio.get_running_loop()

                def notify_handler(sender, data: bytearray):
                    raw = data.decode("utf-8").strip()
                    parts = raw.split(",")
                    if len(parts) != FEATURE_COUNT:
                        return
                    try:
                        feats = [float(p) for p in parts]
                    except ValueError:
                        return
                    state.latest_features = feats
                    asyncio.run_coroutine_threadsafe(_on_sensor_data(feats), loop)

                await client.start_notify(BLE_CHAR_TX_UUID, notify_handler)

                while client.is_connected:
                    await asyncio.sleep(1)

        except Exception as e:
            await state.broadcast({
                "type": "status",
                "ble": False,
                "ble_status": f"Conn lost: {type(e).__name__}",
            })

        state.ble_connected = False
        await asyncio.sleep(3)


_last_sensor_t = 0.0
_last_pred_t = 0.0
SENSOR_BROADCAST_INTERVAL = 0.1  # 10 Hz max to UI
PREDICTION_INTERVAL = 0.1       # 10 Hz max for prediction

async def _on_sensor_data(feats: list[float]):
    global _last_sensor_t, _last_pred_t

    if state.training:
        return

    if state.recording and state.current_label:
        append_csv_row(feats, state.current_label)

    now = time.time()
    if now - _last_sensor_t >= SENSOR_BROADCAST_INTERVAL:
        _last_sensor_t = now
        await state.broadcast({"type": "sensor", "values": feats})

    if state.trained and not state.recording and state.model is not None:
        if now - _last_pred_t >= PREDICTION_INTERVAL:
            _last_pred_t = now
            await _run_prediction(feats)


async def _csv_flusher():
    while True:
        await asyncio.sleep(0.5)
        _flush_csv()


async def _run_prediction(feats: list[float]):
    try:
        X = np.array([feats])
        pred = state.model.predict(X)[0]
        probs = state.model.predict_proba(X)[0]
        idx = list(state.model.classes_).index(pred)
        conf = float(probs[idx])
    except Exception:
        return

    state.prediction_stability.append(pred)

    if conf < CONFIDENCE_THRESHOLD:
        return

    if len(state.prediction_stability) < state.stability_window:
        return

    counter = Counter(state.prediction_stability)
    most_common, count = counter.most_common(1)[0]

    if count < state.stability_threshold:
        return

    # Check for Idle/Rest states
    is_idle = most_common.lower() in ["idle", "rest"]
    if is_idle:
        # Display Idle in UI, but don't append to history and don't speak
        state.last_spoken = most_common
        await state.broadcast({
            "type": "prediction",
            "gesture": "Idle",
            "confidence": round(conf, 3),
            "history": state.history_text,
        })
        return

    if not state.can_trigger():
        return

    if most_common == state.last_spoken:
        return

    state.last_spoken = most_common
    state.mark_triggered()

    if state.history_text:
        state.history_text += " " + most_common
    else:
        state.history_text = most_common

    speak(most_common)

    await state.broadcast({
        "type": "prediction",
        "gesture": most_common,
        "confidence": round(conf, 3),
        "history": state.history_text,
    })

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
static_dir = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app):
    _ensure_csv_headers()
    load_model()
    ble_task = asyncio.create_task(ble_loop())
    csv_task = asyncio.create_task(_csv_flusher())
    yield
    ble_task.cancel()
    csv_task.cancel()
    _flush_csv()
    try:
        await ble_task
    except asyncio.CancelledError:
        pass
    try:
        await csv_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="ASL Translation Interface", lifespan=lifespan)

if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>ASL Glove Server</h1><p>Static frontend not found.</p>")


@app.get("/api/status")
async def api_status():
    X, y = load_csv_data()
    sample_count = len(y)
    unique_labels = sorted(set(y)) if sample_count else []
    from collections import Counter
    label_counts = dict(Counter(y)) if sample_count else {}
    samples_map = get_csv_samples_mapping()
    return {
        "ble": state.ble_connected,
        "ble_status": "Connected" if state.ble_connected else "Disconnected",
        "recording": state.recording,
        "current_label": state.current_label,
        "trained": state.trained,
        "classes": state.classes,
        "samples": sample_count,
        "unique_labels": unique_labels,
        "label_counts": label_counts,
        "samples_map": samples_map,
        "history": state.history_text,
        "cooldown": state.cooldown_seconds,
        "stability_window": state.stability_window,
        "stability_threshold": state.stability_threshold,
    }


@app.post("/api/record/start")
async def api_record_start(body: dict):
    label = body.get("label", "").strip()
    if not label:
        return {"ok": False, "error": "Label is required"}
    state.recording = True
    state.current_label = label
    await state.broadcast({
        "type": "status",
        "recording": True,
        "current_label": label,
    })
    return {"ok": True}


@app.post("/api/record/stop")
async def api_record_stop():
    state.recording = False
    state.current_label = ""
    await state.broadcast({"type": "status", "recording": False})
    return {"ok": True}


# ---------------------------------------------------------------------------
# Training helper (runs CPU work in thread to keep UI responsive)
# ---------------------------------------------------------------------------
def _train_sync(X, y, classes):
    n_classes = len(classes)

    y_encoded = np.array([classes.index(c) for c in y])
    min_class_count = int(min(np.bincount(y_encoded)))
    n_splits = max(2, min(5, min_class_count))

    cm = None
    cr = None
    cv_scores = None

    def get_new_model():
        return Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42
            ))
        ])

    model = get_new_model()
    model.fit(X, y)
    accuracy = float(model.score(X, y))

    if n_splits >= 2 and n_classes >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_model = get_new_model()
        try:
            y_pred = cross_val_predict(cv_model, X, y, cv=cv, method="predict")
            cm = confusion_matrix(y, y_pred).tolist()
            cr = classification_report(y, y_pred, output_dict=True)
            cv_scores = cross_val_score(cv_model, X, y, cv=cv).tolist()
        except Exception:
            pass

    if cm is None:
        try:
            y_pred_train = model.predict(X)
            cm = confusion_matrix(y, y_pred_train).tolist()
            cr = classification_report(y, y_pred_train, output_dict=True)
            cv_scores = [accuracy]
        except Exception:
            pass

    mlp_model = model.named_steps['mlp']

    # Compute feature importance as the mean absolute weights connected to each input feature
    try:
        weights = mlp_model.coefs_[0]
        fi = np.mean(np.abs(weights), axis=1).tolist()
    except Exception:
        fi = [0.0] * FEATURE_COUNT

    # Extract validation and training loss curves
    training_loss = []
    validation_scores = []
    try:
        training_loss = [float(loss) for loss in mlp_model.loss_curve_]
        if hasattr(mlp_model, "validation_scores_") and mlp_model.validation_scores_ is not None:
            validation_scores = [float(score) for score in mlp_model.validation_scores_]
    except Exception:
        pass

    return {
        "model": model,
        "classes": classes,
        "fi": fi,
        "oob_score": None,
        "accuracy": accuracy,
        "cm": cm,
        "cr": cr,
        "cv_scores": cv_scores,
        "n_splits": n_splits,
        "training_loss": training_loss,
        "validation_scores": validation_scores,
    }



async def _train_model() -> dict:
    if state.training:
        return {"ok": False, "error": "Already training"}
    state.training = True
    try:
        X, y = load_csv_data()
        if len(X) < 5:
            return {"ok": False, "error": f"Not enough data ({len(X)} samples, need at least 5)"}
        unique = set(y)
        if len(unique) < 2:
            return {"ok": False, "error": f"Need at least 2 classes, got {len(unique)}"}

        classes = sorted(unique)

        result = await asyncio.to_thread(_train_sync, X, y, classes)

        state.model = result["model"]
        state.classes = result["classes"]
        state.trained = True
        state.prediction_stability.clear()
        state.last_spoken = ""
        # Cache training results for download
        state.last_cm = result["cm"]
        state.last_cm_classes = result["classes"]
        state.last_training_loss = result["training_loss"]
        state.last_validation_scores = result["validation_scores"]
        save_model()

        from collections import Counter
        label_counts = dict(Counter(y)) if len(y) else {}
        samples_map = get_csv_samples_mapping()

        await state.broadcast({
            "type": "trained",
            "classes": result["classes"],
            "samples": len(y),
            "label_counts": label_counts,
            "samples_map": samples_map,
            "accuracy": round(result["accuracy"], 4),
            "oob_score": None,
            "confusion_matrix": result["cm"],
            "classification_report": result["cr"],
            "feature_importance": result["fi"],
            "feature_names": FEATURE_NAMES,
            "cv_scores": result["cv_scores"],
            "n_splits": result["n_splits"] if result["cv_scores"] else None,
            "training_loss": result["training_loss"],
            "validation_scores": result["validation_scores"],
        })
        return {"ok": True, "classes": result["classes"], "samples": len(y)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        state.training = False


@app.post("/api/train")
async def api_train():
    return await _train_model()


@app.post("/api/clear")
async def api_clear():
    state.recording = False
    state.current_label = ""
    state.trained = False
    state.model = None
    state.classes = []
    state.history_text = ""
    state.prediction_stability.clear()
    state.last_spoken = ""
    with csv_file_lock:
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    _ensure_csv_headers()
    await state.broadcast({"type": "cleared"})
    return {"ok": True}


@app.post("/api/cooldown")
async def api_cooldown(body: dict):
    val = float(body.get("seconds", 2.0))
    state.cooldown_seconds = max(0.5, min(10.0, val))
    return {"ok": True, "cooldown": state.cooldown_seconds}


@app.get("/api/download/confusion_matrix")
async def download_confusion_matrix():
    """Generate confusion matrix PNG from cached training results."""
    if not state.trained:
        return HTMLResponse("<h3>Model is not trained yet.</h3>", status_code=400)
    
    cm = state.last_cm
    classes = state.last_cm_classes
    
    if cm is None or not classes:
        # Try to reconstruct it on the fly using the trained model and the CSV dataset
        try:
            X, y = load_csv_data()
            if len(X) >= 5 and state.model is not None:
                classes = state.classes if state.classes else sorted(set(y))
                y_pred = state.model.predict(X)
                from sklearn.metrics import confusion_matrix as sk_cm
                cm = sk_cm(y, y_pred).tolist()
                state.last_cm = cm
                state.last_cm_classes = classes
        except Exception as e:
            return HTMLResponse(f"<h3>No confusion matrix data available and reconstruction failed: {e}</h3>", status_code=400)
            
    if cm is None or not classes:
        return HTMLResponse("<h3>No confusion matrix data available. Please retrain the model.</h3>", status_code=400)
    
    try:
        cm_arr = np.array(cm)
        n_classes = len(classes)
        
        fig_size = max(8, n_classes * 0.8 + 4)
        plt.figure(figsize=(fig_size, fig_size * 0.8))
        sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('ASL Translation — Confusion Matrix (Cross-Validation / Fallback)',
                  fontsize=14, fontweight='bold', pad=15)
        plt.ylabel('Actual Gesture', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Gesture', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close()
        
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=confusion_matrix.png"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTMLResponse(f"<h3>Error generating confusion matrix: {e}</h3>", status_code=500)


@app.get("/api/download/loss_graph")
async def download_loss_graph():
    """Generate loss graph PNG from cached training results."""
    if not state.trained:
        return HTMLResponse("<h3>Model is not trained yet.</h3>", status_code=400)
    
    loss_curve = state.last_training_loss
    validation_scores = state.last_validation_scores
    
    if not loss_curve:
        # Try to retrieve it from the trained model's classifier step
        try:
            if state.model is not None:
                mlp_model = state.model.named_steps['mlp']
                if hasattr(mlp_model, "loss_curve_") and mlp_model.loss_curve_:
                    loss_curve = [float(loss) for loss in mlp_model.loss_curve_]
                    state.last_training_loss = loss_curve
                if hasattr(mlp_model, "validation_scores_") and mlp_model.validation_scores_ is not None:
                    validation_scores = [float(score) for score in mlp_model.validation_scores_]
                    state.last_validation_scores = validation_scores
        except Exception:
            pass
            
    if not loss_curve:
        return HTMLResponse("<h3>No training loss data available. Please retrain the model.</h3>", status_code=400)
    
    try:
        epochs = list(range(1, len(loss_curve) + 1))
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = '#ef4444'
        ax1.set_xlabel('Epochs', fontweight='bold', labelpad=10)
        ax1.set_ylabel('Training Loss (Cross-Entropy)', color=color, fontweight='bold')
        ax1.plot(epochs, loss_curve, color=color, linewidth=2,
                 marker='o' if len(loss_curve) <= 40 else None,
                 markersize=4, label='Training Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        if validation_scores and len(validation_scores) == len(loss_curve):
            ax2 = ax1.twinx()
            color = '#3b82f6'
            ax2.set_ylabel('Validation Score (Accuracy)', color=color, fontweight='bold')
            ax2.plot(epochs, validation_scores, color=color, linewidth=2,
                     marker='x' if len(loss_curve) <= 40 else None,
                     markersize=4, linestyle='--', label='Validation Score')
            ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('ASL Translation — Loss & Validation Curve',
                  fontsize=14, fontweight='bold', pad=15)
        fig.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close()
        
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=loss_graph.png"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTMLResponse(f"<h3>Error generating loss graph: {e}</h3>", status_code=500)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.add_ws(ws)
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            t = msg.get("type", "")

            if t == "record_start":
                label = msg.get("label", "").strip()
                if label:
                    state.recording = True
                    state.current_label = label
                    await state.broadcast({
                        "type": "status",
                        "recording": True,
                        "current_label": label,
                    })

            elif t == "record_stop":
                state.recording = False
                state.current_label = ""
                await state.broadcast({"type": "status", "recording": False})

            elif t == "train":
                result = await _train_model()
                if not result.get("ok"):
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": result.get("error", "Training failed"),
                    }))

            elif t == "clear_history":
                state.history_text = ""
                await state.broadcast({"type": "history_cleared"})

            elif t == "clear_all":
                state.recording = False
                state.current_label = ""
                state.trained = False
                state.model = None
                state.classes = []
                state.history_text = ""
                state.prediction_stability.clear()
                state.last_spoken = ""
                with csv_file_lock:
                    if os.path.exists(CSV_PATH):
                        os.remove(CSV_PATH)
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                _ensure_csv_headers()
                await state.broadcast({"type": "cleared"})
                await broadcast_updated_status()

            elif t == "cooldown":
                val = float(msg.get("seconds", 2.0))
                state.cooldown_seconds = max(0.5, min(10.0, val))

            elif t == "stability":
                window = int(msg.get("window", 10))
                threshold = int(msg.get("threshold", 7))
                state.update_stability_settings(window, threshold)

            elif t == "feedback":
                label = msg.get("label", "").strip()
                if label and state.latest_features:
                    append_csv_row(state.latest_features, label)
                    _flush_csv()
                    await broadcast_updated_status()

            elif t == "delete_class":
                label = msg.get("label", "").strip()
                if label:
                    delete_csv_class(label)
                    X, y = load_csv_data()
                    unique_labels = sorted(set(y)) if len(y) else []
                    if state.trained and len(X) >= 5 and len(unique_labels) >= 2:
                        await _train_model()
                    else:
                        state.trained = False
                        state.model = None
                        state.classes = []
                        await state.broadcast({"type": "cleared"})
                    await broadcast_updated_status()

            elif t == "delete_sample":
                idx = int(msg.get("index", -1))
                if idx >= 0:
                    delete_csv_sample(idx)
                    X, y = load_csv_data()
                    unique_labels = sorted(set(y)) if len(y) else []
                    if state.trained and len(X) >= 5 and len(unique_labels) >= 2:
                        await _train_model()
                    else:
                        state.trained = False
                        state.model = None
                        state.classes = []
                        await state.broadcast({"type": "cleared"})
                    await broadcast_updated_status()

            elif t == "rename_class":
                old_label = msg.get("old_label", "").strip()
                new_label = msg.get("new_label", "").strip()
                if old_label and new_label:
                    rename_csv_class(old_label, new_label)
                    X, y = load_csv_data()
                    unique_labels = sorted(set(y)) if len(y) else []
                    if state.trained and len(X) >= 5 and len(unique_labels) >= 2:
                        await _train_model()
                    else:
                        state.trained = False
                        state.model = None
                        state.classes = []
                        await state.broadcast({"type": "cleared"})
                    await broadcast_updated_status()

            elif t == "relabel_sample":
                idx = int(msg.get("index", -1))
                new_label = msg.get("new_label", "").strip()
                if idx >= 0 and new_label:
                    relabel_csv_sample(idx, new_label)
                    X, y = load_csv_data()
                    unique_labels = sorted(set(y)) if len(y) else []
                    if state.trained and len(X) >= 5 and len(unique_labels) >= 2:
                        await _train_model()
                    else:
                        state.trained = False
                        state.model = None
                        state.classes = []
                        await state.broadcast({"type": "cleared"})
                    await broadcast_updated_status()

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        state.remove_ws(ws)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
