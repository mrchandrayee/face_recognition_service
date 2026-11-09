from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from .config import CONFIG, GALLERY_DIR

class FaceService:
    """Local, opt-in face matching against images explicitly enrolled by a user."""
    def __init__(self):
        self.app = FaceAnalysis(name=CONFIG["recognition"]["model_name"],
                                providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=tuple(CONFIG["recognition"]["detection_size"]))
        self.threshold = float(CONFIG["recognition"]["match_threshold"])
        self.gallery = {}
        self.reload_gallery()

    @staticmethod
    def _normalise(vector):
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def reload_gallery(self):
        self.gallery.clear()
        for person_dir in GALLERY_DIR.iterdir():
            if not person_dir.is_dir():
                continue
            embeddings = []
            for path in person_dir.iterdir():
                image = cv2.imread(str(path))
                if image is None:
                    continue
                faces = self.app.get(image)
                if faces:
                    # Use the largest face in the submitted enrollment image.
                    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                    embeddings.append(self._normalise(face.embedding))
            if embeddings:
                self.gallery[person_dir.name] = np.mean(embeddings, axis=0)

    def enroll(self, name: str, image_bytes: bytes):
        safe_name = "".join(c for c in name.strip() if c.isalnum() or c in "-_ ").strip()
        if not safe_name:
            raise ValueError("Please provide a valid display name.")
        import re
        safe_name = re.sub(r"\s+", "_", safe_name)[:64]
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Uploaded file is not a readable image.")
        faces = self.app.get(image)
        if not faces:
            raise ValueError("No face detected. Use a clear, front-facing enrollment photo.")
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        person_dir = GALLERY_DIR / safe_name
        person_dir.mkdir(parents=True, exist_ok=True)
        target = person_dir / f"enrollment_{len(list(person_dir.iterdir()))+1}.jpg"
        cv2.imwrite(str(target), image)
        self.gallery[safe_name] = self._normalise(face.embedding)
        return safe_name

    def identify(self, embedding):
        if not self.gallery:
            return "unknown", 0.0
        query = self._normalise(embedding)
        scores = {name: float(np.dot(query, emb)) for name, emb in self.gallery.items()}
        name, score = max(scores.items(), key=lambda item: item[1])
        return (name, score) if score >= self.threshold else ("unknown", score)

    def analyse(self, frame):
        faces = self.app.get(frame)
        results = []
        for face in faces:
            name, score = self.identify(face.embedding)
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            results.append({"bbox": [x1,y1,x2,y2], "name": name,
                            "score": score, "det_score": float(face.det_score)})
        return results
