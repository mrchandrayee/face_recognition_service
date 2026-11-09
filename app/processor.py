import cv2, json
from collections import Counter
from pathlib import Path
from .config import CONFIG, OUTPUT_DIR
from .tracking import CentroidTracker

def process_video(source: str, face_service, output_name="annotated.mp4"):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or CONFIG["video"]["output_fps_fallback"]
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = OUTPUT_DIR / output_name
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    tracker = CentroidTracker(CONFIG["tracking"]["max_disappeared"], CONFIG["tracking"]["max_distance"])
    events, counts, frame_no = [], Counter(), 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        detections = face_service.analyse(frame)
        boxes = [d["bbox"] for d in detections]
        ids = tracker.update(boxes)
        for idx, item in enumerate(detections):
            x1,y1,x2,y2 = item["bbox"]
            track_id = ids.get(idx, -1)
            name = item["name"]
            counts[name] += 1
            color = (30,180,60) if name != "unknown" else (30,120,230)
            label = f"#{track_id} {name} {item['score']:.2f}"
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.rectangle(frame,(x1,max(0,y1-25)),(min(width,x1+220),y1),color,-1)
            cv2.putText(frame,label,(x1+4,max(16,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            if frame_no % max(1,int(fps)) == 0:
                events.append({"time_seconds": round(frame_no/fps,2), "track_id":track_id,
                               "match":name, "similarity":round(item["score"],4)})
        writer.write(frame)
        frame_no += 1
    cap.release(); writer.release()
    report = {"frames_processed":frame_no,"fps":fps,"duration_seconds":round(frame_no/fps,2),
              "match_frame_counts":dict(counts),"events":events,"output":str(output_path)}
    (OUTPUT_DIR/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
