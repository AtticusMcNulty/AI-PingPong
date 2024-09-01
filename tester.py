from video_utils import VideoUtils
import numpy as np
from ultralytics import YOLO
import supervision as sv
import pandas as pd
import cv2


def main():
    video_utils = VideoUtils()

    frames = np.array(video_utils.read_video("input_videos/video.mp4"))

    detectedFrames = []
    counter = 1
    model = YOLO("models/best.pt")

    for frame in frames:
        detectedFrame = model.predict(frame, conf=0.1)
        detectedFrames.append(detectedFrame[0])

        counter += 1
        print(f"Detections: {counter}/{len(frames)}")

    tracks = {"Ball": []}
    tracker = sv.ByteTrack()

    for frameNum, detectedFrame in enumerate(detectedFrames):
        supervisionFrame = sv.Detections.from_ultralytics(detectedFrame)
        trackedFrame = tracker.update_with_detections(supervisionFrame)

        tracks["Ball"].append({})

        for obj in trackedFrame:
            bbox = obj[0].tolist()
            className = obj[5]["class_name"]

            tracks[className][frameNum][1] = {"bbox": bbox}

    ball_positions = [x.get(1, {}).get("bbox", []) for x in tracks["Ball"]]
    df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
    df_ball_positions = df_ball_positions.interpolate()
    df_ball_positions = df_ball_positions.bfill()
    ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

    annotatedFrames = []

    for frameNum, frame in enumerate(frames):
        ballTracks = tracks["Ball"][frameNum]

        for ballId, ball in ballTracks.items():
            bbox = ball["bbox"]
            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (255, 0, 0),
                3,
            )

        annotatedFrames.append(frame)

    video_utils.write_video(np.array(annotatedFrames))


if __name__ == "__main__":
    main()
