import boto3
import cv2
from ultralytics import YOLO
import time
import numpy as np
from collections import defaultdict
import os

model = YOLO('yolov8m.pt')

STREAM_NAME = "BKS_Office_Parking"
kvs = boto3.client("kinesisvideo")

# Grab the endpoint from GetDataEndpoint
endpoint = kvs.get_data_endpoint(
    APIName="GET_HLS_STREAMING_SESSION_URL",
    StreamName=STREAM_NAME
)['DataEndpoint']

# Grab the HLS Stream URL from the endpoint
kvam = boto3.client("kinesis-video-archived-media", endpoint_url=endpoint)
url = kvam.get_hls_streaming_session_url(
    StreamName=STREAM_NAME,
    PlaybackMode="LIVE"
)['HLSStreamingSessionURL']

# video_path = "rtsp://admin:cctv12345@10.10.2.107/Streaming/Channels/0402"

vcap = cv2.VideoCapture(url)

# Store the track history with timestamps
track_history = defaultdict(lambda: {'positions': [], 'first_seen': time.time()})

# Color definitions in BGR format
colors = {
    'Normal': (0, 255, 0),  # Green
    'Anxious': (0, 255, 255),  # Yellow
    'Suspicious': (0, 0, 255)  # Red
}

# Run YOLOv8 tracking on the frame, persisting tracks between frames
results = model.track(source=url, persist=True, stream=True, classes=0, save=False)

for result in results:
    frame = result.orig_img
    current_time = time.time()

    if result.boxes.is_track:
        # Get the boxes and track IDs
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        frame = result[0].plot(labels=False, probs=False)

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track_info = track_history[track_id]
            track_info['positions'].append((float(x), float(y)))  # x, y center point

            if len(track_info['positions']) > 30:  # retain positions for 90 frames
                track_info['positions'].pop(0)

            # Duration the track has been active
            duration = current_time - track_info['first_seen']
            
            # Determine text and color based on duration
            if duration < 8:
                text = "Normal"
                color = colors['Normal']
            elif 8 <= duration < 11:
                text = "Anxious"
                color = colors['Anxious']
            else:
                text = "Suspicious"
                color = colors['Suspicious']

            # Put dynamic text on the frame
            cv2.putText(frame, f"{text} (ID: {track_id})", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # cv2.imshow("People Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break