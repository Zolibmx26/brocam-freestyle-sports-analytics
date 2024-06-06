# Import necessary modules
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import threading
import asyncio
import time
import torch
import json
from collections import defaultdict
from pyartnet import ArtNetNode
from ultralytics import YOLO
from flask import Flask, render_template, jsonify, request

# Initialize Flask app
app = Flask(__name__)

# Global variables
channelPan = 0
channelTilt = 0
channelIntensity = 0
channelRGB = 0

posPan = 0
posTilt = 0

panValue = 30
tiltValue = 20
zoomValue = 0
panPrevValue = 0
tiltPrevValue = 0
panSensitivity = 300

panStartValue = 0
tiltStartValue = 0
zoomStartValue = 0

current_frame = cv2.imread('./work_in_progress.jpg')
annotated_frame = cv2.imread('./work_in_progress.jpg')
current_recorded_frame = cv2.imread('./work_in_progress.jpg')

start_timer = 0
recording_timer_start = 0
recording_length = 15

followed_track_id = 0
recording_flag = False
recording_stopped_flag = False
skeleton_flag = False
handisup_flag = False

prev_target_x = 0
prev_target_y = 0

frame_lock = threading.Lock()
shared_variable_lock = threading.Lock()
frame_loaded_event = threading.Event()

# Initialize YOLO model
model_person = YOLO('weights/yolov8n.pt')
model_keypoint = YOLO('weights/yolov8n-pose.pt')

camera_port = 1
file_counter = 1

# Initialize OpenCV camera capture
cap = cv2.VideoCapture(camera_port)
frame_width = 1920
frame_height = 1080

cap.set(3, frame_width)
cap.set(4, frame_height)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_EXPOSURE, 10000)

current_mode = 'Auto'

success_recorded_frame = False


default_settings = {
    'pan': panValue,
    'tilt': tiltValue,
    'zoom': zoomValue,
    'pan_sensitivity': panSensitivity,
    'recording_time': recording_length
}


# Define route for the main page
@app.route('/')
def index():
    return render_template('index3.html', default_settings=default_settings)


@app.route('/control_pan', methods=['POST'])
def control_pan():
    global panValue
    data = request.get_json()
    posPan = data['panLevel']

    with shared_variable_lock:
        panValue = int(posPan)

    print(f"Received pan command: Pan position: {posPan}")
    asyncio.run(pan_camera_to_pos(int(posPan)))

    return 'Pan data received successfully'


@app.route('/control_tilt', methods=['POST'])
def control_tilt():
    global tiltValue
    data = request.get_json()
    posTilt = data['tiltLevel']

    with shared_variable_lock:
        tiltValue = int(posTilt)

    print(f"Received tilt command: Tilt position: {posTilt}")
    asyncio.run(tilt_camera_to_pos(int(posTilt)))

    return 'Tilt data received successfully'


@app.route('/control_zoom', methods=['POST'])
def control_zoom():

    data = request.get_json()
    zoom_level = data['zoomLevel']

    # Process the received zoom data (e.g., adjust camera zoom)
    print(f"Received zoom command: Zoom level: {zoom_level}")
    cap.set(cv2.CAP_PROP_ZOOM, 100 + int(zoom_level))

    # Return a response
    return 'Zoom data received successfully'


@app.route('/control_intensity', methods=['POST'])
def control_intensity():

    data = request.get_json()
    intensity = data['intensityLevel']

    print(f"Received intensity command: Intensity: {intensity}")
    asyncio.run(set_intensity(int(intensity)))

    # Return a response
    return 'Intensity data received successfully'


@app.route('/control_rgb', methods=['POST'])
def control_rgb():

    data = request.get_json()
    rgb = data['rgbLevel']

    print(f"Received RGB command: RGB: {rgb}")
    asyncio.run(set_RGB(int(rgb)))

    # Return a response if needed
    return 'RGB data received successfully'


# Route for getting the image frame
@app.route('/get_frame')
def get_frame():
    global current_frame, annotated_frame
    global current_mode
    global current_recorded_frame, success_recorded_frame

    with frame_lock:
        frame = annotated_frame.copy()

    with frame_lock:
        frame_recorded = current_recorded_frame.copy()

    if current_mode == 'Manual':
        _, img_encoded = cv2.imencode('.jpg', frame)
        return img_encoded.tobytes(), 200, {'Content-Type': 'image/jpeg'}
    elif current_mode == 'Auto':
        _, img_encoded = cv2.imencode('.jpg', frame_recorded)
        return img_encoded.tobytes(), 200, {'Content-Type': 'image/jpeg'}


@app.route('/get_live_frame')
def get_live_frame():
    global current_frame, annotated_frame

    with frame_lock:
        frame = current_frame.copy()

    _, img_encoded = cv2.imencode('.jpg', frame)
    return img_encoded.tobytes(), 200, {'Content-Type': 'image/jpeg'}


@app.route('/update_mode', methods=['POST'])
def update_mode():
    global current_mode
    data = request.json  # Get the JSON data sent from the HTML page
    new_mode = data.get('newState')  # Get the new mode from the JSON data
    if new_mode in ['Auto', 'Manual']:  # Check if the new mode is valid
        current_mode = new_mode  # Update the current mode
        return 'Mode updated successfully', 200  # Respond with success message and status code 200
    else:
        return 'Invalid mode', 400  # Respond with error message and status code 400 (Bad Request)


# Define a route to handle saving settings
@app.route('/save_settings', methods=['POST'])
def save_settings():

    global panStartValue, tiltStartValue, zoomStartValue, panSensitivity, recording_length
    # Get the settings data from the request body
    settings = request.json

    with shared_variable_lock:
        # Access individual settings values
        panStartValue = int(settings['pan'])
        tiltStartValue = int(settings['tilt'])
        zoomStartValue = int(settings['zoom'])
        panSensitivity = int(settings['pan_sensitivity'])
        recording_length = int(settings['recording_time'])

    # Process the settings data
    print("Received settings:", settings)
    save_global_variables()
    # Return response
    return jsonify({'message': 'Settings saved successfully'})


# Define a function to save global variables to a JSON file
def save_global_variables():
    global panStartValue, tiltStartValue, zoomStartValue, panValue, tiltValue, panSensitivity, recording_length, file_counter
    with shared_variable_lock:
        data = {
            'panStartValue': panStartValue,
            'tiltStartValue': tiltStartValue,
            'zoomStartValue': zoomStartValue,
            'panSensitivity': panSensitivity,
            'recording_length': recording_length,
            'file_counter': file_counter
        }
    # Open a file in write mode and write the data in JSON format
    with open('global_variables.json', 'w') as f:
        json.dump(data, f)

    with shared_variable_lock:
        panValue = panStartValue
        tiltValue = tiltStartValue


# Function to load global variables from the JSON file
def load_global_variables():
    global panStartValue, tiltStartValue, panValue, tiltValue, zoomStartValue, panSensitivity, recording_length, file_counter
    global default_settings
    try:
        with open('global_variables.json', 'r') as f:
            data = json.load(f)
            panStartValue = data.get('panStartValue', panStartValue)
            tiltStartValue = data.get('tiltStartValue', tiltStartValue)
            zoomStartValue = data.get('zoomStartValue', zoomStartValue)
            panSensitivity = data.get('panSensitivity', panSensitivity)
            recording_length = data.get('recording_length', recording_length)
            file_counter = data.get('file_counter', file_counter)

            default_settings = {
                'pan': panStartValue,
                'tilt': tiltStartValue,
                'zoom': zoomStartValue,
                'pan_sensitivity': panSensitivity,
                'recording_time': recording_length
            }

            panValue = panStartValue
            tiltValue = tiltStartValue

    except FileNotFoundError:
        # Handle the case when the file doesn't exist
        pass


async def channels():
    node = ArtNetNode('10.0.0.1', 6454)
    universe = node.add_universe(1)
    global channelPan
    global channelTilt
    global channelIntensity
    global channelRGB

    channelPan = universe.add_channel(start=1, width=1)
    channelTilt = universe.add_channel(start=2, width=1)
    channelIntensity = universe.add_channel(start=3, width=1)
    channelRGB = universe.add_channel(start=10, width=1)


async def pan_camera_to_pos(x):

    if x > 255:
        x = 255
    if x < 0:
        x = 0

    channelPan.add_fade([x], 0)
    await channelPan


async def tilt_camera_to_pos(x):

    if x > 255:
        x = 255
    if x < 0:
        x = 0

    channelTilt.add_fade([x], 0)
    await channelTilt


async def set_intensity(intensity):

    channelIntensity.add_fade([intensity], 2)
    await channelIntensity


async def set_RGB(rgb):
    channelRGB.add_fade([rgb], 2)
    await channelRGB


def update_LED_status():

    global recording_flag, skeleton_flag, handisup_flag
    global current_mode
    with shared_variable_lock:
        if current_mode == 'Auto':
            asyncio.run(set_intensity(100))
            if recording_flag:
                asyncio.run(set_RGB(50))
            elif not recording_flag:
                if skeleton_flag:
                    if handisup_flag:
                        asyncio.run(set_RGB(90))
                    elif not handisup_flag:
                        asyncio.run(set_RGB(190))
                elif not skeleton_flag:
                    asyncio.run(set_RGB(30))


def move_camera_to_pos(pan, tilt, zoom):

    global cap
    asyncio.run(pan_camera_to_pos(pan))
    asyncio.run(tilt_camera_to_pos(tilt))
    # cap.set(cv2.CAP_PROP_ZOOM, 100+zoom)


def capture_frames():
    global current_frame
    global recording_flag, recording_stopped_flag
    global panStartValue, tiltStartValue, zoomStartValue
    global cap
    global file_counter
    # codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    outputFile = cv2.VideoWriter(f'Recordings\CAPTURE1{str(file_counter)}.mp4', codec, 30, (frame_width, frame_height))

    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)

        with frame_lock:
            current_frame = frame.copy()

            if not frame_loaded_event.is_set():
                frame_loaded_event.set()

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        if recording_flag:
            outputFile.write(current_frame)

        if recording_stopped_flag:
            time.sleep(2)
            file_counter += 1
            outputFile = cv2.VideoWriter(f'Recordings\CAPTURE1{str(file_counter)}.mp4', codec, 30, (frame_width, frame_height))
            save_global_variables()
            move_camera_to_pos(panStartValue, tiltStartValue, zoomStartValue)
            # cap.set(cv2.CAP_PROP_ZOOM, 100 + zoomStartValue)
            recording_stopped_flag = False

    cap.release()
    cv2.destroyAllWindows()


def capture_recorded_frames():
    global current_recorded_frame, success_recorded_frame, file_counter
    cap_recording = cv2.VideoCapture(f'Recordings\CAPTURE1{str(file_counter-1)}.mp4')
    cap_recording.set(cv2.CAP_PROP_FPS, 30)

    while True:

        if not success_recorded_frame:
            print('File Counter: ', file_counter)
            cap_recording = cv2.VideoCapture(f'Recordings\CAPTURE1{str(file_counter-1)}.mp4')

        success_recorded_frame, current_recorded_frame = cap_recording.read()

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap_recording.release()


def process_frames():
    global current_frame, annotated_frame

    frame_loaded_event.wait()

    while True:
        with frame_lock:
            frame = current_frame.copy()
            # annotated_frame = current_frame.copy()

        if recording_flag:
            person_tracker(frame)
        else:
            pose_tracker(frame)


def person_tracker(frame):

    global recording_length
    global recording_flag
    global recording_stopped_flag
    global annotated_frame

    start_fps_time = time.perf_counter()

    results = model_person.track(frame,
                                 conf=0.3,
                                 persist=True,
                                 classes=0)

    stop_fps_time = time.perf_counter()
    fps = 1 / (stop_fps_time - start_fps_time)

    # Visualize the results on the frame
    frame = results[0].plot()
    cv2.putText(frame, f'FPS:: {fps:.2f}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    display_frame(frame)
    tracker_processing(results)

    recording_timer_stop = time.perf_counter()
    recording_timer = recording_timer_stop - recording_timer_start

    if recording_timer > recording_length:
        recording_flag = False
        recording_stopped_flag = True
        with shared_variable_lock:
            cap.set(cv2.CAP_PROP_ZOOM, 100 + zoomStartValue)
        update_LED_status()



def tracker_processing(results):

    global followed_track_id
    global prev_target_x
    global prev_target_y
    global panValue
    global tiltValue
    global panPrevValue
    global tiltPrevValue
    global panSensitivity

    min_distance_xy = 5000
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x, y, w, h = box.xywh[0]
            x1, y1, x2, y2 = int(x - (w / 2)), int(y - (h / 2)), int(x + (w / 2)), int(y + (h / 2))
            target_x = (x1 + x2) / 2
            target_y = (y1 + y2) / 2
            track_id = box.id
            distance_x = abs(target_x - prev_target_x)
            distance_y = abs(target_y - prev_target_y)
            distance_xy = distance_x + distance_y

            if distance_xy <= min_distance_xy:
                min_distance_xy = distance_xy
                followed_track_id = track_id

    # Follow person with specified ID
    for result in results:
        # detection
        boxes = result.boxes
        for box in boxes:

            x, y, w, h = box.xywh[0]
            x1, y1, x2, y2 = int(x - (w / 2)), int(y - (h / 2)), int(x + (w / 2)), int(y + (h / 2))
            target_x = (x1 + x2) / 2
            target_y = (y1 + y2) / 2
            track_id = box.id

            if track_id == followed_track_id:
                prev_target_x = target_x
                prev_target_y = target_y
                if (target_x != 0):
                    panStep = round(abs(target_x - (frame_width / 2)) / panSensitivity)
                    print("panStep: ", panStep)
                    if (target_x < frame_width / 2 - frame_width * 0.1):
                        panValue += panStep
                    elif (target_x > frame_width / 2 + frame_width * 0.1):
                        panValue -= panStep
                    else:
                        panValue += 0
                else:
                    panValue += 0

                if (target_y != 0):
                    tiltStep = round(abs(target_y - (frame_height / 2)) / 100)
                    print("tiltStep: ", tiltStep)
                    if (target_y < frame_height / 2 - frame_height * 0.2):
                        tiltValue += tiltStep
                    elif (target_y > frame_height / 2 + frame_height * 0.2):
                        tiltValue -= tiltStep
                    else:
                        tiltValue += 0
                else:
                    tiltValue += 0

                if panPrevValue != panValue:
                    print("PanValue: ", panValue)
                    asyncio.run(pan_camera_to_pos(panValue))
                    panPrevValue = panValue
                if tiltPrevValue != tiltValue:
                    # asyncio.run(tilt_camera_to_pos(tiltValue))
                    tiltPrevValue = tiltValue


def pose_tracker(frame):

    global annotated_frame
    start_fps_time = time.perf_counter()
    # Perform object tracking with YOLOv8
    results = model_keypoint.track(frame,
                                   conf=0.5,
                                   persist=True,
                                   tracker='bytetrack.yaml',
                                   verbose=False)[0]

    frame = results.plot()

    stop_fps_time = time.perf_counter()
    fps = 1 / (stop_fps_time - start_fps_time)

    cv2.putText(frame, f'FPS:: {fps:.2f}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    display_frame(frame)
    process_pose_keypoints(results)


def process_pose_keypoints(results):

    track_history = defaultdict(lambda: [])
    drop_counting = defaultdict(lambda: 0)
    max_miss = 4

    # Extract bounding boxes and keypoints from the detection result
    boxes = results.boxes.xywh.cpu()
    keypoints = results.keypoints.data

    track_ids = results.boxes.id
    if track_ids is None:
        track_ids = []
    else:
        track_ids = track_ids.int().cpu().tolist()

    # Remove old tracks and update history
    diff = list(set(list(set(track_history.keys()))).difference(track_ids))
    for d in diff:
        if drop_counting[d] > max_miss:
            del drop_counting[d]
            del track_history[d]
        else:
            drop_counting[d] += 1

    # Process each detected object
    track_ids_conform_frame_num = []
    poseTrackResult = []
    boxess = []
    for box, track_id, keypoint in zip(boxes, track_ids, keypoints):
        track = track_history[track_id]
        track.append(keypoint.unsqueeze(0))

        poseTrackResult.append(torch.cat(track).cpu().unsqueeze(0))
        track_ids_conform_frame_num.append(track_id)
        boxess.append(box)

    # Plot keypoints and draw bounding boxes on the image
    for resultIdx, track_id in enumerate(track_ids_conform_frame_num):
        current_kpt = poseTrackResult[resultIdx][0, -1, :, :].numpy().flatten()

        x, y, w, h = boxess[resultIdx]
        x1, y1, x2, y2 = int(x - (w / 2)), int(y - (h / 2)), int(x + (w / 2)), int(y + (h / 2))

        hand_is_up_trigger(current_kpt, track_id, x1, y1, x2, y2)


def hand_is_up_trigger(current_kpt, track_id, x1, y1, x2, y2):

    global start_timer
    global followed_track_id
    global recording_flag, skeleton_flag, handisup_flag
    global recording_timer_start
    global prev_target_x
    global prev_target_y

    if current_kpt[3 * 0] and current_kpt[3 * 10]:

        skeleton_flag = True
        # update_LED_status()
        hand_above_nose = current_kpt[3 * 0 + 1] - current_kpt[3 * 10 + 1]

        if hand_above_nose > 0 and (not start_timer):
            start_timer = time.perf_counter()
            followed_track_id = track_id

        if hand_above_nose <= 0 and followed_track_id == track_id:
            start_timer = 0

        stop_timer = time.perf_counter()

        if start_timer:
            handisup_flag = True
            update_LED_status()
        else:
            handisup_flag = False
            update_LED_status()

        if (stop_timer - start_timer) > 3 and start_timer:

            start_timer = 0
            recording_flag = True
            update_LED_status()
            with shared_variable_lock:
                cap.set(cv2.CAP_PROP_ZOOM, 150)

            followed_track_id = track_id
            prev_target_x = (x1 + x2) / 2
            prev_target_y = (y1 + y2) / 2
            recording_timer_start = time.perf_counter()


    else:
        skeleton_flag = False
        update_LED_status()


def display_frame(frame):

    global annotated_frame
    # cv2.resize(frame, (960, 540))
    with frame_lock:
        annotated_frame = frame.copy()

    # cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    # cv2.imshow('Result', frame)
    # cv2.waitKey(1)  # Keep the window open


def start_server():

    app.run(host='0.0.0.0')


def main():
    # Start thread for capturing frames
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()

    # Start thread for processing frames
    process_thread = threading.Thread(target=process_frames)
    process_thread.daemon = True
    process_thread.start()

    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    capture_recorded_thread = threading.Thread(target=capture_recorded_frames)
    capture_recorded_thread.daemon = True
    capture_recorded_thread.start()
    # Wait for threads to finish
    capture_thread.join()
    capture_recorded_thread.join()
    process_thread.join()
    server_thread.join()


if __name__ == '__main__':
    load_global_variables()
    asyncio.run(channels())
    move_camera_to_pos(panStartValue, tiltStartValue, zoomStartValue)
    main()