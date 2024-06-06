# STEP 1: Import the necessary modules.
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import mediapipe as mp
import cv2
import numpy as np
import asyncio
from pyartnet import ArtNetNode



# Define constants for drawing
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
channelPan = 0
channelTilt = 0
channelIntensity = 0
channelRGB = 0
channelRGB = 0
frame_width = 1280
frame_height = 720
panSensitivity = 100
panValue = 130
tiltValue = 20
panPrevValue = panValue
tiltPrevValue = tiltValue
PID_test_flag = True

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

def move_camera_to_pos(pan, tilt):

    asyncio.run(pan_camera_to_pos(pan))
    asyncio.run(tilt_camera_to_pos(tilt))

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error_x = 0
        self.error_y = 0
        self.integral_x = 0
        self.integral_y = 0
        self.previous_error_x = 0
        self.previous_error_y = 0

    def calculate_control_signal(self, target_x, target_y, frame_width, frame_height, dt):
        error_x = (target_x - (frame_width / 2)) / (frame_width / 2)
        error_y = (target_y - (frame_height / 2)) / (frame_height / 2)

        proportional_x = self.Kp * error_x
        proportional_y = self.Kp * error_y
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        derivative_x = (error_x - self.previous_error_x) / dt
        derivative_y = (error_y - self.previous_error_y) / dt
        self.previous_error_x = error_x
        self.previous_error_y = error_y
        control_x = proportional_x + (self.Ki * self.integral_x) + (self.Kd * derivative_x)
        control_y = proportional_y + (self.Ki * self.integral_y) + (self.Kd * derivative_y)

        print("Control: ", round(control_x), error_x)
        # Limit the control signal
        if abs(control_x) > 0.2:
            control_x = max(min(control_x, 3), -3)
        else:
            control_x = 0
        if abs(control_y) > 0.5:
            control_y = max(min(control_y, 3), -3)
        else:
            control_y = 0


        return control_x, control_y

def draw_landmarks_on_image(rgb_image, detection_result):

    hand_landmarks_list = detection_result.multi_hand_landmarks
    handedness_list = detection_result.multi_handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        """
        # Draw the hand landmarks.
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())
        """
        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
        y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        landmark_points = [landmark.x * rgb_image.shape[1] for landmark in hand_landmarks.landmark]
        landmark_y_points = [landmark.y * rgb_image.shape[0] for landmark in hand_landmarks.landmark]
        average_x = sum(landmark_points) / len(landmark_points)
        average_y = sum(landmark_y_points) / len(landmark_y_points)

        # Draw the target shape on the image.
        target_size = 15  # Adjust the size of the target shape as desired
        target_thickness = 2  # Adjust the thickness of the target shape as desired
        target_center = (int(average_x), int(average_y))
        target_color = (255, 0, 0)  # Adjust the color of the target shape as desired
        cv2.circle(annotated_image, target_center, target_size, target_color, target_thickness)

        # Calculate the coordinates for the cross in the middle of the target.
        cross_size = 20  # Adjust the size of the cross as desired
        cross_thickness = 2  # Adjust the thickness of the cross as desired
        cross_center = (int(average_x), int(average_y))
        cross_start_x = cross_center[0] - cross_size
        cross_start_y = cross_center[1] - cross_size
        cross_end_x = cross_center[0] + cross_size
        cross_end_y = cross_center[1] + cross_size

        # Draw the cross on the image.
        cross_color = (255, 0, 0)  # Adjust the color of the cross as desired
        cv2.line(annotated_image, (cross_start_x, cross_start_y), (cross_end_x, cross_end_y), cross_color, cross_thickness)
        cv2.line(annotated_image, (cross_start_x, cross_end_y), (cross_end_x, cross_start_y), cross_color, cross_thickness)

        # Draw handedness (left or right hand) on the image.
        # cv2.putText(annotated_image, f"{handedness.classification[0].label}",
        #            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image, average_x, average_y

# Initialize the video capture.
cap = cv2.VideoCapture(1)
cap.set(3, frame_width)
cap.set(4, frame_height)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_EXPOSURE, 10000)

asyncio.run(channels())
move_camera_to_pos(panValue, tiltValue)


with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        ret, image = cap.read()
        image = cv2.flip(image, 0)
        image = cv2.flip(image, 1)
        if not ret:
            break

        # Convert the BGR image to RGB before processing.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find hands.
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:

            annotated_image, average_x, average_y = draw_landmarks_on_image(rgb_image, results)
            cv2.imshow('Annotated Image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            target_x = int(average_x)
            target_y = int(average_y)

            if not PID_test_flag:
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

            else:
                pid_controller = PIDController(Kp=1.5, Ki=0.1, Kd=0.01)
                control_x, control_y = pid_controller.calculate_control_signal(target_x, target_y, frame_width,
                                                                              frame_height, 0.1)
                panValue -= control_x
                tiltValue -= control_y
                panValue = round(panValue)
                tiltValue = round(tiltValue)

            if panPrevValue != panValue:
                print("PanValue: ", panValue)
                asyncio.run(pan_camera_to_pos(panValue))
                panPrevValue = panValue
            if tiltPrevValue != tiltValue:
                asyncio.run(tilt_camera_to_pos(tiltValue))
                tiltPrevValue = tiltValue


        else:
            cv2.imshow('Annotated Image', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
