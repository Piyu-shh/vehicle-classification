import cv2
import numpy as np
import keras
from keras.models import load_model

# Load the saved model
model = load_model('model.h5')

# Define class labels
class_labels = {0: 'Bus', 1: 'Car', 2: 'Motorcycle', 3: 'Truck'}

# Function to perform object detection on a frame
def detect_objects(frame):
    # Preprocess the frame
    resized_frame = cv2.resize(frame, (256, 256))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Perform inference
    predictions = model.predict(input_frame)
    
    # Process each prediction
    for prediction in predictions:
        # Get the class index with highest probability
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]

        # Only draw bounding box and label if confidence is above 60%
        if confidence > 0.6:
            # Get bounding box coordinates based on detected object size
            height, width, _ = frame.shape
            box_color = (0, 255, 0)  # Green color for the bounding box
            label = f'{class_labels[predicted_class]}: {confidence:.2f}'
            
            # Calculate bounding box coordinates
            box_x = int(prediction[1] * width)  # Top-left corner x-coordinate
            box_y = int(prediction[0] * height)  # Top-left corner y-coordinate
            box_width = int(prediction[3] * width)  # Box width
            box_height = int(prediction[2] * height)  # Box height

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), box_color, 2)
            cv2.putText(frame, label, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    return frame

# Function to capture video and perform object detection
def detect_objects_realtime():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detect_objects(frame)

        cv2.imshow('Object Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run object detection in real-time
detect_objects_realtime()
