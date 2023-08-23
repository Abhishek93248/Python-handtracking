import cv2
import numpy as np

# Load the pre-trained object detection model for human body
model = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Apply the pre-trained object detection model to detect the human body
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            box = box.astype("int")
            x, y, w, h = box
            roi = frame[y:y+h, x:x+w]

            # Convert the region of interest to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Apply edge detection to detect the edges of the clothes
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Apply Hough Transform to detect the lines
            lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

            # Extract the horizontal and vertical lines
            hor_lines = []
            ver_lines = []
            for line in lines:
                rho, theta = line[0]
                if theta < np.pi/4 or theta > 3*np.pi/4:
                    ver_lines.append(line)
                else:
                    hor_lines.append(line)

            # Calculate the length of the horizontal and vertical lines
            hor_lengths = []
            ver_lengths = []
            for line in hor_lines:
                x1, y1, x2, y2 = cv2.fitLine(line, cv2.DIST_L2, 0, 0.01, 0.01)
                hor_lengths.append(abs(y2 - y1))

            for line in ver_lines:
                x1, y1, x2, y2 = cv2.fitLine(line, cv2.DIST_L2, 0, 0.01, 0.01)
                ver_lengths.append(abs(x2 - x1))

            # Determine the size of the clothes
            hor_length = max(hor_lengths)
            ver_length = max(ver_lengths)
            if hor_length > ver_length:
                if hor_length < 50:
                    size = "S"
                elif hor_length < 70:
                    size = "M"
                else:
                    size = "L"
            else:
                if ver_length < 50:
                    size = "S"
                elif ver_length < 70:
                    size = "M"
                else:
                    size = "L"

            # Display the size of the clothes on the image
            cv2.putText(frame, f"Size: {size}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw the lines on the image
            for line in hor_lines:
                x1, y1, x2, y2 = cv2.fitLine(line, cv2.DIST_L2, 0, 0.01, 0.01)
                cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for line in ver_lines:
                x1, y1, x2, y2 = cv2.fitLine(line, cv2.DIST_L2, 0, 0.01, 0.01)
                cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw a rectangle around the detected human body
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the video
    cv2.imshow("Video", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

