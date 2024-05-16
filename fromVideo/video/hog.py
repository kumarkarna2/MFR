import cv2

# Load video
cap = cv2.VideoCapture("video/Assault001_x264.mp4")

# Initialize HOG detector
hog = cv2.HOGDescriptor()


# Process each frame
while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect people
    (rects, weights) = hog.detectMultiScale(
        gray, winStride=(8, 8), scale=1.05, minSize=(30, 30)
    )

    # Draw bounding boxes
    for x, y, w, h in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
