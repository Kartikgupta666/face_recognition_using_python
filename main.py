import cv2 as cv

# Load the face cascade classifier
face_cap = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from the default camera
video_capture = cv.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, img = video_capture.read()
    
    # If frame reading is successful, proceed
    if ret:
        # Convert the frame to grayscale
        col = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cap.detectMultiScale(
            col,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv.CASCADE_SCALE_IMAGE
        )

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display the frame with the detected faces
        cv.imshow("Display window", img)
        
        # Exit the loop when 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to read frame from video capture")
        break

# Release the video capture and close all windows
video_capture.release()
cv.destroyAllWindows()
