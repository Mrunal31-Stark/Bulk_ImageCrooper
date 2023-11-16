import cv2
import os

# Folder containing the images
folder_path = "C:/Users/Acer/Desktop/Data Sample"  # Replace with the actual path to your folder
f2 = "C:/Users/Acer/Desktop/ccrop"
# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Process each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png' , '.gif')):
        image_path = os.path.join(folder_path, filename)
        a = cv2.imread(image_path)

        # Detect faces in the current image
        faces = face_cascade.detectMultiScale(a, scaleFactor=1.05, minNeighbors=3)

        if len(faces) > 0:
            # Find the largest face (central face)
            largest_face = max(faces, key=lambda x: x[2] * x[3])

            x, y, w, h = largest_face
            
            face = a[y:y + h, x:x + w]

            # Save the cropped face and the full image in the same folder
            cv2.imwrite(os.path.join(f2, 'crop_' + filename), face)
            
        else:
            print(f"No faces detected in {filename}")
