import cv2


# Create our body classifier
body_classifier= cv2.CascadeClassifier('haarcascade_fullbody.xml') 

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while(True):
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Pass frame to our body classifier
    body = body_classifier.detectMultiScale(gray, 1.2, 5)
    
    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    

    # Define a video capture object
        cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(25) == 32:
        break
 
 
cap.release()


cv2.destroyAllWindows()

  



