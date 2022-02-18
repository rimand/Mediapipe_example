import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# For webcam input:
cap = cv2.VideoCapture(0)


with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        w = image.shape[0]
        h = image.shape[1]
        
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
        
            # for index, landmark in enumerate(results.pose_landmarks.landmark):
            #     # val = results.pose_landmarks;
            #     # num = len(results.pose_landmarks.landmark);
            #     print("index : " + str(index) + " x: "+ str(landmark.x) + " y: "+ str(landmark.y) + " z: "+ str(landmark.z));
            #     # print(val)
            # index = 8
            # print("['INDEX'] -- " + "index : " + str(index) + " x: "+ str(results.left_hand_landmarks.landmark[index].x) + " y: "+ str(results.left_hand_landmarks.landmark[index].y));
            # pos = (int(results.left_hand_landmarks.landmark[index].x*h),int(results.left_hand_landmarks.landmark[index].y*w))
            # print(pos)
            # cv2.circle(image,pos,20,(255,0,0),-1)
            
            print("data :" + str(results.pose_landmarks.landmark))
            
            mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_CONTOURS,landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION,landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        
        
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
