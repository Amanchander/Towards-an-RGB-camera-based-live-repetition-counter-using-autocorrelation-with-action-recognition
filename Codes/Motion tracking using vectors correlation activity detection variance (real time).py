
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import glob
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tensorflow import keras
import pandas as pd
from scipy import stats as st
import statistics
         
from scipy.stats import differential_entropy
from scipy.spatial.distance import cdist
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

counting_results = []  
 
desired_activity = 'standing_shoulder_extension'


headers = ['0_x', '0_y', '0_z', '0_vis', '1_x', '1_y', '1_z', '1_vis', '2_x', '2_y', '2_z', '2_vis', '3_x', '3_y', '3_z', '3_vis', '4_x', '4_y', '4_z', '4_vis', '5_x', '5_y', '5_z', '5_vis', '6_x', '6_y', '6_z', '6_vis', '7_x', '7_y', '7_z', '7_vis', '8_x', '8_y', '8_z', '8_vis', '9_x', '9_y', '9_z', '9_vis', '10_x', '10_y', '10_z', '10_vis', '11_x', '11_y', '11_z', '11_vis', '12_x', '12_y', '12_z','12_vis','13_x', '13_y', '13_z', '13_vis', '14_x', '14_y', '14_z', '14_vis', '15_x', '15_y', '15_z', '15_vis', '16_x', '16_y', '16_z', '16_vis', '17_x', '17_y', '17_z', '17_vis', '18_x', '18_y', '18_z', '18_vis', '19_x', '19_y', '19_z', '19_vis', '20_x', '20_y', '20_z', '20_vis', '21_x', '21_y', '21_z', '21_vis', '22_x', '22_y', '22_z', '22_vis', '23_x', '23_y', '23_z', '23_vis', '24_x', '24_y', '24_z', '24_vis', '25_x', '25_y', '25_z', '25_vis', '26_x', '26_y', '26_z', '26_vis', '27_x', '27_y', '27_z', '27_vis', '28_x', '28_y', '28_z', '28_vis', '29_x', '29_y', '29_z', '29_vis', '30_x', '30_y', '30_z', '30_vis', '31_x', '31_y', '31_z', '31_vis', '32_x', '32_y', '32_z', '32_vis']

save_video = True

activitiesName = ['hurdle_step', 'idle', 'inline_lunge', 'jump', 'run' ,'side_lunge',
'sit_to_stand' ,'squats', 'standing_shoulder_abduction',
'standing_shoulder_extension',
'standing_shoulder_internal_external_rotation', 
'standing_shoulder_scapation']



save_model_path = "_path_to_saved_mdoel_"


# Define the codec and create VideoWriter object
output_path = "_path_to_your_output_videos_"  # To save output videos



from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import register_keras_serializable


try:
    TransformerBlock
except NameError:

    @register_keras_serializable()
    class TransformerBlock(Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
            super(TransformerBlock, self).__init__(**kwargs)
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.rate = rate
    
            self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = Sequential([
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim),
            ])
            self.layernorm1 = LayerNormalization(epsilon=1e-6)
            self.layernorm2 = LayerNormalization(epsilon=1e-6)
            self.dropout1 = Dropout(rate)
            self.dropout2 = Dropout(rate)
    
        def call(self, inputs, training):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)
    
        def get_config(self):
            config = super().get_config()
            config.update({
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            })
            return config




myModel = load_model(save_model_path + '/transformer_model.h5',
                   custom_objects={'TransformerBlock': TransformerBlock})



# 🔹 Function to compute entropy over sliding windows
def compute_entropy(signal, window_size=50):
    entropy_values = []
    for i in range(len(signal) - window_size):
        window = signal[i:i + window_size]
        entropy_values.append(differential_entropy(window))
    return np.array(entropy_values)





def pearson_autocorrelation(signal):
    N = len(signal)
    mean_signal = np.mean(signal)
    autocorr = []

    for lag in range(N):
        # Pearson correlation formula
        numerator = np.sum((signal[:N-lag] - mean_signal) * (signal[lag:] - mean_signal))
        denominator = np.sqrt(np.sum((signal[:N-lag] - mean_signal)**2)) * np.sqrt(np.sum((signal[lag:] - mean_signal)**2))
        
        if denominator == 0:
            autocorr.append(0)
        else:
            autocorr.append(numerator / denominator)
    
    return np.array(autocorr)





desired_width = 380   # Set your desired width
desired_height = 448  # Set your desired height




def ResizeWithAspectRatio(image, desiredFrameWidth, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    # Compute the scaling factor based on desired width
    r = desiredFrameWidth / float(w)
    desired_width = desiredFrameWidth
    desired_height = int(h * r)

    dim = (desired_width, desired_height)

    return cv2.resize(image, dim, interpolation=inter)



def get_direction(idx, x_projection, y_projection, projection_45, neg45_projection):
    
    direction = +1
    
    if idx == 0: # Movement along x-axis
    
        if x_projection > 0:
            direction = +1
            
        else: 
            direction = -1
            
        # print("Along x ", " ", direction)    
            
        
    elif idx == 1: # Movement along y-axis
            
        if y_projection > 0:
            direction = +1
            
        else: 
            direction = -1
            
        # print("Along y ", " ", direction)   
    
    
    elif idx == 2: # Movement along line y = x
            
        if projection_45 > 0:
            direction = +1
            
        else: 
            direction = -1
            
        # print("Along 45 ", " ", direction)   
    
    
    else: 
            
        if neg45_projection > 0:
            direction = +1
            
        else: 
            direction = -1
            
        # print("Along neg45 ", " ", direction)   
        
    return direction


    
def process_video():
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    mpDrawing = mp.solutions.drawing_utils  # Setup mediapipe
    
    cap = cv2.VideoCapture(0)
 
    TotalFramesInVideo = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
    print("The length of the cap is ", TotalFramesInVideo)
    
    repetition_count = 0

    
    peaks, properties = [], []
    
    wait_idx = 0
    
    dynamic_prominence_ratio = 0.8
    
    min_distance = 10
    
    peak_detect_threshold = 0.

    if desired_activity == 'run' or desired_activity == 'jump':
        alpha = 0.4  # Adjust for smoother motion tracking 
        min_distance = 2 
        peak_detect_threshold = 0.4
    
    else:
        alpha = 0.1
        min_distance = 10


    
    frame_count = 0
    
    landmarksArray = []
    columnArray    = []
    motion_distance = []
    motion_history_full = []

    
    avg_dx = 0 
    avg_dy = 0
    
    smoothed_dx = 0
    smoothed_dy = 0 
    autocorr = 0
    autocorr_array = []
    

    motion_history = []
    motion_history = deque(maxlen=8) 
    
    avg_dx, avg_dy = 0, 0

      
    frame_count = 0
    total_dx, total_dy = 0, 0
    motion_amplitude = 0
    prev_x, prev_y = 0, 0

    prev_landmarks = None
    
    motion_history_x = deque(maxlen=(100))
    motion_history_y = deque(maxlen=(100))

    
    motion_history_45 = deque(maxlen=(100))
    motion_history_neg45 = deque(maxlen=(100))
    
    variance_x = 0
    variance_y = 0
    variance_45 = 0
    variance_neg45 = 0
    

    
    prev_peak_array_length = 0
    

    
    noOfFrameSize = 16
    noOfFeatures  = 132
    lastLandmarkPoint = 33 
    
    
    motion_amplitude_threshold = 8

    motion_distance_arr_limit = 800
    

    
    variance_array = []
    resultIndex = 0

    overall_direction = +1
    
    
    desiredFrameWidth = 400
    
    prev_repetition_count = 0

    
    for i in range(0, lastLandmarkPoint*4):
        columnArray.append(i)
    
    landMarksDf = pd.DataFrame(columns = columnArray)
    
    first_actvity_detected = False
    

    
    if save_video:
        # fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 8
        # 🔹 Save as MP4 instead of AVI
        output_video_path = os.path.join(output_path, desired_activity + "_real_time.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (desired_width, desired_height))

    
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
            
        while cap.isOpened():
            ret, frame = cap.read()
            # frame = cv2.flip(frame, 0)
            # frame = cv2.flip(frame, 1)
            
            if not ret:
                break
            
    
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            
            landmarksArray    = []
 
            
            if results.pose_landmarks:
                
                curr_landmarks = results.pose_landmarks.landmark 
    
                
                try: # Try and except are used because we are not going to get the landmarks all the time may be due to interruptions in the camera view 
                    capturedLandmarks = results.pose_world_landmarks.landmark # Extract all the landmarks
                    
                    for i in range(0, lastLandmarkPoint):
                                          
                        landmarksArray.append(capturedLandmarks[i].x) 
                        landmarksArray.append(capturedLandmarks[i].y)
                        landmarksArray.append(capturedLandmarks[i].z)
                        landmarksArray.append(capturedLandmarks[i].visibility)
                                             
                             
                except: 
                    pass;
                
                
                landmarksArray = [landmarksArray]
                addDf = pd.DataFrame(landmarksArray)
                
                if len(landMarksDf) < noOfFrameSize:  
                    landMarksDf = pd.concat([landMarksDf, addDf], ignore_index=True, axis =0)

                
                elif len(landMarksDf) >= noOfFrameSize:    
                    landMarksDf = pd.concat([landMarksDf, addDf], ignore_index=True, axis =0)
                    landMarksDf.drop(axis = 0, index = 0, inplace = True)
                    landMarksDf = landMarksDf.reset_index(drop =True)
                    inputArray = np.array(landMarksDf) 
                    
       
                    result = myModel(inputArray.reshape(1,noOfFrameSize, noOfFeatures,1))            # print(np.argmax(result))
                    
                    resultIndex = np.argmax(result)
                    # print(activitiesName[resultIndex])
                        
                # print("Input array length : ", len(landMarksDf))    
                
        
                if prev_landmarks:
                    
                    total_dx, total_dy = 0, 0
                    count = 0
        
        
                    for i, landmark in enumerate(curr_landmarks):
                        x, y = (landmark.x * frame.shape[1]), (landmark.y * frame.shape[0])
                        prev_x, prev_y = (prev_landmarks[i].x * frame.shape[1]), (prev_landmarks[i].y * frame.shape[0])
        
                        dx, dy = x - prev_x, y - prev_y
                        
                        
                        delta_limit = 60
                        
                        if dx > delta_limit:
                            dx = delta_limit 
                            
                        elif dx < -delta_limit:
                            dx = -delta_limit
                            
                            
                        if dy > delta_limit:
                            dy = delta_limit
                            
                        elif dy < -delta_limit:
                            dy = -delta_limit
                        
                        
                        
                        # print("dx : ", dx, " dy : ", dy)
                        
                        total_dx += dx
                        total_dy += dy
                        count += 1
        
                        # Draw motion vectors
                        cv2.arrowedLine(frame, (int(prev_x), int(prev_y)), (int(x), int(y)), (0, 255, 0), 1)

     
    
    
                    if count > 0:
                        # Compute resultant motion
                        resultant_dx, resultant_dy = total_dx, total_dy
                    
                        # Apply Exponential Moving Average
                        smoothed_dx = alpha * resultant_dx + (1 - alpha) * smoothed_dx
                        smoothed_dy = alpha * resultant_dy + (1 - alpha) * smoothed_dy
                    
                        # Store in motion history
                        motion_history.append((smoothed_dx, smoothed_dy))
                     

                        
                        # Compute the combined vector's amplitude
                        motion_amplitude = np.sqrt(smoothed_dx**2 + smoothed_dy**2)
                        

    
                        motion_history_x.append(smoothed_dx)
                        motion_history_y.append(smoothed_dy)
                

                
                        smoothed_45 = (smoothed_dx + smoothed_dy) / np.sqrt(2)
                        smoothed_neg45 = (smoothed_dx - smoothed_dy) / np.sqrt(2)
                        
                        
                        # Store motion along diagonal axes
                        motion_history_45.append(smoothed_45)
                        motion_history_neg45.append(smoothed_neg45)
                        
                        
                        # print(smoothed_dx, smoothed_dy, smoothed_45, smoothed_neg45)
                       
                        
                        
                        if len(motion_history_x) > 3:
                            # Compute variances
                            variance_x = statistics.variance(motion_history_x)  # X-axis
                            variance_y = statistics.variance(motion_history_y)  # Y-axis
                            variance_45 = statistics.variance(motion_history_45)  # 45° axis
                            variance_neg45 = statistics.variance(motion_history_neg45)  # -45° axis
                        
                            # print(f"X variance: {variance_x}, Y variance: {variance_y}, 45-degree variance: {variance_45}, -45-degree variance: {variance_neg45}")
                            
                            
                            variance_array = np.array([variance_x, variance_y, variance_45, variance_neg45])
                            max_variance_idx = np.argmax(variance_array)
                            # print(variance_array, "   idx :", max_variance_idx)
                            
                            
                            overall_direction = get_direction(max_variance_idx, smoothed_dx, smoothed_dy, smoothed_45, smoothed_neg45)
                                

                        
                        # Determine motion direction using normalized values
                        if motion_amplitude > motion_amplitude_threshold:
                            norm_dx = smoothed_dx / motion_amplitude
                            norm_dy = smoothed_dy / motion_amplitude
                            
                                                                                
                            if len(peaks) == 0 and (desired_activity == activitiesName[resultIndex]) and first_actvity_detected == False:
                                repetition_count+=1
                                first_actvity_detected = True
                                
                            
                        else:
                            norm_dx, norm_dy = 0, 0  # Avoid division by zero

    
                        motion_distance.append(motion_amplitude * overall_direction)
                        motion_history_full.append(motion_amplitude * overall_direction)

    
                        
                        if frame_count > 2 and wait_idx <=0:  
                        
                            if len(motion_distance) > motion_distance_arr_limit:
                                motion_distance.pop(0)
                        
    
                            # Compute the autocorrelation of the motion distance array
                            autocorr = pearson_autocorrelation(np.array(motion_distance))
                            
                                                       
                            # Normalize the autocorrelation
                            autocorr = autocorr / np.max(autocorr)   
    
                            
                            prominence_threshold = np.max(autocorr) * dynamic_prominence_ratio  
                            
                        
                            
                            if len(autocorr) > 20:
                                
                                autocorr = autocorr[:-20]

                                
                                peaks, properties = find_peaks(
                                    autocorr,
                                    height= peak_detect_threshold,  # Ensures height is at least 0
                                    prominence=prominence_threshold,  
                                    distance=min_distance,  
                                    width=2  
                                )
                            
                                peak_buffer_limit = 3
                                
                                # Compute distances between peaks (time period of repetitions)
                                peak_distances = np.diff(peaks) if len(peaks) > 1 else [0]
                                avg_period = np.mean(peak_distances) if len(peak_distances) > 0 else 0
           
                               
                                if motion_amplitude > motion_amplitude_threshold:
                                       
                                    # print("Length of peaks : ", peaks, " prev peak length : ", prev_peak_array_length)
                                    
                                    if len(peaks) > prev_peak_array_length and len(peaks) < peak_buffer_limit:
                                        repetition_count+=1
                                    
                                    
                                    elif len(peaks) >= peak_buffer_limit: # Check if there are 3 peaks. 
                                        wait_idx = peaks[0] # Get the difference of frames between the first and the second peak so that we can skip first peak
                                        
                                        motion_distance = motion_distance[wait_idx:]
                                        repetition_count+=1

         
         
                                    prev_peak_array_length = len(peaks)
                            
                            
      
                        prev_repetition_count = repetition_count
                        
                        
                        mpDrawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mpDrawing.DrawingSpec(color = (0, 0, 255), thickness = 2, circle_radius = 4), mpDrawing.DrawingSpec(color = (0, 255, 0), thickness = 2, circle_radius = 0)) # Render detections
                         
                                                                   
                        # Display motion vector 
                        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                        
                        cv2.arrowedLine(frame, (center_x, center_y),
                                        (int(center_x + norm_dx * motion_amplitude),
                                          int(center_y + norm_dy * motion_amplitude)), (0, 0, 255), 3)
                                            
                        cv2.putText(frame, str(activitiesName[resultIndex]), (50,200), cv2.FONT_HERSHEY_TRIPLEX,1, (0, 255, 0),2, bottomLeftOrigin = False)
                                        
                     
                prev_landmarks = curr_landmarks 
                
                
            frame_count+=1
            
            if wait_idx > 0:
                wait_idx -= 1
            

            
            if cv2.waitKey(10) & 0xFF==ord('q'): # Enter the if statement if the key 'q' is pressed or the screen is closed
                break;
              
            elif cv2.waitKey(10) & 0xFF==ord('Q'):
                break;   
                
                                        
            frame = cv2.resize(frame, (desired_width, desired_height))
        
            # Display repetition count
            cv2.putText(frame, f"Repetitions: {repetition_count}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
            cv2.imshow("Filtered 3D Motion Tracking with Repetition Counting", frame)
        
            if save_video:
                out.write(frame)  # ✅ now frame size matches writer
               
        counting_results.append(repetition_count)
        print("Current exercise : ", activitiesName[resultIndex], " total repetitions counted : ", repetition_count)
            
       
        window_size = 200
        entropy_repeat = compute_entropy(motion_history_full, window_size)
        

        
        # try:
        #     # Plot the motion
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(motion_history_full, marker='.', linestyle='-', color='b', label='Motion')
        
        #     # Labels and title
        #     plt.xlabel('Frame Index')
        #     plt.ylabel('Motion')
        #     plt.title('Motion vs time')
        #     plt.legend()
        #     plt.grid(True)
            

            
        #     plt.show() 
                  
    
            
            
        #     # Plot the autocorrelation for motion with detected peaks
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(autocorr, marker='.', linestyle='-', color='b', label='Autocorrelation for motion')
        
        #     # Labels and title
        #     plt.xlabel('Frame Index')
        #     plt.ylabel('Autocorrelation')
        #     plt.title('Autocorrelation for motion')
        #     plt.legend()
        #     plt.grid(True)
        #     plt.show()
            
        #     plt.plot(autocorr)
        #     plt.plot(peaks, autocorr[peaks], "x")
        #     plt.vlines(x=peaks, ymin=autocorr[peaks] - properties["prominences"],
        #                 ymax = autocorr[peaks], color = "C1")
        #     plt.show()
    
        
    
        #     # 🔹 Plot results
        #     fig, axes = plt.subplots(3, 2, figsize=(12, 12))

        #     # Plot original signals
        #     axes[0, 0].plot(motion_history_full, label="Repeating Signal", color='blue')
        #     axes[0, 0].set_title("🔵 Repeating Signal")

        #     # Plot entropy
        #     axes[1, 0].plot(entropy_repeat, label="Entropy (Repeating)", color='blue')
        #     axes[1, 0].set_title("🔵 Entropy of Repeating Signal")

        #     plt.tight_layout()
        #     plt.show()
                
    
    
        # except Exception as e:
        #     print(e)
        

        cap.release()
        cv2.destroyAllWindows()
        
        print(desired_activity)
        print(counting_results)
        
        if save_video:
            out.release()   # 🔹 Important! Finalize and save the file
            print(f"✅ Saved video at: {output_video_path}")
        




process_video()
    
    
print(desired_activity)
print(counting_results)




    
    
    
    
    
    