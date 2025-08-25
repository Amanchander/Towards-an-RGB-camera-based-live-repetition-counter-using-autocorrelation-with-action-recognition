import cv2
import numpy as np
from collections import deque
import glob
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import statistics
         
from scipy.stats import differential_entropy
from tensorflow.keras.models import load_model
import tensorflow as tf

  

# pip install scikit-learn

print(tf.__version__)

from tensorflow.keras import Sequential
import tensorflow as tf


counting_results = []  

desired_activity = 'hurdle_step'

# Set the activity you want to read (e.g., 'm01', 'm02', etc.)
desired_activity_code = desired_activity


activitiesName = ['m01', 'm02', 'm03', 'm04', 'm05', 'm06', 'm07', 'm08', 'm09', 'm10']


input_data_path = "_path_to_joint_coordinates_data_" # For UI-PRMD Vicon data


# Get all CSV files in all subdirectories
all_files = glob.glob(os.path.join(input_data_path, "*.csv"), recursive=True)


# Filter files that start with the desired activity
filtered_files = [f for f in all_files if os.path.basename(f).startswith(desired_activity_code)]

# Print the filtered list
print(f"Files for activity {desired_activity_code}:")

for file in filtered_files:
    print(file)


def read_file(in_file):
    return pd.read_csv(in_file)


exercise_data = [] 


for file in filtered_files:
    exercise_data.append(read_file(file))
    print(read_file(file))


print("Length of the data is : ", len(exercise_data))


print(np.array([exercise_data[0].iloc[0,:]]).shape)

##################################################################################################################################################################################
##################################################################################################################################################################################



save_model_path = "_path_to_your_saved_model_"

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


try:
    
    myModel = load_model(save_model_path + '/transformer_model.h5',
                       custom_objects={'TransformerBlock': TransformerBlock})

except:
    pass

save_img_path  = "_path_to_save_output_graphs_"


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



    
def process_video(input_data, sub_no):
    
    print("Current activity is  ", desired_activity_code)

    print("The length of the cap is ", len(input_data))
    
    repetition_count = 0

    
    peaks, properties = [], []
    
    wait_idx = 0
    
    dynamic_prominence_ratio = 0.4
    
    min_distance = 10



    alpha = 0.08
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
    
    motion_amplitude_threshold = 2

    motion_distance_arr_limit = 800
    

    
    variance_array = []
    resultIndex = 0

    overall_direction = +1
    
    
    desiredFrameWidth = 550
    
    prev_repetition_count = 0

    
    for i in range(0, 117):
        columnArray.append(i)
     
    landMarksDf = pd.DataFrame(columns = columnArray)
    
    first_actvity_detected = False
    
    
    if len(input_data) > 1:
            
        for idx in range(0, len(input_data)): 

            frame = np.array(input_data.iloc[idx,:])
            curr_landmarks = frame
            
            
            if len(frame)>1:
                
                addDf = pd.DataFrame(frame)
                           
                landmarksArray = [frame]
                addDf = pd.DataFrame(landmarksArray)
                
                
                
                if len(landMarksDf) < noOfFrameSize:  
                    landMarksDf = pd.concat([landMarksDf, addDf], ignore_index=True, axis = 0)

                
                elif len(landMarksDf) >= noOfFrameSize:    
                    landMarksDf = pd.concat([landMarksDf, addDf], ignore_index=True, axis = 0)
                    landMarksDf.drop(axis = 0, index = 0, inplace = True)
                    landMarksDf = landMarksDf.reset_index(drop =True)
                    inputArray = np.array(landMarksDf) 
                    
                    try:
                        result = myModel(inputArray.reshape(1,noOfFrameSize, noOfFeatures,1))               
                        resultIndex = np.argmax(result)
                        
                    except:
                        pass
                
                if frame_count > 2: 
                    
                    total_dx, total_dy = 0, 0
                    count = 0
        
        
                    n_landmarks = len(curr_landmarks) // 4 # should be 33 for MediaPipe
                    frame_width = frame.shape[0]
                    
                    
                    for i in range(n_landmarks):
                        x = (curr_landmarks[i * 3 + 0])
                        y = (curr_landmarks[i * 3 + 1])
                    
                        prev_x = (prev_landmarks[i * 3 + 0] )
                        prev_y = (prev_landmarks[i * 3 + 1] )
                    
                        dx = x - prev_x
                        dy = y - prev_y
                    
                        delta_limit = 8
                        
                        # Clamp dx, dy
                        dx = max(min(dx, delta_limit), -delta_limit)
                        dy = max(min(dy, delta_limit), -delta_limit)
                    
                        total_dx += dx
                        total_dy += dy
                        count += 1
                    


                    if count > 0:
                        # Compute resultant motion
                        resultant_dx, resultant_dy = total_dx, total_dy
                    
                        # Apply Exponential Moving Average
                        smoothed_dx = alpha * resultant_dx + (1 - alpha) * smoothed_dx
                        smoothed_dy = alpha * resultant_dy + (1 - alpha) * smoothed_dy

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
                        
                
                        
                        if len(motion_history_x) > 3:
                            # Compute variances
                            variance_x = statistics.variance(motion_history_x)  # X-axis
                            variance_y = statistics.variance(motion_history_y)  # Y-axis
                            variance_45 = statistics.variance(motion_history_45)  # 45° axis
                            variance_neg45 = statistics.variance(motion_history_neg45)  # -45° axis
                        
                            
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
    
                            # Set prominence dynamically as 20% of the highest autocorrelation value
                            prominence_threshold = np.max(autocorr) * dynamic_prominence_ratio  
                                                   
                            
                            if len(autocorr) > 20:
                                
                                autocorr = autocorr[:-20]
                                
                                
                                peaks, properties = find_peaks(
                                    autocorr,
                                    height= 0.4,  # Ensures height is at least 0
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
                        
                     
            prev_landmarks = curr_landmarks 
                
                
            frame_count+=1
            
            if wait_idx > 0:
                wait_idx -= 1
            
                
        counting_results.append(repetition_count)

        window_size = 200
        entropy_repeat = compute_entropy(motion_history_full, window_size)
              
        
        try:
            # Plot the motion
            plt.figure(figsize=(10, 5))
            plt.plot(motion_history_full, marker='.', linestyle='-', color='b', label='Motion')
        
            # Labels and title
            plt.xlabel('Frame Index')
            plt.ylabel('Motion')
            plt.title('Motion vs time')
            plt.legend()
            plt.grid(True)
            plt.show() 
         
            # Plot the autocorrelation for motion with detected peaks
            plt.figure(figsize=(10, 5))
            plt.plot(autocorr, marker='.', linestyle='-', color='b', label='Autocorrelation for motion')
        
            # Labels and title
            plt.xlabel('Frame Index')
            plt.ylabel('Autocorrelation')
            plt.title('Autocorrelation for motion')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            plt.plot(autocorr)
            plt.plot(peaks, autocorr[peaks], "x")
            plt.vlines(x=peaks, ymin=autocorr[peaks] - properties["prominences"],
                        ymax = autocorr[peaks], color = "C1")
            plt.show()
    
        
    
            # 🔹 Plot results
            fig, axes = plt.subplots(3, 2, figsize=(12, 12))

            # Plot original signals
            axes[0, 0].plot(motion_history_full, label="Repeating Signal", color='blue')
            axes[0, 0].set_title("🔵 Repeating Signal")


            plt.tight_layout()
            plt.show()
            # Plot the motion
            plt.figure(figsize=(10, 5))
            plt.plot(motion_history_full, marker='.', linestyle='-', color='b', label='Motion')
        
            # Labels and title
            plt.xlabel('Frame Index')
            plt.ylabel('Motion')
            plt.title('Motion vs time')
            plt.legend()
            plt.grid(True)
            plt.show() 
                  
            
            # Plot the autocorrelation for motion with detected peaks
            plt.figure(figsize=(10, 5))
            plt.plot(autocorr, marker='.', linestyle='-', color='b', label='Autocorrelation for motion')
        
            # Labels and title
            plt.xlabel('Frame Index')
            plt.ylabel('Autocorrelation')
            plt.title('Autocorrelation for motion') 
            plt.legend()
            plt.grid(True)
            plt.show()
            
            plt.plot(autocorr)
            plt.plot(peaks, autocorr[peaks], "x")
            plt.vlines(x=peaks, ymin=autocorr[peaks] - properties["prominences"],
                        ymax = autocorr[peaks], color = "C1")
            plt.show()

    
    
        except Exception as e:
            print(e)
        

        print(desired_activity)
        print(counting_results)
        

subject_number = 0

# for i in range(0,1):
for data in exercise_data:
    subject_number+=1
    process_video(data, subject_number)
    
    
print(desired_activity)
print(counting_results)

    
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error



ground_truth_value = 10 # Change the ground truth value as required. 

ground_truth = [ground_truth_value] * len(counting_results) 

 
# Calculate MAE
mae = mean_absolute_error(ground_truth, counting_results)

# Calculate RMSE
rmse = mean_squared_error(ground_truth, counting_results, squared=False)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    
    