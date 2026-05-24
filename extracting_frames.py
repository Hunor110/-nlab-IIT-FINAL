import os
import cv2
import numpy as np
import pyrealsense2 as rs #for extraction from .bag files

# ========= USER SETTINGS =========
#1355  42 kesz
# 417       441  101


bag_file = r"C:\Users\ahuno\Documents\20260305_123406.bag"  # Path to your .bag file, probably should use os...
output_dir_rgb = r"C:\Users\ahuno\Desktop\programming\cv\rgb"        # Where frames will be saved, | -- > here too...
output_dir_depth = r"C:\Users\ahuno\Desktop\programming\cv\depth"
frame_interval = 10
# 15  ~ 2 FPS golden mittelweg => for cv its perfect, for SLAM, we will probably need a more FPS ~5-6FPS(=every 5th frame)
# ==================================

os.makedirs(output_dir_rgb, exist_ok=True)
os.makedirs(output_dir_depth, exist_ok=True)

# Creating pipeline
pipeline = rs.pipeline()
config = rs.config() #enabling the stream

# Enable playback from .bag file, ONLY WORKS WITH PRERECORDED FILES!!!
config.enable_device_from_file(bag_file, repeat_playback=False) 

# Enable streams (must match recording resolution)
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)

profile = pipeline.start(config) #opening the ".bag" file

# Align depth so its matching the rgb-images
align = rs.align(rs.stream.color)

# Disable real-time playback (important for processing speed)
playback = profile.get_device().as_playback()
playback.set_real_time(False)

frame_count = 0
saved_count = 0

try:
    while True:
        try:
            frames = pipeline.wait_for_frames() #read depth-color-timestamps-metadata
        except RuntimeError:
            print("No frames found!") #exit loop!
            break

        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()  #16 - bit depth
        color_frame = aligned_frames.get_color_frame()  #8 - bit BGR image

        if not depth_frame or not color_frame:  #if one of the images are missing => exit loop!
            continue

        if frame_count % frame_interval == 0:   #only for adjustment to save the nth image
            # Convert to numpy, to minimize loss
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())     #(H,W,3channels)

            # Convert RGB → BGR for OpenCV
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # Save RGB
            cv2.imwrite(
                os.path.join(output_dir_rgb, f"color_{saved_count:06d}.jpg"), # not sure about this one, check it before use
                # cuz png is not compressed on the contrary to jpgs... 
                color_image
            )

            # Saving 16-bit depth info, NOTE: this is in milimeters!!!
            cv2.imwrite(
                os.path.join(output_dir_depth, f"depth_{saved_count:06d}.png"),
                depth_image
            )

            saved_count += 1

        frame_count += 1

finally:
    pipeline.stop()

print(f"The process has finished! Saved {saved_count} frame pairs.")
print(f"Folder: {output_dir_rgb} contains the rgb images \n Fodler: {output_dir_depth} contains the depth images!")