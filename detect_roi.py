import pyrealsense2 as rs
import numpy as np

# ตั้งค่าการเชื่อมต่อกับกล้อง
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# เริ่มสตรีม
pipeline.start(config)

# ดึงข้อมูล depth frame
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()

# ดึงค่าภายในกล้อง (Intrinsic parameters)
profile = depth_frame.profile
intr = profile.as_video_stream_profile().get_intrinsics()

# จุด X,Y ในภาพ (พิกัดพิกเซล)
x_pixel = 320
y_pixel = 240

# ดึงค่า depth ที่พิกเซล
z_depth = depth_frame.get_distance(x_pixel, y_pixel)

# คำนวณพิกัดโลก (X_world, Y_world, Z_world)
x_world = (x_pixel - intr.ppx) * z_depth / intr.fx
y_world = (y_pixel - intr.ppy) * z_depth / intr.fy
z_world = z_depth

print(f'X: {x_world}, Y: {y_world}, Z: {z_world}')
