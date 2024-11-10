import pyrealsense2 as rs
import numpy as np
import cv2
class RealSenseCamera:

    def __init__(self, width=640, height=480, fps=15,serial_number=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
              
        if serial_number:
            self.config.enable_device(serial_number)
            
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    def start(self):
        self.pipeline.start(self.config)

    def stop(self):
        self.pipeline.stop()
    def get_frames(self):
        # Wait for a new set of frames
        frames = self.pipeline.wait_for_frames()

        # Get the depth frame
        depth_frame = frames.get_depth_frame() 
        color_frame = frames.get_color_frame()

        # Check if both frames are valid
        if not depth_frame or not color_frame:
            return None, None

        # Convert depth and color frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image  # <-- คืนค่า depth_image และ color_image

    def get_intrinsics(self):
        # ดึงข้อมูลอินทรินซิกส์ของกล้อง
        profile = self.pipeline.get_active_profile()
        intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        return intrinsics

    def get_xyz(self, depth_image, cx, cy):
        # Get the intrinsic parameters
        intrinsics = self.get_intrinsics()
        fx = intrinsics.fx
        fy = intrinsics.fy
        ppx = intrinsics.ppx
        ppy = intrinsics.ppy
        # Get the depth value at (cx, cy)
        z = depth_image[cy, cx] * self.get_depth_scale()  # Depth in meters
        # Calculate the real-world coordinates
        x_world = (cx - ppx) * z / fx  *1000
        y_world = (cy - ppy) * z / fy  *1000
        z_world = z *1000  # Z value is the depth
        return x_world, y_world, z_world
    
    def get_depth_scale(self):
        # Get the depth sensor from the pipeline's active profile
        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        # Get the depth scale (conversion from depth units to meters)
        depth_scale = depth_sensor.get_depth_scale()
        return depth_scale
    
    