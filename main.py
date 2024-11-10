import cv2
import numpy as np
import supervision as sv
import os
import torch
from camera import RealSenseCamera
from time import sleep
from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
from segment_anything import sam_model_registry, SamPredictor
from dobotapi import ArmControl

# !!!--------------------------------------------ฟังก์ชันแปลงพิกัดจากกล้องไปยังหุ่นยนต์---------------------------------!!!
def camera_to_robot(camera_coord):
    A = np.array([[0.94290462, -0.11734718],
                  [0.01075611, -0.8782723]])
    t = np.array([53.0280588, -580.70712467])
    robot_coord = np.dot(A, camera_coord) + t
    robot_coord[0] += -30  # บวกค่า X ด้วย -30
    robot_coord[1] += -25  # บวกค่า Y ด้วย -60
    return robot_coord
# !!!--------------------------------------------จบฟังก์ชันแปลงพิกัดจากกล้องไปยังหุ่นยนต์---------------------------------!!!

# GroundingDINO setup
GROUNDING_DINO_CONFIG_PATH = 'groundingdino/config/GroundingDINO_SwinT_OGC.py'
GROUNDING_DINO_CHECKPOINT_PATH = 'weights/groundingdino_swint_ogc.pth'
GD_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
# SAM setup
MODEL_TYPE = "vit_b"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'weights/sam_vit_b_01ec64.pth'
# ตรวจสอบเส้นทางของไฟล์ 
if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint file not found at: {CHECKPOINT_PATH}")
# โหลดโมเดล
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
# สร้าง SamPredictor
mask_predictor = SamPredictor(sam)

#!!------------------------------------ ส่วนของกล้อง -----------------------------------!!#
# camera = RealSenseCamera(width=640, height=480, fps=15)
roi_x, roi_y, roi_w, roi_h = 130, 30, 360, 320
# กำหนด Serial Number ของกล้องตัวที่ 1
serial_number_1 = '215122252229'  # แทนที่ด้วย Serial Number ของกล้องตัวที่ 1
# สร้างออบเจกต์กล้องพร้อม Serial Number
camera = RealSenseCamera(width=640, height=480, fps=15, serial_number=serial_number_1)

def camera_thread():
    try:
        camera.start() 
        while True:
            depth_image, frame =  camera.get_frames()   
            # แปลงเฟรมเป็น RGB โดยตรงจาก frame ที่จับได้
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_original = frame  # เก็บเฟรมต้นฉบับไว้สำหรับการแสดงผล   
            # # วาดกรอบ ROI
            cv2.rectangle(image_original, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
            # detect objects
            CLASSES = ['product','bottle']
            detections = GD_model.predict_with_classes(
                image=image_rgb,
                classes=CLASSES,
                box_threshold=0.30, #37 ดีสุด
                text_threshold=0.10
            )
            print(detections)
            print(detections.xyxy, type(detections.xyxy))
            detected_boxes = detections.xyxy
            class_id = detections.class_id
            print(class_id)
            mask_annotator = sv.MaskAnnotator(color=sv.Color.BLUE, color_lookup=sv.ColorLookup.INDEX)
            # ตรวจสอบว่ามีการตรวจจับหรือไม่
            if len(detections.xyxy) == 0:
                print("ไม่พบproduct")
                continue  # ข้ามการประมวลผลถัดไปหากไม่มีการตรวจจับ
            segmented_mask = []
            counter = 0
            for mybox in detected_boxes:
                mybox = np.array(mybox)
                # ตรวจสอบว่ากล่องอยู่ภายใน ROI หรือไม่
                if (mybox[0] < roi_x or mybox[1] < roi_y or mybox[2] > roi_x + roi_w or mybox[3] > roi_y + roi_h):
                    continue  # ข้ามการตรวจจับถ้าอยู่นอก ROI             
                object_detected_in_roi = True  # พบวัตถุใน ROI
                mask_predictor.set_image(image_rgb)
                masks, scores, logits = mask_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=mybox,
                    multimask_output=False
                )
                segmented_mask.append(masks)
                binary_mask = (masks[0].astype(np.uint8)) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)                
                    if w > 10 and h > 10:
                        # ใช้ minAreaRect เพื่อสร้าง rotated bounding box
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)  # แปลงให้เป็น int สำหรับการวาดกรอบ
                        # วาด rotated bounding box
                        cv2.drawContours(image_original, [box], 0, (0, 255, 0), 2)
                        
    #!!------------------------------------# คำนวณความกว้าง (Width) และความสูง (Height) -----------------------------------!!#
                        # ดึงพิกัดมุม
                        p1, p2, p3, p4 = box
                        width_x = np.linalg.norm(p1 - p2)  
                        height_y = np.linalg.norm(p1 - p4)  

                        # ตรวจสอบว่า width_x และ height_y เป็นบวก
                        if width_x > 0 and height_y > 0:
                            # เลือกความกว้างหรือความสูงที่สั้นที่สุด
                            shortest_side = min(width_x, height_y)
                            if shortest_side == width_x:
                                cv2.line(image_original, tuple(p1), tuple(p2), (0, 0, 255), 6)  
                            else:
                                cv2.line(image_original, tuple(p1), tuple(p4), (0, 0, 255), 6) 

                            side_label = 'Width' if shortest_side == width_x else 'Height'
                            print(f"Shortest Side: {shortest_side:.2f} (Chosen: {side_label})")
                        cv2.drawContours(image_original, [box], 0, (0, 255, 0), 2)
                        # วาดจุดที่มุม
                        for point in box:
                            cv2.circle(image_original, tuple(point), 2, (0, 0, 255), -1)  
                        cv2.putText(image_original, f'Shortest: {shortest_side:.2f} ({side_label})', 
                                    (int(p1[0]), int(p1[1] - 75)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    #!!------------------------------------จบ คำนวณความกว้าง (Width) และความสูง (Height) -----------------------------------!!
    #!!!--------------------------------------------  หาพิกัดกล้องและROBOT  ------------------------------------------------------!!!
                        M = cv2.moments(contour)                   
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])                        
                            print("cx",cx)
                            print("cy",cy)
                            cv2.circle(image_original, (cx, cy), 2, (0, 255, 0), -1)
                        # ตรวจสอบค่าความลึกก่อนที่จะคำนวณ
                            depth_value = depth_image[cy, cx]
                            print(f'Depth Value: {depth_value}')
                            # คำนวณพิกัด (X, Y, Z) หากค่า depth ไม่เป็นศูนย์
                            if depth_value > 0:  
                                x_world, y_world, z_world = camera.get_xyz(depth_image, cx, cy)
                                print(f'X: {x_world:.2f} mm, Y: {y_world:.2f} mm, Z: {z_world:.2f} mm')
                                # เรียกใช้ฟังก์ชัน camera_to_robot
                                # แปลงพิกัดจากกล้องเป็นพิกัดหุ่นยนต์
                                camera_coord = np.array([x_world, y_world])
                                robot_coord = camera_to_robot(camera_coord)
                                z_robot = (z_world * (-1.6)) + 1700
                                print(f'Robot coordinate: X: {robot_coord[0]:.2f}, Y: {robot_coord[1]:.2f},Z:{z_robot:.2f} ')
                                arm_control.set_target_position(robot_coord[0], robot_coord[1], z_robot)
                                sleep(5)
                                # แสดงพิกัดหุ่นยนต์บนภาพ
                                robot_text = f'Robot X: {robot_coord[0]:.2f}m, Y: {robot_coord[1]:.2f}m, Z:{z_robot:.2f}m '
                                cv2.putText(image_original, robot_text, (x - 10, y - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0, 255), 1)
                                # แสดงค่า X, Y, Z บนภาพ
                                text_x = f'X:{x_world:.2f}mm'
                                text_y = f'Y:{y_world:.2f}mm'
                                text_z = f'Z:{z_world:.2f}mm'
                                cv2.putText(image_original, text_x, (x - 10, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                cv2.putText(image_original, text_y, (x - 10, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                cv2.putText(image_original, text_z, (x - 10, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            else:
                                print("Depth value is zero, cannot calculate world coordinates.")
                            # arm_control.set_target_position(robot_coord[0], robot_coord[1], z_robot)
                           
    #!!!--------------------------------------------   จบ  หาพิกัดกล้องและROBOT  ------------------------------------------------------!!!
                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks=masks),
                    mask=masks
                )
                annotated_image = mask_annotator.annotate(scene=image_original.copy(), detections=detections)
                image_original = annotated_image
                counter += 1
            # # Display the image
            cv2.imshow('Live Camera Detection', image_original)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("ปิดการแสดงผลด้วยคีย์บอร์ด")
    finally:
        camera.stop() 
        cv2.destroyAllWindows()


import cv2
from inference_sdk import InferenceHTTPClient
from camera import RealSenseCamera

# Global variables for x and y coordinates
barcode_x, barcode_y = None, None

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="rjLBeYIactn64VMw8BPH"  # Replace with your actual API key
)

model_id = "barcodes-zmxjq/4"  # Make sure to use the correct model version
serial_number_2 = '133522250552'  # Replace with Serial Number of camera 2

def run_barcode_detection():
    global barcode_x, barcode_y  # Make x and y global
    camera = RealSenseCamera(width=640, height=480, fps=15, serial_number=serial_number_2)
    camera.start()

    try:
        while True:
            depth_image, frame = camera.get_frames() 
            img_file = "frame.jpg"
            cv2.imwrite(img_file, frame)

            result = CLIENT.infer(img_file, model_id=model_id)
            print(result)

            for prediction in result['predictions']:
                barcode_x = int(prediction['x'])
                barcode_y = int(prediction['y'])
                width = int(prediction['width'])
                height = int(prediction['height'])

                center_x, center_y = barcode_x, barcode_y
                cv2.rectangle(frame, (barcode_x - width // 2, barcode_y - height // 2), 
                              (barcode_x + width // 2, barcode_y + height // 2), (0, 255, 0), 2)
                print(f"Barcode Center: X={center_x}, Y={center_y}")

                text = f"X: {center_x}, Y: {center_y}"
                cv2.putText(frame, text, (barcode_x - width // 2, barcode_y - height // 2 - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Live Camera Inference', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Terminated with Keyboard")
    finally:
        camera.stop()
        cv2.destroyAllWindows()

def get_barcode_position():
    return barcode_x, barcode_y
    #!!!--------------------------------------------   จบการทำงานของ camera ------------------------------------------------------!!!

import threading
import numpy as np
from time import sleep
from dobotapi import ArmControl

if __name__ == '__main__':
    # # สร้างออบเจ็กต์ควบคุมแขนกล
    arm_control = ArmControl()
    arm_control.start()  # เริ่มการทำงานของ thread แขนกล

    # สร้าง thread สำหรับกล้อง
    camera_thread_instance = threading.Thread(target=camera_thread)
    camera_thread_instance.setDaemon(True)
    camera_thread_instance.start()

    # thread2 = threading.Thread(target=run_barcode_detection)
    # thread2.start()

    try:
        while True:
            # เช็คพิกัดและทำงานร่วมกันระหว่างกล้องและแขนกล
            sleep(5)

    except KeyboardInterrupt:
        print("Program interrupted. Shutting down...")
    
    finally:
        # ปิดการทำงานของหุ่นยนต์และเคลียร์ข้อผิดพลาด
        arm_control.dashboard.DisableRobot()
        arm_control.dashboard.ClearError()
        
        # หยุดการทำงานของแขนกลและกล้อง
        arm_control.stop()  # หยุดการทำงานของแขนกล
        arm_control.join()  # รอให้ thread หยุดทำงาน
        camera_thread_instance.join()  # รอให้กล้องปิดตัวลงก่อน
        cv2.destroyAllWindows()