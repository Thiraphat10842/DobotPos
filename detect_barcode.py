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
serial_number_2 = '215122252229'  # Replace with Serial Number of camera 2

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
