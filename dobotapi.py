from threading import Thread
from dobot_api import DobotApiDashboard, DobotApiMove, DobotApi
from time import sleep
import numpy as np
from detect_barcode import run_barcode_detection, get_barcode_position

# โค้ดส่วนของแขนกล
class ArmControl(Thread):
    def __init__(self, ip: str="192.168.5.1") -> None:
        # เรียก super() เพื่อให้แน่ใจว่า Thread ถูกตั้งค่าอย่างถูกต้อง
        super().__init__()

        self.ip: str = ip
        self.dashboardPort: int = 29999
        self.feedPort: int = 30004
        self.movePort: int = 30005
        self.current_position = None
        self.running = True  # เพิ่มตัวแปร running ให้เป็น True

        # เชื่อมต่อ Robot
        dashboard, move, feed = self.ConnectRobot()       
        self.dashboard: DobotApiDashboard = dashboard
        self.move: DobotApiMove = move
        self.feed: DobotApi = feed
        dashboard.SpeedL(30)

        print("Start power up.")
        self.dashboard.PowerOn()
        print("Please wait patiently, Robots are working hard to start.")
        count = 3
        while count > 0:
            print(count)
            count = count - 1
            sleep(1)

        dashboard.SpeedL(10)
        print("Clear error.")
        self.dashboard.ClearError()

        print("Start enable.")
        self.dashboard.EnableRobot()
        print("Complete enable.")

        self.current_target = None

    def ConnectRobot(self):
        try:
            ip = self.ip
            dashboardPort = 29999
            movePort = 30003
            feedPort = 30004
            print("Connecting...")
            dashboard = DobotApiDashboard(ip, dashboardPort)
            move = DobotApiMove(ip, movePort)
            feed = DobotApi(ip, feedPort)
            print(">.<Connecting successed>!<")
            return dashboard, move, feed
        except Exception as e:
            print(":(Connecting fail:(")
            raise e

    def grip_close(self):
        # self.dashboard.ToolDO(1, 1)
        self.dashboard.ToolDO(2, 1)
        # self.dashboard.ToolDO(2, close_percentage)
        sleep(1)

    def grip_open(self):
        self.dashboard.ToolDO(1, 0)
        self.dashboard.ToolDO(2, 0)
        # self.dashboard.ToolDO(2, open_percentage)
        sleep(1)


    def set_target_position(self, x, y, z):
        self.current_target = (x, y, z)
        print(f"Target set to: x={x}, y={y}, z={z}")
        self.move.MovL(x, y, z, -177, 0, 178)


    # def set_target_barcode(self, x, y ):
    #       # Access x, y coordinates
    #     x, y = get_barcode_position()
    #     self.current_barcode = (x, y)
    #     print(f"Retrieved Barcode Position: X={x}, Y={y}")
    

    def home(self):
        print("Returning to home position...")
        self.move.MovL(-170, -290, 540, -177, 0, 178)
        print("Reached home position")
        sleep(2)
    
    def move_to_barcode(self):
        #ไปหาบาโค้ด
        self.move.MovL(154.49, -296.67, 442, -177, 0, 178)

    def move_to_place(self):
        #ไปวางสินค้า
        self.move.MovL(-309.617, -449.855, 464.553, -177, 0, 178) 

    def move_to_scan(self):
        #ไปวางสินค้า
        self.move.MovL(167.73, -585.80, 442,-177, 0, 178) 

    def move_to_target(self):
        # if self.current_target is not None:
            x, y, z = self.current_target
            print(f"Moving to target: x={x}, y={y}, z={z}")
            self.move.MovL(x, y, z, -177, 0, 178)  # เคลื่อนไปยังตำแหน่งเป้าหมาย
            print("Reached target position")
            sleep(1)
        # else:
        #     print("No target set in move_to_target")
    def down(self):
        x, y, z = self.current_target
        print(f"Current target: {self.current_target}")
        # ลดแกน Z เล็กน้อยเพื่อนำกริปเปอร์ลงใกล้วัตถุ
        z_lowered = z -65# ปรับตามที่เหมาะสม
        self.move.MovL(x, y, z_lowered, -177, 0, 178)
        print("Lowered to pick position")
        sleep(1)
    
    def joint6(self):
        """
        หมุนข้อต่อที่ 6 ไปยังมุม 90 องศา
        """
        angle = 90 # กำหนดมุมที่ต้องการให้ข้อต่อที่ 6 หมุน
        print(f"Rotating joint 6 to {angle} degrees")
        # เรียกใช้ RelMovJ เพื่อควบคุมเฉพาะข้อต่อที่ 6
        self.move.RelMovJ(0, 0, 0, 0, 0, angle)
        sleep(3)  # หน่วงเวลาให้การหมุนเสร็จสิ้น


    def run(self):
        # ตัวแปร state สำหรับติดตามขั้นตอนปัจจุบัน
        state = "move_to_target"  # เริ่มที่ `move_to_target`
        while self.running:
            if self.current_target is not None:
                if state == "move_to_target":
                    # ไปที่เป้าหมายและจับวัตถุ
                    self.move_to_target()
                    self.grip_open()
                    sleep(2)
                    self.down()
                    self.grip_close()
                    sleep(2)
                    self.home()
                    state = "move_to_barcode"  # ตั้งสถานะไปที่ `move_to_barcode`

                elif state == "move_to_barcode":
                    # ไปที่จุดสแกนบาร์โค้ด
                    print("Moving to barcode scan position...")
                    self.move_to_barcode()  # ไปยังตำแหน่งสแกนบาร์โค้ด
                    print("Reached barcode scan position")  
                    
                    # รอให้แขนหยุดที่จุดสแกนบาร์โค้ดก่อนที่จะทำการหมุน
                    sleep(10)

                    # หมุนข้อต่อที่ 6 สำหรับการสแกน
                    print("Rotating joint 6 for barcode scan...")
                    self.joint6()
                    sleep(5)  # รอให้การหมุนเสร็จสิ้น

                    state = "move_to_scan"  # ตั้งสถานะไปยัง `move_to_scan`

                elif state == "move_to_scan":
                    # ไปที่จุดสแกนเพิ่มเติม
                    print("Moving to additional scan position...")
                    self.move_to_scan()  # ไปยังตำแหน่ง `move_to_scan`
                    print("Reached additional scan position")                      
                    # หมุนข้อต่อที่ 6 เป็น 90 องศา
                    print("Rotating joint 6 to 90 degrees...")
                    self.joint6()
                    sleep(5)

                    self.home()
                    state = "move_to_place"  # ตั้งสถานะไปที่ `move_to_place`

                elif state == "move_to_place":
                    # ไปที่จุดวางสินค้า
                    self.move_to_place()
                    sleep(5)
                    self.grip_open()
                    sleep(3)
                    self.home()
                    state = "move_to_target"  # ตั้งสถานะกลับไปเริ่มที่ `move_to_target` อีกครั้ง
            else:
                print("No current target available")
            
            # เพิ่มการหน่วงเวลาสั้น ๆ ก่อนการทำงานถัดไป
            sleep(5)

    def stop(self):
        self.running = False  # หยุดลูป