import asyncio
from bleak import BleakClient, BleakScanner
import sys

# ชื่อบอร์ดบลูทูธที่ตั้งไว้ใน ESP32-S3
DEVICE_NAME = "Leon_Glove_BLE"
# UUID ของท่อข้อมูล TX (ตรงกับฝั่งบอร์ด)
CHARACTERISTIC_UUID_TX = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

def notification_handler(sender, data):
    """ฟังก์ชันทำงานอัตโนมัติเมื่อมีข้อมูลเดต้ายิงมาจากบลูทูธถุงมือ"""
    try:
        raw_string = data.decode('utf-8')
        data_list = raw_string.split(',')
        
        if len(data_list) == 11:
            # แปลงข้อความเป็นตัวเลข Float ล้วนสำหรับพร้อมเข้าโมเดล
            features = [float(x) for x in data_list]
            
            # พ่นดาต้าตัวเลขล้วนที่เลออนต้องการเอาไปใช้ Train
            sys.stdout.write(f"\rReady for Train/Predict via BLE: {features}")
            sys.stdout.flush()
    except Exception as e:
        pass

async def main():
    print("==================================================")
    print("        BLE BLUETOOTH DATA RECEIVER ACTIVE        ")
    print(f"       Searching for device: '{DEVICE_NAME}'...   ")
    print("==================================================")

    # 1. ค้นหาที่อยู่ (Address) ของถุงมือในอากาศ
    device = await BleakScanner.find_device_by_name(DEVICE_NAME)
    
    if not device:
        print(f"Error: Could not find '{DEVICE_NAME}'. Make sure the glove is powered on.")
        return

    print(f"[FOUND] Found glove at address: {device.address}")
    print("Connecting to glove...")

    # 2. ทำการเชื่อมต่อเข้ากับถุงมือ
    async with BleakClient(device) as client:
        print("[CONNECTED] Secure Bluetooth link established!")
        print("Listening for trainable values... (Press Ctrl+C to stop)")
        
        # 3. เปิดท่อฟังข้อมูลเปิดโหมดดักรับ Notification สตรีมมิ่ง
        await client.start_notify(CHARACTERISTIC_UUID_TX, notification_handler)
        
        # วนลูปรันระบบทิ้งไว้เรื่อยๆ จนกว่าจะกดปิดโปรแกรม
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping BLE Server. Disconnected from glove. Goodbye!")