import json
import requests
import time
import base64

SERVER_URL = "https://88ee4ecb-f81d-4701-af21-6f56f6525bb0.mock.pstmn.io"

POST_API = "/meter_data"
GET_API = "/test"

DEVICE_ID = "0001"
SW_VER = "0.1.0"
HW_VER = "1.0.0"

def send_data_to_server(gauge_readings, image, run_time):
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    # encode image int base64
    im_base64 = base64.b64encode(image).decode("utf8")

    payload = json.dumps({
                            "device_id": DEVICE_ID,
                            "sw_ver": SW_VER,
                            "hw_ver": HW_VER,
                            "ts": time.time(),
                            "run_time": run_time,
                            "meter_reading": gauge_readings
                            # "image": im_base64
    })
    print(f"Payload: {payload}")
    response = requests.post(SERVER_URL + POST_API, data=payload, headers=headers)

    try:
        data = response.json()
        print(data)
        return True
    except requests.exceptions.RequestException:
        print(response.text)
        return False

def get_data_from_server():
    # check if server is running
    response = requests.get(SERVER_URL + GET_API)
    print(response.status_code)

    if response.status_code == 200:
        return True
    else:
        return False