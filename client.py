import cv2
import socket
import struct
import pickle
import time

cap = cv2.VideoCapture(0)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 6942))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        _, buffer = cv2.imencode('.jpg', frame)
        data = pickle.dumps(buffer)
        size = struct.pack("!L", len(data))
        
        client_socket.sendall(size + data)
        time.sleep(0.5)  # Send every 0.5 seconds
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    client_socket.close()
