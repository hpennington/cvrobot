import cv2
import zmq
import numpy as np
import queue
import threading

JETSON_IP = "10.0.0.103"  # set this

ctx = zmq.Context()
queues = {"Webcam": queue.Queue(maxsize=1), "RealSense": queue.Queue(maxsize=1)}

def make_sub(ip, port):
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(f"tcp://{ip}:{port}")
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    return sock

def recv_loop(name, sock):
    while True:
        buf = sock.recv()
        frame = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        try:
            queues[name].put_nowait(frame)
        except queue.Full:
            pass  # drop stale frame

webcam_sock = make_sub(JETSON_IP, 5556)
realsense_sock = make_sub(JETSON_IP, 5557)

threading.Thread(target=recv_loop, args=("Webcam", webcam_sock), daemon=True).start()
threading.Thread(target=recv_loop, args=("RealSense", realsense_sock), daemon=True).start()

print("Press Q to quit")
while True:
    for name, q in queues.items():
        try:
            frame = q.get_nowait()
            cv2.imshow(name, frame)
        except queue.Empty:
            pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
