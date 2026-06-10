"""
leader_pub.py — runs on Mac mini
Reads SO101 leader arm, publishes joint positions over ZMQ PUB socket.

Usage:
    conda activate lerobot
    python leader_pub.py
"""

import json
import time
import zmq

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

# ── Config ────────────────────────────────────────────────────────────────────

LEADER_PORT   = "/dev/tty.usbmodem5B415334921"
LEADER_ID     = "my_awesome_leader_arm"
ZMQ_PORT      = 5555
HZ            = 50

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ZMQ
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"ZMQ PUB bound on tcp://*:{ZMQ_PORT}")

    # Leader arm
    cfg = SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID)
    leader = SO101Leader(cfg)
    leader.connect()
    print(f"Leader arm connected: {LEADER_PORT}")
    print(f"Publishing at {HZ} Hz. Ctrl-C to stop.\n")

    period = 1.0 / HZ
    try:
        while True:
            t0 = time.perf_counter()

            action = leader.get_action()          # dict[str, float]
            sock.send_string(json.dumps(action))

            elapsed = time.perf_counter() - t0
            sleep = period - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        leader.disconnect()
        sock.close()
        ctx.term()
        print("Done.")

if __name__ == "__main__":
    main()
