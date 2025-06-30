import signal
import sys

def signal_handler(sig, frame):
    sys.exit(0)

def register_sigint_handler():
    signal.signal(signal.SIGINT, signal_handler)

