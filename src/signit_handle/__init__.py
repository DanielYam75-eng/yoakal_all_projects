import signal
import sys

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!, exiting gracefully...')
    sys.exit(0)

def register_sigint_handler():
    signal.signal(signal.SIGINT, signal_handler)
    print('SIGINT handler registered')
