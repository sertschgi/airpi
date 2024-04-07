import cv2
from picamera2 import Picamera2
from threading import Thread
import time

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(650, 420), framerate=30):
        self.picam2 = Picamera2()

        config = self.picam2.create_preview_configuration(main={'format': 'RGB888', 'size': resolution})
        self.picam2.configure(config)

        self.picam2.start()
        time.sleep(1)

        self.frame = self.picam2.capture_array("main")

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            self.frame = self.picam2.capture_array("main")

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True