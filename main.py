import cv2
import numpy as np
import serial
import time
import argparse

def parseArgs():
    # Define and parse input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--serial_port', help='Name of the Serial-Port', default='ttyAMA0')
    parser.add_argument('--serial_baudrate', help='Name of the Serial-Port', default=9600)

    parser.add_argument('--modeldir', help='Folder the .tflite file is located in', default='model')
    parser.add_argument('--checkpoint', help='Path of the Checkpoint (tflite file)',
                        default='detect.tflite')
    parser.add_argument('--labelmap', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.8)
    parser.add_argument('--resolution',
                        help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

    return parser.parse_args()


def inference(videostream, det, ser):
    while True:
        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (det.height, det.width))
        input_data = np.expand_dims(frame_resized, axis=0)

        det.detect(input_data)
        offset = det.getOffset(det.boxes[0])

        outFormat = det.getRawFormatForArdu(det.boxes[0])
        outStr = f"R{outFormat[0]};{outFormat[1]};{outFormat[2]};{outFormat[3]}"
        ser.write(outStr)
        print(outStr)


if __name__ == '__main__':
    args = parseArgs()
    use_TPU = args.edgetpu

    from detection import Detector
    from video import VideoStream

    SERIAL_PORT = args.serial_port
    SERIAL_BAUDRATE = int(args.serial_baudrate)

    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)

    offsetThreshold = 0.2

    ser = serial.Serial(f'/dev/{SERIAL_PORT}', SERIAL_BAUDRATE)

    PATHS = {
        'CKPT': args.checkpoint,
        'LABELMAP': args.labelmap
    }

    det = Detector(PATHS, imW, imH, useTPU=use_TPU)

    # Initialize video stream
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
    time.sleep(1)

    inference(videostream, det, ser)

