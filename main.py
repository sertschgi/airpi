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
                        default='model/detect.tflite')
    parser.add_argument('--labelmap', help='Name of the labelmap file, if different than labelmap.txt',
                        default='model/labelmap.txt')
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

        # if not frame1:
        #     print("Could not read frame. Exiting... Consider checking your camera configuration.")
        #     videostream.stop()
        #     quit(-1)

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (det.height, det.width))
        input_data = np.expand_dims(frame_resized, axis=0)

        cv2.imshow('image', frame_rgb)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

        det.detect(input_data)
        if det.boxes is not None:
            outFormat = det.getRawFormatForArdu(det.boxes[0])
            print(det.boxes)
            outStr = f"R{outFormat[0]};{outFormat[1]};{outFormat[2]};{outFormat[3]}"
            ser.write(outStr)
            print(outStr)


def setup():
    args = parseArgs()

    from config.config import set_USE_TPU
    set_USE_TPU(args.edgetpu)

    from detection import Detector
    from video import VideoStream

    SERIAL_PORT = args.serial_port
    SERIAL_BAUDRATE = int(args.serial_baudrate)

    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)

    ser = serial.Serial(f'/dev/{SERIAL_PORT}', SERIAL_BAUDRATE)

    PATHS = {
        'CKPT': args.checkpoint,
        'LABELMAP': args.labelmap
    }

    # Initialize video stream
    videostream = VideoStream.VideoStream(resolution=(imW, imH), framerate=30).start()
    time.sleep(1)

    det = Detector.Detector(PATHS, imW, imH)

    return videostream, det, ser


def main():
    vid, det, ser = setup()
    inference(vid, det, ser)


if __name__ == '__main__':
    main()
