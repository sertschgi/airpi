import numpy as np
import importlib.util

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate


class Detector:
    def __init__(self, PATHS, IMW, IMH, useTPU=False, MCT=0.5):
        self.scores = []
        self.boxes = []
        self.classes = []

        self.IMW = IMW
        self.IMH = IMH
        self.useTPU = useTPU
        self.MCT = MCT

        self.imgCenter = (IMW / 2, IMH / 2)

        with open(PATHS["LABELMAP"], 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        if use_TPU:
            self.interpreter = Interpreter(
                model_path=PATHS['CKPT'],
                experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
            )
            print(PATHS['CKPT'])
        else:
            self.interpreter = Interpreter(model_path=PATHS['CKPT'])

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

        outname = self.output_details[0]['name']

        if ('StatefulPartitionedCall' in outname):
            self.boxes_idx, self.classes_idx, self.scores_idx = 1, 3, 0
        else:
            self.boxes_idx, self.classes_idx, self.scores_idx = 0, 1, 2

    def detect(self, input_data):
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        self.boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])[
            0].sort()  # Bounding box coordinates of detected objects
        self.classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])[
            0]  # Class index of detected objects
        self.scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])[
            0]  # Confidence of detected objects

    def getOffset(self, box):
        boxCenter = ((box[1] + box[3]) / 2, (box[0] + box[2]) / 2)

        return (boxCenter[0] - self.imgCenter[0], boxCenter[1] - self.imgCenter[1])

    def getRawFormatForArdu(self, box):
        xMin = box[1]
        yMin = box[0]
        xMax = box[3]
        yMax = box[2]
        boxHeight = xMax - xMin
        boxWidth = yMax - yMin
        return xMin, yMin, boxWidth, boxHeight