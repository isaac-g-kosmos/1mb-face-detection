from onnxruntime.quantization import quantize_static, QuantType, QuantFormat,quantize_dynamic
from onnxruntime.quantization.calibrate import CalibrationDataReader
from PIL import Image
import onnx
import numpy as np
import cv2
#%%
model=r'Face_detection.onnx'
output_model=r'model.quant.onnx'

class reader(CalibrationDataReader):
    def __init__(self, data_set):
        self.data_set = data_set

    def get_next(self):
        if len(self.data_set) == 0:
            return None
        else:
            path= self.data_set.pop()
            img=Image.open(path)
            img=np.array(img)
            img = img / 255
            img = img.astype(np.float32)
            img=cv2.resize(img,(320,240))
            img = img.reshape(1, 240, 320, 3)
            return {'x': img}
    def get_all(self):
        return self.data_set

#make a list wiht all the  file in C:\Users\isaac\Downloads\wider_images\train ussing os.walk
dataset=[]
import os
for root, dirs, files in os.walk(r"C:\Users\isaac\Downloads\wider_images\train"):
    for file in files:
        if file.endswith(".jpg"):
            dataset.append(os.path.join(root, file))
            # print(os.path.join(root, file))


costum_reader=reader(dataset)

quantize_dynamic(model, output_model, weight_type=QuantType.QUInt8, op_types_to_quantize=['Conv', 'MatMul'])
#%%
    #dinamicly quantize the model
    import onnx
    import onnxruntime as rt
    import numpy as np
    from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat
    model=r'C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\complete_model_files\back_2_neck.onnx'
    output_model=r'models/onnx/back2neck.quant.onnx'
    quantize_dynamic(model, output_model, weight_type=QuantType.QUInt8, op_types_to_quantize=['Conv', 'MatMul'])
    output_model=r'C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\complete_model_files\back2neck.quant.onnx'
    quantize_dynamic(model, output_model, weight_type=QuantType.QUInt8, op_types_to_quantize=['Conv', 'MatMul'])

#%%
import onnxruntime as rt
import numpy as np
sess = rt.InferenceSession(output_model)
# sess = rt.InferenceSession(r"models\onnx\version-RFB-320.onnx")
input_name = sess.get_inputs()[0].name
from PIL import Image
import numpy as np
import cv2
img=Image.open(r"C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\interview_images\0.jpg")
img=Image.open(r"C:\Users\isaac\Downloads\wider_images\test\BENCHAMARK\0_Parade_Parade_0_708.jpg")
img=np.array(img)
#resize  the img to 320x240
img=cv2.resize(img,(256,256))
img=img/255
img=img.astype(np.float32)
img=img.reshape((1,256,256,3))
#%%
output_names = [output.name for output in sess.get_outputs()]

# Prepare input feed
input_feed = {sess.get_inputs()[0].name: img}

pred_onx1 = sess.run(input_feed= {input_name: img},output_names=output_names)
# Run inference
# pred_onx1 = sess.run(input_feed=input_feed, output_names=output_names)

#%%
