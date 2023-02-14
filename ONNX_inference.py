# import onnx
# onnx_model = onnx.load(r"C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\models\onnx\version-slim-320.onnx")
# onnx.checker.check_model(onnx_model)
#%%
from PIL import Image
import numpy as np
import cv2
img=Image.open(r"C:\Users\isaac\PycharmProjects\face_exctraction\Ultra-Light-Fast-Generic-Face-Detector-1MB\interview_images\0.jpg")
img=Image.open(r"C:\Users\isaac\Downloads\wider_images\test\BENCHAMARK\0_Parade_Parade_0_708.jpg")
img=np.array(img)
#resize  the img to 320x240
img=cv2.resize(img,(320,240))
img=img/255
img=img.astype(np.float32)


#make an inference with the onnx_model
import onnxruntime as rt
import numpy as np
# sess = rt.InferenceSession(r"models\onnx\version-slim-320_simplified.onnx")
img=img.reshape(1,240,320,3)
# img=img/255
# img=img.astype(np.float32)
sess = rt.InferenceSession(r"models\onnx\model (1).onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
# boxes_name= sess.get_outputs()[1].name
pred_onx = sess.run([label_name], {input_name: img})
#%%
import onnxruntime as rt
import numpy as np
sess = rt.InferenceSession(r"models\onnx\version-slim-320_simplified.onnx")
# sess = rt.InferenceSession(r"models\onnx\version-RFB-320.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
boxes_name= sess.get_outputs()[1].name
pred_onx1 = sess.run([label_name,boxes_name], {input_name: img})
