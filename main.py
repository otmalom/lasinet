import os
import cv2
import imageio
from engine import Engine

model = Engine()
data_dir = 'data/08-25'
spectrum, spec_frame, echograms, echo_frames, echo_gif, plot, metrics = model(data_dir)

# 
os.makedirs('result/spectrum',exist_ok=True)
os.makedirs('result/echogram',exist_ok=True)
os.makedirs('result/gif',exist_ok=True)

#
cv2.imwrite(f'result/spectrum/{spec_frame}',spectrum)
cv2.imwrite(f'result/plot/{spec_frame}',plot)
for image, frame in zip(echograms, echo_frames):
    cv2.imwrite(f'result/echogram/{frame}',image)
imageio.mimsave(f'result/gif/{spec_frame.split(".")[0]}.gif', echo_gif, format='GIF', duration=0.5, loop=0)  # 设置每帧持续时间（以秒为单位）
print(metrics)