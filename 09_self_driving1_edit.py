from buildhat import *
import picamera
from picamera.array import PiRGBArray
import cv2
import numpy as np
import time
import argparse
from PIL import Image
import tflite_runtime.interpreter as tflite

l_motor = Motor('B')
r_motor = Motor('A')
g = Motor('D')


def motor_tank(a,b):
    a = a/100
    b = b/100
    l_motor.pwm(-a)
    r_motor.pwm(b)

def motor_stop():
    l_motor.stop()
    r_motor.stop()

def make_black(image, threshold = 117):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_image=cv2.inRange(gray_image, threshold, 255)
    return black_image, gray_image

def path_decision(image, limit = 100):
    height, width = image.shape
    image = image[height-limit-10:height-10,:]
    height = limit -1
    width = width -1
    image = np.flipud(image)
    mask = image!=0

    white_distance = np.where(mask.any(axis=0), mask.argmax(axis=0), height)

    left=0
    right=width
    center=int((left+right)/2)
    left_sum = np.sum(white_distance[left:center-60])
    right_sum = np.sum(white_distance[center+60:right])
    forward_sum = np.sum(white_distance[center-60:center+60])
    print(left_sum, right_sum, forward_sum, end="")

    # 방향 결정 부분
    if forward_sum < 280: #300
        decision = 'b'

    elif left_sum > right_sum:
        decision = 'l'

    elif left_sum <= right_sum :
        decision = 'r'

    if forward_sum > 6800 :
        decision = 'f'

    elif (left_sum + right_sum + forward_sum) <19: #25
        decision = 'except'

    return decision

def motor_control(decision):
    if decision == 'except':
        motor_stop()
    if decision == 'f':
        motor_stop()
        time.sleep(1000)
        motor_tank(25,25)
    if decision == 'r':
        #motor_stop()
        #time.sleep(1000)
        motor_tank(29,0)
    if decision == 'b':
        #motor_stop()
        #time.sleep(1000)
        motor_tank(-30,-30)
    if decision == 'l':
        #motor_stop()
        #time.sleep(1000)
        motor_tank(0,29)
        
def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:,:] = image

def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
    
    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='File path', required=False, default = './model.tflite')
    parser.add_argument('--labels', help='labels path', required=False, default = './labels.txt')
    args = parser.parse_args()
    labels = load_labels(args.labels)
    interpreter = tflite.Interpreter(model_path = args.model)
    interpreter.allocate_tensors()

    for frame in camera.capture_continuous(rawCapture,format='bgr', use_video_port=True):
        rawCapture.truncate(0)
        key = cv2.waitKey(1) & 0xFF
        image = frame.array
        cvtimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        reimage = cv2.resize(cvtimage,(224,224), Image.ANTIALIAS)
        result = classify_image(interpreter,reimage)
        label_id, prob = result[0]
        print(labels[label_id], prob)
        cv2.putText(image, labels[label_id], (20, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
        cv2.imshow('img',reimage)
        
        if key == ord('q'):
            break

camera = picamera.PiCamera()
camera.resolution = (320,240)
camera.vflip = True
camera.hflip = True
camera.framerate = 10
rawCapture = PiRGBArray(camera, size =(320,240))
decision = None
time.sleep(0.1) #0.5
if __name__ == '__main__':
    main()

for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port=True):
    try:
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        image = frame.array
        rawCapture.truncate(0)
        black_image, gray_image = make_black(image)

        decision = path_decision(black_image)
        print(decision)
        motor_control(decision)
        cv2.rectangle(image,(0,220),(320,100),(0,255,0),3)
        cv2.imshow("image", image)
        cv2.imshow("black",black_image)

    except Exception as e:
        print('except!',e) #오류출력
        break