from buildhat import *                          #빌드햇 라이브러리
import pygame                                   #키보드 입력 받는 라이브러리
import time                                     # time

l_motor = Motor('B')
r_motor = Motor('A')

def motor_tank(a,b):
    a = a/100
    b = b/100
    l_motor.pwm(-a)
    r_motor.pwm(b)

def motor_stop():
    l_motor.stop()
    r_motor.stop()


pygame.init()                                   #pygame initialize
pygame.display.set_caption("Keyboard Control")  #창 이름을 keyboard control이라 명명
screen = pygame.display.set_mode((200,200))     #창의 크기 세팅
screen.fill((0, 0, 0))                          #검은 색으로 창 채우기

def motor_control(key):                         #key값에 따라 모터 동작
    if key == 'No Input':
        motor_stop()
    if key == 'f':
        motor_tank(-30,30)
    if key == 'r':
        motor_tank(-30,0)
    if key == 'b':
        motor_tank(20,-20)
    if key == 'l':
        motor_tank(0,30)
    

try:
    exit=False
    while not exit:
        for event in pygame.event.get():        #key값 받기
            pressed = pygame.key.get_pressed()  #눌린 key
            key="No Input"
            if pressed[pygame.K_q]:
                exit = True
            if pressed[pygame.K_UP]:
                key="f"
            elif pressed[pygame.K_DOWN]:
                key="b"
            elif pressed[pygame.K_LEFT]:
                key="l"
            elif pressed[pygame.K_RIGHT]:
                key="r"
        motor_control(key)
finally:
    print("Control End")
    motor_stop()

