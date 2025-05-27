import math

import numpy as np

from math import radians, sin, cos
from pygame.math import Vector2


def get_angle(vec):
    if vec.length() == 0:
        return 0
    
    return math.degrees(math.atan2(vec.y, vec.x))


def linesCollided(x1, y1, x2, y2, x3, y3, x4, y4):
    if (y4-y3) == 0 or (x4-x3) == 0:
        uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((1) * (x2 - x1) - (1) * (y2 - y1))
        uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((1) * (x2 - x1) - (1) * (y2 - y1))
        if 0 <= uA <= 1 and 0 <= uB <= 1:
            return True
        return False
    else:
        uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        if 0 <= uA <= 1 and 0 <= uB <= 1:
            return True
        return False


def getCollisionPoint(x1, y1, x2, y2, x3, y3, x4, y4):
    if (y4-y3) == 0 or (x4-x3) == 0:
        uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((1) * (x2 - x1) - (1) * (y2 - y1))
        uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((1) * (x2 - x1) - (1) * (y2 - y1))
        if 0 <= uA <= 1 and 0 <= uB <= 1:
            intersectionX = x1 + (uA * (x2 - x1))
            intersectionY = y1 + (uA * (y2 - y1))
            return Vector2(intersectionX, intersectionY)
        return None
    else:
        uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        if 0 <= uA <= 1 and 0 <= uB <= 1:
            intersectionX = x1 + (uA * (x2 - x1))
            intersectionY = y1 + (uA * (y2 - y1))
            return Vector2(intersectionX, intersectionY)
        return None


def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_direction(angle):
        if angle > 180 and angle < 360:
            angle = angle - 180 * int(angle / 180)
            angle = 180 - angle
            angle = -angle
        elif angle >= 360:
            angle = angle - 360 * int(angle / 360)
            if angle > 180:
                angle = angle - 180 * int(angle / 180)
                angle = 180 - angle
                angle = -angle
        elif angle < -180 and angle > -360:
            angle = angle - 180 * int(angle / 180)
            angle = 180 - angle
            angle = -angle
        elif angle <= -360:
            angle = angle - 360 * int(angle / 360)
            if angle > 180:
                angle = angle - 180 * int(angle / 180)
                angle = 180 - angle
        angle = radians(angle)
        
        return -Vector2(sin(angle), cos(angle))
    
    
def angle_to_wind(angle):
        if angle > 180 and angle < 360:
            angle = angle - 180 * int(angle / 180)
            angle = 180 - angle
        elif angle >= 360:
            angle = angle - 360 * int(angle / 360)
            if angle > 180:
                angle = angle - 180 * int(angle / 180)
                angle = 180 - angle
        elif angle < 0 and angle >= -180:
            angle = abs(angle)
        elif angle < -180 and angle > -360:
            angle = abs(angle)
            angle = angle - 180 * int(angle / 180)
            angle = 180 - angle
        elif angle <= -360:
            angle = abs(angle)
            angle = angle - 360 * int(angle / 360)
            if angle > 180:
                angle = angle - 180 * int(angle / 180)
                angle = 180 - angle
                
        return radians(angle)
    

def vel1(theta, theta_0=0, theta_dead=np.pi / 12):
    return 1 - np.exp(-(theta - theta_0) ** 2 / theta_dead)

def vel2(theta):
    return theta/(theta + 1) * 1.64

def vel3(theta):
    return theta/(theta - 0.2) * 0.975

def vel4(theta):
    return theta/(theta - 0.8) * 0.704
    


