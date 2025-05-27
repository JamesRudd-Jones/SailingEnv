import numpy as np
from math import radians
import scipy.optimize

angle = 100

def vel1(theta, theta_0=0, theta_dead=np.pi / 12):
    return 1 - np.exp(-(theta - theta_0) ** 2 / theta_dead)
    
def vel2(theta):
    return theta/(theta + 1) * 1.64

def vel3(theta):
    return theta/(theta - 0.2) * 0.975

def vel4(theta):
    return theta/(theta - 0.8) * 0.704
    

def rew(angle, theta, theta_0=0, theta_dead=np.pi / 12):
    if angle_to_wind(angle) <= 7*np.pi/36:
        return vel1(theta, theta_0, theta_dead) * np.cos(theta)
    elif angle_to_wind(angle) > 7*np.pi/36 and angle_to_wind(angle) <= 5*np.pi/8:
        return vel2(theta)
    elif angle_to_wind(angle) > 5*np.pi/8 and angle_to_wind(angle) <= 3*np.pi/4:
        return vel3(theta)
    elif angle_to_wind(angle) > 3*np.pi/4 and angle_to_wind(angle) <= np.pi:
        return vel4(theta)




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

#speed = rew(angle, angle_to_wind(angle)) * 2

max_speed_angle = scipy.optimize.fmin(lambda x: -rew(x, angle_to_wind(x)), 0, disp=False)
max_speed = rew(max_speed_angle, angle_to_wind(max_speed_angle)) * 4

#print(max_speed * 10)