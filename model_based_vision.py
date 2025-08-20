from gui import *
import cv2
import torch
import numpy as np
from torch import sin, cos

def vec(us):
    return torch.cat(tuple(torch.unsqueeze(u, 0) for u in us), 0)

def project_point(pose, f, p):
    [[theta, phi, psi], [x, y, t]] = pose
    theta = deg(theta)
    phi = deg(phi)
    psi = deg(psi)
    Rtheta = vec(
        (vec((torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0))),
         vec((torch.tensor(0.0), cos(theta), sin(theta))),
         vec((torch.tensor(0.0), -sin(theta), cos(theta)))))
    Rphi = vec((vec((cos(phi), torch.tensor(0.0), sin(phi))),
                vec((torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0))),
                vec((-sin(phi), torch.tensor(0.0), cos(phi)))))
    Rpsi = vec((vec((cos(psi), sin(psi), torch.tensor(0.0))),
                vec((-sin(psi), cos(psi), torch.tensor(0.0))),
                vec((torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)))))
    R = torch.matmul(Rtheta, torch.matmul(Rphi, Rpsi))
    t = vec((x, y, t))
    [x, y, z] = torch.matmul(R, p-t)
    print([x, y, z])
    u = f*x/z
    v = f*y/z
    print([u, v])
    return vec((u, v))

def project_line(pose, f, l):
    [p1, p2] = l
    i1 = project_point(pose, f, p1)
    i2 = project_point(pose, f, p2)
    return vec((i1, i2))

def project_model(pose, f, m):
    return vec(tuple(project_line(pose, f, l) for l in m))

def deg(d): return np.pi*d/180

pose = vec((vec((torch.tensor(0.0),
                 torch.tensor(0.0),
                 torch.tensor(0.0))),
            vec((torch.tensor(0.0),
                 torch.tensor(0.0),
                 torch.tensor(-100.0)))))

f = torch.tensor(1000.0)

model = vec((vec((vec((torch.tensor(0.0),
                       torch.tensor(0.0),
                       torch.tensor(0.0))),
                  vec((torch.tensor(1.0),
                       torch.tensor(0.0),
                       torch.tensor(0.0))))),
             vec((vec((torch.tensor(0.0),
                       torch.tensor(0.0),
                       torch.tensor(0.0))),
                  vec((torch.tensor(0.0),
                       torch.tensor(2.0),
                       torch.tensor(0.0))))),
             vec((vec((torch.tensor(0.0),
                       torch.tensor(0.0),
                       torch.tensor(0.0))),
                  vec((torch.tensor(0.0),
                       torch.tensor(0.0),
                       torch.tensor(3.0))))),
             vec((vec((torch.tensor(1.0),
                       torch.tensor(0.0),
                       torch.tensor(0.0))),
                  vec((torch.tensor(1.0),
                       torch.tensor(2.0),
                       torch.tensor(0.0))))),
             vec((vec((torch.tensor(1.0),
                       torch.tensor(0.0),
                       torch.tensor(0.0))),
                  vec((torch.tensor(1.0),
                       torch.tensor(0.0),
                       torch.tensor(3.0))))),
             vec((vec((torch.tensor(0.0),
                       torch.tensor(2.0),
                       torch.tensor(0.0))),
                  vec((torch.tensor(1.0),
                       torch.tensor(2.0),
                       torch.tensor(0.0))))),
             vec((vec((torch.tensor(0.0),
                       torch.tensor(2.0),
                       torch.tensor(0.0))),
                  vec((torch.tensor(0.0),
                       torch.tensor(2.0),
                       torch.tensor(3.0))))),
             vec((vec((torch.tensor(0.0),
                       torch.tensor(0.0),
                       torch.tensor(3.0))),
                  vec((torch.tensor(1.0),
                       torch.tensor(0.0),
                       torch.tensor(3.0))))),
             vec((vec((torch.tensor(0.0),
                       torch.tensor(0.0),
                       torch.tensor(3.0))),
                  vec((torch.tensor(0.0),
                       torch.tensor(2.0),
                       torch.tensor(3.0))))),
             vec((vec((torch.tensor(1.0),
                       torch.tensor(2.0),
                       torch.tensor(0.0))),
                  vec((torch.tensor(1.0),
                       torch.tensor(2.0),
                       torch.tensor(3.0))))),
             vec((vec((torch.tensor(0.0),
                       torch.tensor(2.0),
                       torch.tensor(3.0))),
                  vec((torch.tensor(1.0),
                       torch.tensor(2.0),
                       torch.tensor(3.0))))),
             vec((vec((torch.tensor(1.0),
                       torch.tensor(0.0),
                       torch.tensor(3.0))),
                  vec((torch.tensor(1.0),
                       torch.tensor(2.0),
                       torch.tensor(3.0)))))))

def process():
    image = np.zeros((height, width, 3), np.uint8)
    lines = project_model(pose, f, model)
    for [[x1, y1], [x2, y2]] in lines:
        cv2.line(image,
                 (width/2.0+x1, height/2.0-y1),
                 (width/2.0+x2, height/2.0-y2),
                 (255, 255, 255),
                 2)
    get_window().show_image(image)

def dummy_command():
    message("")

def decrement_theta():
    global pose
    pose[0][0] -= 10
    print(pose)
    process()

def increment_theta():
    global pose
    pose[0][0] += 10
    print(pose)
    process()

def decrement_phi():
    global pose
    pose[0][1] -= 10
    print(pose)
    process()

def increment_phi():
    global pose
    pose[0][1] += 10
    print(pose)
    process()

def decrement_psi():
    global pose
    pose[0][2] -= 10
    print(pose)
    process()

def increment_psi():
    global pose
    pose[0][2] += 10
    print(pose)
    process()

def decrement_x():
    global pose
    pose[1][0] -= 10
    print(pose)
    process()

def increment_x():
    global pose
    pose[1][0] += 10
    print(pose)
    process()

def decrement_y():
    global pose
    pose[1][1] -= 10
    print(pose)
    process()

def increment_y():
    global pose
    pose[1][1] += 10
    print(pose)
    process()

def decrement_z():
    global pose
    pose[1][2] -= 10
    print(pose)
    process()

def increment_z():
    global pose
    pose[1][2] += 10
    print(pose)
    process()

add_button(0, 4, "Dummy", dummy_command, nothing)
add_button(0, 5, "Exit", done, nothing)
add_button(1, 0, "-Theta", decrement_theta, nothing)
add_button(1, 1, "+Theta", increment_theta, nothing)
add_button(1, 2, "-Phi", decrement_phi, nothing)
add_button(1, 3, "+Phi", increment_phi, nothing)
add_button(1, 4, "-Psi", decrement_psi, nothing)
add_button(1, 5, "+Psi", increment_psi, nothing)
add_button(2, 0, "-X", decrement_x, nothing)
add_button(2, 1, "+X", increment_x, nothing)
add_button(2, 2, "-Y", decrement_y, nothing)
add_button(2, 3, "+Y", increment_y, nothing)
add_button(2, 4, "-Z", decrement_z, nothing)
add_button(2, 5, "+Z", increment_z, nothing)
message = add_message(1, 0, 6)
width = 1920
height = 1080
start_video(width, height, 3, 6)
process()
