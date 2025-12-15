#Parallel Simulation
# create n_evns scene

import genesis as gs 
import numpy as np
#from genesis import world,physics
import torch
################ init ##########################
#gs.init(backend=gs.gpu)

gs.init(backend=gs.gpu)
############################## create a scene #####################
scene = gs.scene(
    show_viewer = True,
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),

    viewer_options = gs.options.ViewerOptions(
        camera_pos = (3.5,-1.0,2.5),
        camera_lookat = (0.0,0.0,0.5),
        camera_fov    =  40,
        max_FPS       =  60,
    ),
    show_viewer = True,
    #rigid_options = gs.options.RigidOptions(
     #   dt                  = 0.01,            
    #),
                 
)

###########################entities##############################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.MJCF(
    file = 'xml/franka_emika_panda/panda.xml',
    pos = (1.0,1.0,0.0),
    euler = (0,0,0),
    ),
)


cam = scene.add_camera(
    res = (640,480),
    pos = (3.5,0.0,2.5),
    lookat =(0,0,0.5),
    fov = 30,
    GUI =False
)

scene.build()
#render rgb,depth,segmentation and normal
#rgb,depth, segmentation,normal = cam.render(rgb=True,depth=True,segmentation=True,normal=True)
cam.start_recording()


for i in range(120):
    scene.step()
    cam.render()
cam.stop_recording(save_to_filename='video.mp4',fps=60)
###################### entities ######################################3
plane = scene.add_entity(
    gs.morphs.Plane(),
)
########################### PID Joints#####################################
jnt_names = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
    'finger_joint1',
    'finger_joint2',
]                                   # To obtain local idx of dof with respect to the robot robot itself        
dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]
################################ build ###########################
# create 20 parallel environments
B=20
scene.build(n_envs=B,env_spacing=(1.0,1.0))

#control all robots 
franka.control_dofs_postion(
        torch.tile(
            torch.tensor([0,0,0,-1.0,0,1.0,0,0.02,0.02],device=gs.device),(3000,1)
    ),
)
#returns the steps for 1000 times each step
for i in range(1000):
    scene.step()