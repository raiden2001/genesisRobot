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
    #show_viewer = True,
    # sim_options = gs.options.SimOptions(
    #     dt = 0.01,
    # ),

    viewer_options = gs.options.ViewerOptions(
        camera_pos = (3.5,-1.0,2.5),
        camera_lookat = (0.0,0.0,0.5),
        camera_fov    =  40,
        max_FPS       =  60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
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

# plane = scene.add_entity(
#     gs.morphs.Plane(),
# )
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

motors_dof = np.arange(7)
fingers_dof = np.arange(7,9)



for i in range(120):
    scene.step()
    cam.render()
cam.stop_recording(save_to_filename='video.mp4',fps=60)


########################### PID Joints Build #####################################
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

############ Optional: set control gains ############
# set positional gains
franka.set_dofs_kp(
    kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    dofs_idx_local = dofs_idx,
)
# set velocity gains
franka.set_dofs_kv(
    kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    dofs_idx_local = dofs_idx,
)
# set force range for safety
franka.set_dofs_force_range(
    lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    dofs_idx_local = dofs_idx,
)

#########################################hard reset############################## 
for i in range(150):
    if i < 50:
        franka.set_dofs_position(np.array([1,1,0,0,0,0,0,0.04,0.04]),dofs_idx)
    elif i < 100:
        franka.set_dofs_position(np.array([-1,0.8,1,-2,1,0.5,-0.5,0.04,0.04]),dofs_idx)
    else:
        franka.set_dofs_position(np.array([0,0,0,0,0,0,0,0,0]),dofs_idx)
    scene.step()    

############## PD Control ########################3

control_phases ={
    "reached_out" :{
    "step":0,
    "type": "position",
    "target": [1, 1, 0, 0, 0, 0, 0, 0.04, 0.04],
    "dofs":dofs_idx,
    },
    "grasp_pose":{
    "step": 250,
    "type": "position",
    "target":[-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04],
    "dofs": dofs_idx
    },
    "home" :{
    "step" : 500,
    "type":"position",
    "target":[0]*9,
    "dofs":dofs_idx,
    },
    "split":{
    "step": 750,
    "type":"mixed",
    "target":[0]*8,
    "pos_dofs":dofs_idx[1:],
    "vel_target":[1,0],
    "vel_dofs": dofs_idx[:1]   

    },
    "final":{
    "step":1000,
    "type":"force",
    "target":[0]*9,
    "dofs":dofs_idx
    }


}

for i in range(1250):
    for phase_name,config in control_phases.items():
        if i == config["step"]:
            if i == config["type"] == "position":
                franka.control_dofs_postion(np.array(config["reached-out"]),config["target"])
            elif config["type"] == "mixed":
                franka.control_dofs_postion(np.array(config["pos_target"]),config["pos_dofs"])
                franka.control_dofs_postion(np.array(config["vel_target"]),config["vel_dofs"])
            elif config["final"] == "force":
                frank.control_dofs_postion(np.array(config["force"]),config["dofs"])
            print(f"activate: phrase name:{phase_name}")
            break # break the loop
           

    #Diagnostics
    if i % 1000 == 0:
        print(f"Step {i} | COntrol force:", franka.get_dofs_control_force(dofs_idx))

    scene.step()
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

    #test
    # --- (scene and robot creation omitted, identical to the sections above) ---

# Retrieve some commonly used handles
rigid        = scene.sim.rigid_solver          # low-level rigid body solver
end_effector = franka.get_link("hand")        # Franka gripper frame
cube_link    = cube.get_link("box_baselink")   # the link we want to pick

################ Reach pre-grasp pose ################
q_pregrasp = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.13]),  # just above the cube
    quat = np.array([0, 1, 0, 0]),        # down-facing orientation
)
franka.control_dofs_position(q_pregrasp[:-2], np.arange(7))  # arm joints only
for _ in range(50):
    scene.step()

################ Attach (activate suction) ################
link_cube   = np.array([cube_link.idx],   dtype=gs.np_int)
link_franka = np.array([end_effector.idx], dtype=gs.np_int)
rigid.add_weld_constraint(link_cube, link_franka)

################ Lift and transport ################
q_lift = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.28]),  # lift up
    quat = np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(q_lift[:-2], np.arange(7))
for _ in range(50):
    scene.step()

q_place = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.4, 0.2, 0.18]),  # target place pose
    quat = np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(q_place[:-2], np.arange(7))
for _ in range(100):
    scene.step()

################ Detach (release suction) ################
rigid.delete_weld_constraint(link_cube, link_franka)
for _ in range(400):
    scene.step()

################## IK and MOTION#####################################3

#get the end-effector link
end_effector = franka.get_link('hand')

#move to pre-grasp pose
qpos = franka.inverse_kinematics(
    link = end_effector,
    pos = np.array([0.65,0.0,0.25]),
    quat = np.array([0,1,0,0]),
)
#gripper open pos
qpos[-2:] = 0.04
path = franka.plan_path(
    qpos_goal = qpos,
    num_waypoints = 200, #2s duration sets
)
#Execute the planned path
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step() #finalize the scene

#allow robot to reach last waypoint nad point 
for i in range(100):
    scene.step()

