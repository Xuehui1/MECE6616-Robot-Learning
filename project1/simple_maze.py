import os.path as osp
import time

import gym
import gym.spaces as spaces
import numpy as np
import pybullet as p

from shapely import affinity
from shapely.geometry import Polygon


OBSTACLE_INFO = [
    {
        'size': np.asarray([0.1, 0.1]),
        'x': 0.3,
        'y': -0.1,
    },
    {
        'size': np.asarray([0.1, 0.1]),
        'x': 0.7,
        'y': 0.1,
    },
]

WALL_INFO = [
    {
        'size': np.asarray([.01, 0.5]),
        'x': 0.0,
        'y': 0.0,
    },
    {
        'size': np.asarray([.5, 0.01]),
        'x': 0.5,
        'y': 0.5,
    },
    {
        'size': np.asarray([.5, 0.01]),
        'x': 0.5,
        'y': -0.5,
    },
    {
        'size': np.asarray([.01, 0.5]),
        'x': 1.0,
        'y': 0.0,
    },
]

GOAL = {
    'size': np.asarray([0.05, 0.05]),
    'x': 0.1,
    'y': 0.4,
}

BLACK = [0,0,0,1]
GRAY1 = [0.9,0.9,0.9,1]

IDX = 0

class SimpleMaze(gym.Env):
    
    def __init__(self, gui_enabled=False, img_obs=True):
        super().__init__()
        self.object_ids = []
        self.obj_size_space = spaces.Box(low=0.04, high=0.04, shape=(2,))
        self.img_obs = img_obs

        if gui_enabled:
            self._physics_client = p.connect(p.GUI)  # graphical version
        else:
            self._physics_client = p.connect(p.DIRECT)  # non-graphical version
        self.reset()

    def _load_goal(self, goal_info):
        size = goal_info['size']
        x, y = goal_info['x'], goal_info['y']
        visual_shape_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[size[0], size[1], 0.01],  rgbaColor=[1, 0, 0, 1])
        position = np.asarray([x, y, 0.025])
        # create body in bullet and fix it
        self._goal_body_id = p.createMultiBody(
            basePosition=[0., 0., 0.,],
            linkMasses=[0],
            linkCollisionShapeIndices=[-1],
            linkVisualShapeIndices=[visual_shape_id],
            linkPositions=[position],
            linkParentIndices=[0],
            linkInertialFramePositions=[np.asarray([0, 0, 0])],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkOrientations=[p.getQuaternionFromEuler([0, 0, 0])],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]])

    def _load_box(self, box_info):

        size = box_info['size']
        x, y = box_info['x'], box_info['y']
        mass = 1 

        collision_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[size[0], size[1], 0.025])
        position = np.asarray([x, y, 0.025])

        visual_shape_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[size[0], size[1], 0.025], rgbaColor=BLACK)

        position = np.asarray([x, y, 0.025])

        # create body in bullet and fix it
        body_id = p.createMultiBody(
            linkMasses=[mass],
            linkCollisionShapeIndices=[collision_id],
            linkVisualShapeIndices=[visual_shape_id],
            linkPositions=[position],
            linkParentIndices=[0],
            linkInertialFramePositions=[position],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkOrientations=[p.getQuaternionFromEuler([0, 0, 0])],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]])

        self.object_ids.append(body_id)

    def _setup_top_view(self):
        self.viewMatrix = p.computeViewMatrix(cameraEyePosition=[0.5, 0, 1.3], 
                                        cameraTargetPosition=[0.5, 0, 0], 
                                        cameraUpVector=[0, 1, 0])

        self.projectionMatrix = p.computeProjectionMatrixFOV(fov=45.0, 
                                                        aspect=1.0, 
                                                        nearVal=0.1, 
                                                        farVal=3.1)
    
    def _get_top_view(self):
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=64, 
                                                                height=64, 
                                                                viewMatrix=self.viewMatrix,
                                                                projectionMatrix=self.projectionMatrix)
        return rgbImg[..., :3].copy()

    def reset(self):
        global IDX
        p.resetSimulation()
        body_ids = p.loadMJCF('mjcf/point_mass.xml')
        self._world_id = 0
        self._goal_body_id = None

        p.syncBodyInfo()
        self.object_ids = []
        for info in WALL_INFO:
            self._load_box(info)
        for info in OBSTACLE_INFO:
            self._load_box(info)

        self._setup_top_view()
        p.setTimeStep(0.1)

        dict_copy = {}
        for key, val in GOAL.items():
            dict_copy[key] = val
        # x_min = 0.2
        # x_max = 0.8
        # xs = np.linspace(x_min, x_max, 10, endpoint=True)
        dict_copy['x'] = 0.2
        # IDX = (1 + IDX) % 10
        self.goal_pos = np.asarray([dict_copy['x'], dict_copy['y']])
        if self._goal_body_id is not None:
            p.removeBody(self._goal_body_id)
        self._load_goal(dict_copy)
        rgb_img = self._get_top_view()
        if self.img_obs:
            obs = rgb_img
        else:
            obs = p.getLinkState(self._world_id, 1)[4][:2]
        return obs

    def step(self, action):
        VEL = 0.1
        # action    command     on maze
        # 0         Forward     y++
        # 1         Left        x--
        # 2         Right       x++

        x_vel = 0
        y_vel = 0

        if action == 0:
            y_vel = VEL
        elif action == 1:
            x_vel = -VEL
        elif action == 2:
            x_vel = VEL
        # set controls
        p.setJointMotorControl2(self._world_id, 0, p.VELOCITY_CONTROL, targetVelocity=x_vel)
        p.setJointMotorControl2(self._world_id, 1, p.VELOCITY_CONTROL, targetVelocity=y_vel)

        for _ in range(10):
            p.stepSimulation()

        POINT_MASS_LINK_ID = 1
        pos = p.getLinkState(self._world_id, POINT_MASS_LINK_ID)[4][:2]
        # print('pos:', pos, 'goal:', self.goal_pos)

        done = False
        dist = np.linalg.norm(self.goal_pos-pos)
        if dist < 0.1:
            done = True

        info = {
            'agent_pos': pos,
            'obstacle_infos': OBSTACLE_INFO,
            'goal': self.goal_pos,
        }
        rgb_img = self._get_top_view()
        if self.img_obs:
            obs = rgb_img
        else:
            obs = pos
        return obs, 0, done, info

    # Step through simulation time
    def step_simulation(self):
        p.stepSimulation()

    def close(self):
        p.disconnect()
