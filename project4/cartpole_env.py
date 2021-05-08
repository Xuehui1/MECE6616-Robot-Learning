import logging
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
from math import radians
import pybullet_data


def make_cart_pole_env(action_dims=3):
    """ Wrapper function for making the env with some default values for all scripts """
    env = CartPoleEnv(timestep=0.02,
                      initial_position=[0, 0, 0],
                      initial_orientation=[0, 0, 0, 1],
                      urdf_path='cartpole.urdf',
                      angle_threshold_low=-15,
                      angle_threshold_high=15,
                      distance_threshold_low=-0.6,
                      distance_threshold_high=0.6,
                      force_mag=10,
                      action_dims=action_dims)
    return env


def step_duration(duration=1.0):
    for i in range(int(duration * 240)):
        p.stepSimulation()


def configure_pybullet(rendering=False,
                       debug=False,
                       yaw=50.0,
                       pitch=-35.0,
                       dist=1.2,
                       target=(0.0, 0.0, 0.0),
                       gravity=9.8):
    if not rendering:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if not debug:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.resetSimulation()
    p.setGravity(0, 0, -gravity)


def convert_action_to_force(action):
    action_map = {
        0: -10,
        1: 0,
        2: 10
    }
    return action_map[action]


class CartPole:
    """ A controller of the cartpole """
    CART_LINK = 0
    LOW_POLE_LINK = 1
    HIGH_POLE_LINK = 2
    SLIDER_LINK = -1
    CART_POLE_JOINT = 1
    SLIDER_CART_JOINT = 0

    def __init__(self, urdf_path, initial_position, initial_orientation):
        self.urdf_path = urdf_path
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation
        self.urdf_path = urdf_path
        self.id = p.loadURDF(self.urdf_path, initial_position, initial_orientation, useFixedBase=True)
        p.changeDynamics(self.id, -1, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.id, 0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.id, 1, linearDamping=0, angularDamping=0)

        # these two lines are important for simulation
        p.setJointMotorControl2(self.id, 1, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.id, 0, p.VELOCITY_CONTROL, force=0)

    def reset_cart_pole_joint(self, position_value, velocity_value):
        p.resetJointState(self.id, self.CART_POLE_JOINT, position_value, velocity_value)

    def reset_slider_cart_joint(self, position_value, velocity_value):
        p.resetJointState(self.id, self.SLIDER_CART_JOINT, position_value, velocity_value)

    def apply_force_on_slider_cart_joint(self, force):
        p.setJointMotorControl2(self.id, self.SLIDER_CART_JOINT, p.TORQUE_CONTROL, force=force)

    def get_cart_position(self):
        # the link pose of cart does not match prismatic joint position 100% after some simulation
        return p.getJointState(self.id, self.SLIDER_CART_JOINT)[0]

    def get_cart_velocity(self):
        return p.getJointState(self.id, self.SLIDER_CART_JOINT)[1]

    def get_pole_position(self):
        # angular joint position
        return p.getJointState(self.id, self.CART_POLE_JOINT)[0]

    def get_pole_velocity(self):
        # angular joint velocity
        return p.getJointState(self.id, self.CART_POLE_JOINT)[1]

    def reset_state(self, state):
        """ state is a list of four values: cart position + cart velocity + joint position + joint velocity """
        self.reset_slider_cart_joint(state[0], state[1])
        self.reset_cart_pole_joint(state[2], state[3])

    def get_state(self):
        """ state is a list of four values: cart position + cart velocity + joint position + joint velocity """
        return [self.get_cart_position(), self.get_cart_velocity(), self.get_pole_position(), self.get_pole_velocity()]

    def make_invisible(self):
        p.changeVisualShape(self.id, self.HIGH_POLE_LINK, rgbaColor=(0, 0, 0, 0))

    def make_visible(self):
        p.changeVisualShape(self.id, self.HIGH_POLE_LINK, rgbaColor=(0, 0, 0, 1))


class CartPoleEnv(gym.Env):
    def __init__(self,
                 urdf_path,
                 initial_position,
                 initial_orientation,
                 timestep,
                 angle_threshold_low,
                 angle_threshold_high,
                 distance_threshold_low,
                 distance_threshold_high,
                 force_mag,
                 action_dims):
        self.urdf_path = urdf_path
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation
        self.timestep = timestep
        self.angle_threshold_low = angle_threshold_low
        self.angle_threshold_high = angle_threshold_high
        self.distance_threshold_low = distance_threshold_low
        self.distance_threshold_high = distance_threshold_high
        self.force_mag = force_mag
        self.action_dims = action_dims
        self.max_episode_steps = 500

        # initialize world by loading objects
        self.cartpole = CartPole(self.urdf_path, self.initial_position, self.initial_orientation)
        self.action_space = spaces.Discrete(action_dims)
        self.observation = None
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,))
        self.seed()
        self.num_steps = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """ return observation, reward, done, info """
        self.num_steps += 1
        force = convert_action_to_force(action)
        self.cartpole.apply_force_on_slider_cart_joint(force)
        step_duration(self.timestep)

        self.observation = np.array(self.cartpole.get_state())
        # ground truth used to check done
        x, x_dot, theta, theta_dot = self.cartpole.get_state()

        # done indicates a failure or reaching max steps
        done = x < self.distance_threshold_low \
               or x > self.distance_threshold_high \
               or theta < radians(self.angle_threshold_low) \
               or theta > radians(self.angle_threshold_high) \
               or self.num_steps >= self.max_episode_steps
        done = bool(done)
        reward = 1.0
        return self.observation, reward, done, {}

    def reset(self, randstate=None):
        """ return an initial observation """
        self.num_steps = 0
        randstate = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)) if randstate is None else randstate
        self.cartpole.reset_state(randstate)

        self.observation = np.array(self.cartpole.get_state())
        # print("-----------reset simulation---------------")
        return self.observation
