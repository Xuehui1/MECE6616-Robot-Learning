import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors

import time
import threading
import multiprocessing as mp
import os
from geometry import rot


class ArmGUI(object):

    def __init__(self, render_rate=50):
        self.rate = render_rate
        self.t = 0
        self.start_time = time.time()
        self.close_gui = False

    def call_back(self):
        if self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.plot(command)
        return True

    def terminate(self):
        plt.close('all')

    def __call__(self, pipe):
        print('starting plotter...')

        self._fig = plt.figure(figsize=(10, 10))
        self._ax1 = self._fig.add_subplot(1, 1, 1)

        self.pipe = pipe
        timer = self._fig.canvas.new_timer(interval=1)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        plt.show()

    def plot(self, robots):
        self._ax1.clear()
        for robot in robots:
            self.plot_robot(*robot)

        # Clock based on last robot
        robot, _ = robot
        state = robot.get_state()
        mclock = round(robot.get_t(), 3)
        rclock = round(time.time() - self.start_time, 3)
        s = "Model clock: {}s \n".format(mclock)
        s += "Real clock: {}s \n".format(rclock)

        num_links = robot.dynamics.get_num_links()
        link_lengths = robot.dynamics.get_link_lengths()
        robot_length = 0
        for i in range(0, num_links):
            robot_length += link_lengths[i]

        plt.text(x=-robot_length, y=robot_length, ha='left', va='top', s=s)

        self._fig.canvas.draw()

    def plot_robot(self, robot, color):

        p = np.zeros((2, 1))
        R = np.eye(2)
        state = robot.get_state()
        q = robot.dynamics.get_q(state)
        pos_0 = robot.dynamics.get_pos_0(state)
        num_links = robot.dynamics.get_num_links()
        link_lengths = robot.dynamics.get_link_lengths()

        lim_x = 0
        lim_y = 0
        off_x, off_y = pos_0[0], pos_0[1]

        robot_length = 0
        for i in range(0, num_links):
            robot_length += link_lengths[i]
        plt.ylim(- 1.1 * robot_length, 1.1 * robot_length)
        plt.xlim(- 1.1 * robot_length, 1.1 * robot_length)

        for i in range(0, num_links):
            R = np.dot(R, rot(q[i]))
            l = np.zeros((2, 1))
            l[0, 0] = link_lengths[i]
            p_next = p + np.dot(R, l)
            self._ax1.add_line(mlines.Line2D(
                (off_x + p[0], off_x + p_next[0]), (off_y + p[1], off_y + p_next[1]), color=color))
            p = p_next


class Renderer:
    """ send data to gui and invoke plotting """

    def __init__(self):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ArmGUI()
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

    def plot(self, data):
        send = self.plot_pipe.send
        if data is not None:
            send(data)
        else:
            send(None)
