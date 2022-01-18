import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import car.trajectory as t
from torchvision.utils import save_image
import torch
from math import ceil, isclose
import os
import random


def set_seed(seed=10):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ACTIONS = list(range(9))
ACTIONS = np.array([(1, -1), (1, 0), (1, 1),
                    (0, -1), (0, 0), (0, 1),
                    (-1, -1), (-1, 0), (-1, 1)])
# actions
# 0 1 2
# 3 4 5
# 6 7 8


class Env:
    def __init__(self, custom_tr=None, marks=None, reward_power=1.8, max_back_steps=1, gamma=0.9, beta=0.77, seed=None, debug=False) -> None:

        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.custom_tr = custom_tr

        if custom_tr is None:
            custom_tr = t.Trajectory(seed=self.seed).build()

        self.tr = custom_tr
        self.area = None

        self.marks = {"current_loc": 1.,
                      "previous_loc": 0.6,
                      "backward": -3.,
                      "soft_stop": -10.0,
                      "hard_stop": -0.7,
                      "finish": 0}

        if marks is not None:
            self.marks.update(marks)

        # not changes
        self.gamma = gamma
        self.beta = beta
        self.reward_power = reward_power
        self.field = 40
        self.shift = self.field*2
        self.debug = debug
        self.max_back_steps = max_back_steps

    def reset_state(self):

        if self.custom_tr is not None:
            self.tr = self.custom_tr
        else:
            self.tr = t.Trajectory(seed=self.seed).build()

        self.max_steps = len(self.tr.path)*1.5
        self.num_step = 0
        self.done = False
        self.back_steps = 0
        self.mv_velosity = 0
        self.rews = []

        shift = self.shift

        self.area = np.full((self.tr.H + 2*shift, self.tr.W +
                            2*shift), self.tr.color_lawn, dtype=np.float32)

        self.area[shift:self.tr.H+shift, shift:self.tr.W+shift] = self.tr.area

        # self.turn_points = np.array(self.tr.turn_points) + shift

        # initial velocity
        self.delta = np.zeros(2, dtype="int32")

        # finish point
        yf, xf = self.tr.turn_points[-1]
        # print(self.tr.turn_points[-1], self.tr.path[-1])

        self.area[yf+shift-self.tr.width_road:yf+shift +
                  self.tr.width_road+1, xf+shift+1:] = self.tr.color_finish

        self.rand_rotate()
        # start point
        self.loc = self.turn_points[0]

        # self.way=[self.loc]
        # self.velocities = []

    def reset(self):

        self.reset_state()

        y, x = self.loc
        f = self.field

        self.area[y, x] = self.marks["current_loc"]

        obs = self.area[y-f:y+f+1, x-f:x+f+1].copy()

        self.area[y, x] = self.marks["previous_loc"]

        return obs

    def rand_rotate(self):
        n = self.rng.integers(-1, 3)

        self.rand_angle = n
        self.h, self.w = (self.tr.W, self.tr.H) if abs(
            self.rand_angle) == 1 else (self.tr.H, self.tr.W)

        s0, s1 = self.area.shape

        self.area = np.rot90(self.area, n)
        shift = self.shift

        if n == -1:
            self.turn_points = np.array(
                [(x+shift, s0-1-(y+shift)) for y, x in self.tr.turn_points])
        elif n == 1:
            self.turn_points = np.array(
                [(s1-1-(x+shift), y+shift) for y, x in self.tr.turn_points])
        elif n == 2:
            self.turn_points = np.array(
                [(s0-1-(y+shift), s1-1-(x+shift)) for y, x in self.tr.turn_points])
        else:
            self.turn_points = np.array(
                [(y+shift, x+shift) for y, x in self.tr.turn_points])

    def sample(self):
        return self.rng.integers(len(ACTIONS)).item()

    def closest_segment(self, point):

        a = self.turn_points[:-1]
        b = self.turn_points[1:]

        dists = t.dist_point_to_segment(point, a, b)

        ix = np.argmin(dists)

        return b[ix] - a[ix]

    def check_backward_move(self):

        vect_seg = self.closest_segment(self.loc)

        if np.dot(self.delta, vect_seg) < 0:
            return True

        return False

    def step(self, action):

        if self.done:
            raise RuntimeError("action after done. need to reset")

        self.num_step += 1

        prev_loc = self.loc
        act = ACTIONS[action]

        self.delta += act
        velosity = norm(self.delta)

        self.mv_velosity = (1 - self.beta) * velosity + \
            self.beta * self.mv_velosity

        self.loc = prev_loc + self.delta

        y, x = self.loc
        f = self.field

        info = {"truncated": False, "finished": False,
                "velosity": velosity, "reason": None}

        atol = 0.0001

        if isclose(self.area[y, x], self.tr.color_road, abs_tol=atol):
            self.area[y, x] = self.marks["current_loc"]

        # position in which decisions will be made
        yD, xD = self.loc + self.delta
        obs = self.area[yD-f:yD+f+1, xD-f:xD+f+1].copy()

        if isclose(self.area[y, x], self.marks["current_loc"], abs_tol=atol):
            self.area[y, x] = self.marks["previous_loc"]

        # taken into account here prev_loc may = self.loc
        # move_points includes both points on the edges
        move_points = t.connect_points(prev_loc, self.loc)
        move_points.append(self.loc)

        if self.debug:
            print(
                f"  act={act}, prev_loc={prev_loc}, delta={self.delta},  loc={self.loc},\n move_points={move_points}")

        reward = min(30, round(velosity**self.reward_power, 2))

        for po in move_points:
            y, x = po

            # if go out to lawn
            if isclose(self.area[y, x], self.tr.color_lawn, abs_tol=atol):
                self.area[y, x] = self.marks["current_loc"]
                info["reason"] = "lawn"
                self.done = True

                if len(self.rews) < 3:
                    reward = -15
                else:
                    reward = self.adjust_reward(reward, 2)

                return obs, reward, self.done, info

            # if finish
            if isclose(self.area[y, x], self.tr.color_finish, abs_tol=atol):
                self.area[y, x] = self.marks["current_loc"]
                info["reason"] = "finished"
                self.done = True

                info["finished"] = True
                return obs, reward + self.marks["finish"], self.done, info

        # truncated if it wasn't finish or lawn
        if self.num_step > self.max_steps:
            info["truncated"] = True
            info["reason"] = "max_steps"
            self.done = True
            return obs, reward, self.done, info

        # penalty for backward move
        if self.check_backward_move():
            self.back_steps += 1
            if self.back_steps > self.max_back_steps:
                info["truncated"] = True
                info["reason"] = "backward"
                self.done = True

            if len(self.rews) < 3:
                reward = -15
            else:
                reward = self.adjust_reward(reward, 1)

            self.rews.append(reward)

            return obs, reward, self.done, info
        else:
            self.back_steps = 0

        if reward == 0:
            # penalty for stop
            reward = self.marks["soft_stop"]
            info["truncated"] = True
            info["reason"] = "soft_stop"
            self.done = True

        self.rews.append(reward)

        return obs, reward, self.done, info

    def adjust_reward(self, reward, h):

        k = (self.mv_velosity / (1 - self.beta**self.num_step)) / 2

        k = int(round(k)) + h

        old = 0
        for rw in self.rews[:-1-k:-1]:
            old = rw + old * self.gamma

        if old/(self.gamma**k) < reward:
            reward = - reward
        else:
            reward = - old/(self.gamma**k)

        return reward

    @staticmethod
    def render(obs, figsize=None):

        if not isinstance(obs, list):
            obs = [obs]

        l = len(obs)
        cols = min(3, l)
        rows = max(ceil(l//3), 1)

        fsize = figsize if figsize else (7*cols, 7*rows)

        _, axes = plt.subplots(rows, cols, figsize=fsize)

        axes = np.array(axes)

        for i, ax in enumerate(axes.flat):
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.yaxis.set_major_locator(MultipleLocator(20))
            ax.grid(which='both')
            ax.imshow(obs[i], origin="lower")

    def area_trajectory(self):
        if self.area is None:
            print("Need reset to show trajectory")
            return np.array([0])
        # copy to avoid negative strides
        return self.area[self.shift:self.h +
                         self.shift, self.shift:self.w+self.shift].copy()

    def render_area(self, figsize=(15, 15)):
        Env.render(self.area_trajectory(), figsize)

    def save_area_to_image(self, name):
        os.makedirs("images", exist_ok=True)

        save_image(torch.as_tensor(self.area_trajectory()),
                   f'images/{name}.png')


class VecEnv:
    def __init__(self, make_env_fn, num_env=3) -> None:

        self.num_env = num_env
        self.envs = []
        self._gamma = 0.9
        for _ in range(num_env):
            self.envs.append(make_env_fn())

    def reset(self):
        states = []
        for env in self.envs:
            states.append(env.reset())
        return states

    def step(self, actions):
        assert len(actions) == self.num_env

        states = []
        rewards = []
        dones = []
        infos = []

        for i, action in enumerate(actions):

            if self.envs[i].done:
                state = self.envs[i].reset()
                reward = 0
                done = False
                info = {"truncated": False}
            else:
                state, reward, done, info = self.envs[i].step(action)

            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info["truncated"])

        return states, rewards, dones, infos

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, g):
        self._gamma = g
        for env in self.envs:
            env.gamma = g
