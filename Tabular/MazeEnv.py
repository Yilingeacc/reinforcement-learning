import gym
import numpy as np
from numpy import cos, sin, radians, random
from gym.envs.classic_control import rendering as rendering
from time import sleep

MAZE_SIZE = 9
CELL_SIZE = 50
CIRCLE_SIZE = 20
WINDOW_WIDTH = CELL_SIZE * (MAZE_SIZE + 1)
WINDOW_HEIGHT = WINDOW_WIDTH
OBSTACLES = {0, 3, 7, 10, 14, 16, 21, 25, 27, 29, 30, 32, 34, 38, 41, 45, 48, 51, 54, 55, 63, 66, 68, 69, 70, 74}
END_POINT = MAZE_SIZE - 1
NEGATIVE_INFINITY = float('-inf')


def set_colors():
    """
    set colors of every parts on the canvas
    """
    global BACKGROUND_COLOR, LINE_COLOR
    global START_COLOR, END_COLOR, AGENT_COLOR
    BACKGROUND_COLOR = (255, 255, 255)
    LINE_COLOR = (0, 0, 0)
    START_COLOR = (255, 0, 0)
    END_COLOR = (0, 255, 0)
    AGENT_COLOR = (255, 0, 255)


set_colors()


def _add_attrs(geom, color):
    r, g, b = color
    geom.set_color(r/255., g/255., b/255.)


def create_canvas(canvas, color):
    rect(canvas, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2, WINDOW_WIDTH, WINDOW_HEIGHT, 0, color=color, fill=True)
    return canvas


def get_corner_points(width, height, theta):
    r = radians(theta)
    rotate_factor = complex(cos(r), sin(r))
    a = complex(width / 2, height / 2) * rotate_factor
    b = complex(width / 2, -height / 2) * rotate_factor
    c = complex(-width / 2, -height / 2) * rotate_factor
    d = complex(-width / 2, height / 2) * rotate_factor
    return a, b, c, d


def rect(canvas, x, y, width, height, theta, color, fill):
    a, b, c, d = get_corner_points(width, height, theta)
    box = rendering.make_polygon([(a.real, a.imag), (b.real, b.imag), (c.real, c.imag), (d.real, d.imag)], filled=fill)
    trans = rendering.Transform()
    trans.set_translation(x, y)
    _add_attrs(box, color)
    box.add_attr(trans)
    canvas.add_onetime(box)
    return canvas


def circle(canvas, x, y, r, color, fill):
    geom = rendering.make_circle(r, res=40, filled=fill)
    trans = rendering.Transform()
    trans.set_translation(x, y)
    _add_attrs(geom, color)
    geom.add_attr(trans)
    canvas.add_onetime(geom)
    return canvas


class MazeEnv(gym.Env):
    """
    n*n maze environment with discrete action space
    tabular reinforcement learning testbed
    """
    def __init__(self):
        self.state = None
        self.observation_space = MAZE_SIZE * MAZE_SIZE
        self.action_space = 4
        self.maze_size = MAZE_SIZE
        self.viewer = None

    def init_q_table(self):
        q_table = np.zeros([self.observation_space, self.action_space], dtype=np.float32)
        for state in range(self.observation_space):
            if not state % MAZE_SIZE or state - 1 in OBSTACLES:
                q_table[state][0] = NEGATIVE_INFINITY
            if state >= MAZE_SIZE * (MAZE_SIZE - 1) or state + MAZE_SIZE in OBSTACLES:
                q_table[state][1] = NEGATIVE_INFINITY
            if not (state + 1) % MAZE_SIZE or state + 1 in OBSTACLES:
                q_table[state][2] = NEGATIVE_INFINITY
            if state < MAZE_SIZE or state - MAZE_SIZE in OBSTACLES:
                q_table[state][3] = NEGATIVE_INFINITY
        return q_table

    def reset(self):
        while not self.state or self.state == END_POINT or self.state in OBSTACLES:
            self.state = random.randint(0, MAZE_SIZE * MAZE_SIZE)
        return self.state

    def step(self, action):
        """
        action = 0: left
        action = 1: up
        action = 2: right
        action = 3: down
        """
        reward = -1
        done = False
        next_state = self.state
        if action == 0 and self.state % MAZE_SIZE:
            next_state -= 1
        elif action == 1 and self.state < MAZE_SIZE * (MAZE_SIZE - 1):
            next_state += MAZE_SIZE
        elif action == 2 and (self.state + 1) % MAZE_SIZE:
            next_state += 1
        elif action == 3 and self.state >= MAZE_SIZE:
            next_state -= MAZE_SIZE
        if next_state in OBSTACLES:
            next_state = self.state
            reward = 0
        if next_state == END_POINT:
            reward = 0
            done = True
        self.state = next_state
        return self.state, reward, done, None

    def render(self, mode='human'):
        self.viewer = rendering.Viewer(WINDOW_WIDTH, WINDOW_HEIGHT) if not self.viewer else self.viewer
        canvas = create_canvas(self.viewer, color=BACKGROUND_COLOR)
        for i in range(MAZE_SIZE):
            for j in range(MAZE_SIZE):
                state = i * MAZE_SIZE + j
                canvas = rect(canvas, (j + 1) * CELL_SIZE, (i + 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE, 0, LINE_COLOR,
                              state in OBSTACLES)
                if state == self.state:
                    canvas = circle(canvas, (j + 1) * CELL_SIZE, (i + 1) * CELL_SIZE, CIRCLE_SIZE, AGENT_COLOR, True)
                if state == END_POINT:
                    canvas = circle(canvas, (j + 1) * CELL_SIZE, (i + 1) * CELL_SIZE, CIRCLE_SIZE, END_COLOR, True)
        sleep(0.01)
        return self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
