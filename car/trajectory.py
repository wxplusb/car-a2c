import numpy as np
from math import acos, degrees
from numpy.linalg import norm

Cyx = [(-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
Cyx1 = [(-1, 0), (0, 0), (1, 0), (0, 1), (1, 1)]
Cyx2 = [(-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1)]


def dist_point_to_segment(p, a, b):
    """distance from point to line segment

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                           .reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)


def angle(turning_points, point1):

    if len(turning_points) < 2:
        return 180

    mid_point = turning_points[-1]
    point2 = turning_points[-2]

    vect1 = [point1[1]-mid_point[1], point1[0]-mid_point[0]]
    vect2 = [point2[1]-mid_point[1], point2[0]-mid_point[0]]

    cos = round(np.dot(vect1, vect2)/(norm(vect1)*norm(vect2)), 2)

    return degrees(acos(cos))


def connect_points(p1, p2):
    if p1[0] == p2[0] and p1[1] == p2[1]:
        return [p1]

    y1, x1 = p1
    y2, x2 = p2

    num_steps = max(abs(y2 - y1), abs(x2 - x1))

    step_y = (y2-y1)/num_steps
    step_x = (x2-x1)/num_steps

    line_points = [(int(y1+step_y*i), int(x1+step_x*i))
                   for i in range(num_steps)]

    return line_points


def make_path(turning_points):

    path = []

    for p1, p2 in zip(turning_points, turning_points[1:]):
        line_points = connect_points(p1, p2)
        path.extend(line_points)

    return path


class Trajectory:
    def __init__(self, H=80, W=140, size_square=20, width_road=3, min_angle=45, color_lawn=0., color_road=0.3, color_finish=0.301, seed=None) -> None:

        self.rng = np.random.default_rng(seed)

        self.H = H
        self.W = W
        self.size_square = size_square
        self.width_road = width_road
        self.min_angle = min_angle

        self.color_lawn = color_lawn
        self.color_road = color_road
        self.color_finish = color_finish

        self.max_ysq = int(H/size_square)
        self.max_xsq = int(W/size_square)

        # occupied squares
        self.squares = np.zeros((self.max_ysq, self.max_xsq))

        # trajectory turning points
        self.turn_points = []

        self.area = np.full(
            (self.H, self.W), self.color_lawn, dtype=np.float32)

        self.path = None

    def approachable_squares(self, square):
        y, x = square
        Cd = Cyx
        if y == 1 and x == self.max_xsq - 2:
            Cd = Cyx1
        elif y == self.max_ysq - 2 and x == self.max_xsq - 2:
            Cd = Cyx2

        return [(y+dy, x+dx) for dy, dx in Cd if 0 <= y+dy < self.max_ysq and 0 <= x+dx < self.max_xsq and self.squares[y+dy, x+dx] == 0]

    def rand_coord(self, ix_square):
        return self.rng.integers(self.width_road+3, self.size_square-self.width_road).item() + ix_square*self.size_square

    def rand_point(self, square):
        return (self.rand_coord(square[0]), self.rand_coord(square[1]))

    def rand_square(self, s):
        return tuple(self.rng.choice(s))

    def choose_turn_point(self, appr_squares):

        n = 0

        while True:

            n += 1

            square = self.rand_square(appr_squares)
            turn_point = self.rand_point(square)

            if angle(self.turn_points, turn_point) > self.min_angle:
                break

        return turn_point, square

    def pave_around(self, point):
        y, x = point

        y1 = max(y-self.width_road, 0)
        y2 = min(y+self.width_road, self.H)

        x1 = max(x-self.width_road, 0)
        x2 = min(x+self.width_road, self.W)

        self.area[y1:y2+1, x1:x2+1] = self.color_road

    def pave_path(self, path):
        for point in path:
            self.pave_around(point)

    def pave_finish(self, point):
        y, x = point

        y1 = max(y-self.width_road, 0)
        y2 = min(y+self.width_road, self.H)

        self.area[y1:y2+1, x] = self.color_finish

    def build(self):

        start_square = (self.rng.integers(0, self.max_ysq).item(), 0)
        self.squares[start_square] = 1
        # trajectory = [start_square]

        second_turn_point = self.rand_point(start_square)
        start_turn_point = (second_turn_point[0], 0)

        self.turn_points.extend([start_turn_point, second_turn_point])

        square = start_square
        num_step = 1

        while True:
            sqs = self.approachable_squares(square)
            if not sqs:
                break

            turn_point, square = self.choose_turn_point(sqs)

            self.turn_points.append(turn_point)
            # trajectory.append(square)

            num_step += 1
            self.squares[square] = num_step

        # add finish point
        finish_point = (self.turn_points[-1][0], self.W-1)
        self.turn_points.append(finish_point)

        self.path = make_path(self.turn_points)
        self.pave_path(self.path)

        # highlight the finish line
        self.pave_finish(finish_point)

        return self
