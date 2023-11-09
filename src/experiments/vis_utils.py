import enum

from src.experiments.perception import *
from src.experiments.transform import Rotation, Transform

def workspace_lines(size):
    return [
        [0.0, 0.0, 0.0],
        [size, 0.0, 0.0],
        [size, 0.0, 0.0],
        [size, size, 0.0],
        [size, size, 0.0],
        [0.0, size, 0.0],
        [0.0, size, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, size],
        [size, 0.0, size],
        [size, 0.0, size],
        [size, size, size],
        [size, size, size],
        [0.0, size, size],
        [0.0, size, size],
        [0.0, 0.0, size],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, size],
        [size, 0.0, 0.0],
        [size, 0.0, size],
        [size, size, 0.0],
        [size, size, size],
        [0.0, size, 0.0],
        [0.0, size, size],
    ]

class Label(enum.IntEnum):
    FAILURE = 0  # grasp execution failed due to collision or slippage
    SUCCESS = 1  # object was successfully removed


class Grasp(object):
    def __init__(self, pose, width, metric=-1, uid=None, success=False):
        self.pose = pose
        self.width = width
        self.metric = metric
        self.uid = uid
        self._success = success
        self._left_contact = None
        self._right_contact = None
        self._visible = None

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, visible):
        self._visible = visible

    @property
    def success(self):
        return self._success

    def switch_contacts(self):
        self.pose = self.pose * Transform(Rotation.from_euler('z', np.pi), [0, 0, 0])
        lc = self.left_contact
        rc = self.right_contact
        self.left_contact = rc
        self.right_contact = lc

    @success.setter
    def success(self, success):
        self._success = success

    @property
    def left_contact(self):
        return self._left_contact

    @left_contact.setter
    def left_contact(self, contact):
        self._left_contact = contact

    @property
    def right_contact(self):
        return self._right_contact

    @right_contact.setter
    def right_contact(self, contact):
        self._right_contact = contact

def create_csv(path, columns):
    with path.open("w") as f:
        f.write(",".join(columns))
        f.write("\n")

def append_csv(path, *args):
    row = ",".join([str(arg) for arg in args])
    with path.open("a") as f:
        f.write(row)
        f.write("\n")