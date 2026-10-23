from experiments.tasks.catenoid import Catenoid
from experiments.tasks.enneper import Enneper
from experiments.tasks.cone import Cone
from experiments.tasks.implicit_shape import RockerArm, Bunny

TASKS = {
    "catenoid": Catenoid,
    "enneper": Enneper,
    "cone": Cone,
    "rockerarm": RockerArm,
    "bunny": Bunny,
}
