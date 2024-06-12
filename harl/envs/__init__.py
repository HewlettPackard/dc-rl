from absl import flags
from harl.envs.sustaindc.sustaindc_logger import SustainDCLogger

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "sustaindc": SustainDCLogger,
}
