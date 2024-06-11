from absl import flags
from harl.envs.dcrl.dcrl_logger import DCRLLogger

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "dcrl": DCRLLogger,
}
