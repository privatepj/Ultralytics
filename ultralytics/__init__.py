# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

__version__ = "8.3.167"

import os

# Set ENV variables (place before imports)
# 设置环境变量（在导入之前）
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training
    # 默认设置，减少训练期间的CPU利用率

from ultralytics.models import NAS, RTDETR, SAM, YOLO, YOLOE, FastSAM, YOLOWorld
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "ASSETS",
    "NAS",
    "RTDETR",
    "SAM",
    "YOLO",
    "YOLOE",
    "FastSAM",
    "YOLOWorld",
    "__version__",
    "checks",
    "download",
    "settings",
)
