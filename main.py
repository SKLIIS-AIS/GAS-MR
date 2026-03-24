import torch
import easyfl
import torch.distributed as dist
from server import RobustServer
from client import CustomizedClient
import numpy as np  # 添加 numpy 导入
import torch.nn as nn
from easyfl.models import BaseModel
import logging
from models import AlexNet
# ,cnn # 导入模型
logger = logging.getLogger(__name__)
CPU = "cpu"
def get_model(model_name, num_classes=100):
    if model_name == "AlexNet":
        return AlexNet(num_classes=num_classes)

config = {
    "attacker": {"byz_ratio": 0.2, "lie_z": 1.5,'type':'lie'},
    "data": {"dataset": "cifar10", "root": "./datasets", "split_type": "dir", "num_of_clients": 50},
    "server": {"rounds": 200, "clients_per_round": 5, "use_gas":True,  "use_gas_mr": False ,"gas_p":100, "base_agg": "trimmean"},
    "client": {"local_epoch": 5},
    "model": "AlexNet",
    "test_mode": "test_in_server",
    "gpu":1
}
config_file = "config.yaml"
config = easyfl.load_config(config_file, config)
easyfl.register_model(AlexNet(num_classes=10))
easyfl.register_server(RobustServer)
easyfl.register_client(CustomizedClient)
easyfl.init(config)
easyfl.run()

