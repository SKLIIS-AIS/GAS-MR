from easyfl.client import BaseClient

import logging

logger = logging.getLogger(__name__)

import logging
import time
import torch
from collections import defaultdict

from easyfl.distributed.distributed import CPU

logger = logging.getLogger(__name__)


class CustomizedClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        """
        初始化自定义客户端

        Args:
            cid: 客户端ID
            conf: 配置对象
            train_data: 训练数据
            test_data: 测试数据
            device: 计算设备
            **kwargs: 其他关键字参数
        """
        super(CustomizedClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        self.is_byz = False  # 标记客户端是否为拜占庭攻击者
        self.state = defaultdict(dict)  # 存储参数状态

    def set_byz(self, is_byz: bool = True):
        """
        设置客户端为拜占庭攻击者

        Args:
            is_byz: 是否为拜占庭攻击者，默认为True
        """
        self.is_byz = is_byz
        pass

    def pretrain_setup(self, conf, device):
        """Setup loss function and optimizer before training."""
        self.simulate_straggler()
        self.model.train()
        self.model.to(device)
        loss_fn = self.load_loss_fn(conf)
        self.optimizer = self.load_optimizer(conf)  # 保存optimizer为实例变量
        if self.train_loader is None:
            self.train_loader = self.load_loader(conf)
        return loss_fn, self.optimizer  # 返回实例变量


    def train(self, conf, device=CPU):

        """Execute client training.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        print(f"Client {self.cid} starting training")
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        self.train_loss = []
        for i in range(conf.local_epoch):
            batch_loss = []
            for batched_x, batched_y in self.train_loader:
                x, y = batched_x.to(device), batched_y.to(device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = loss_fn(out, y)
                loss.backward()


                # ======== 梯度裁剪：在 optimizer.step() 之前添加 ========
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=2.0  # 论文中提到的 norm=2
                )

                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def post_train(self):
        """
        训练后处理方法
        将模型移动到CPU上
        """
        self.model.cpu()

    def get_model_structure(self, state_dict):
        """
        从模型状态字典中提取结构信息，匹配 unflatten_tensor 函数期望的格式
        Args:
            state_dict: 模型状态字典
        Returns:
            dict: 包含 name_shape_tuples 的结构信息
        """
        name_shape_tuples = [(key, value.shape) for key, value in state_dict.items()]
        struct = {
            'name_shape_tuples': name_shape_tuples
        }
        return struct
