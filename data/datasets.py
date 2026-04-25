"""
持续学习数据集加载器
支持 Split、Permuted、Rotated 等任务划分方式
"""
import torch
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class ContinualDataset:
    """持续学习数据集基类"""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 128,
        num_workers: int = 4,
        transform=None
    ):
        """
        Args:
            dataset_name: 数据集名称 ('cifar10', 'cifar100', 'mnist')
            batch_size: 批次大小
            num_workers: 数据加载线程数
            transform: 数据转换
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 数据转换
        if transform is None:
            if dataset_name in ['cifar10', 'cifar100']:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            else:  # MNIST
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
        else:
            self.transform = transform

        self.full_dataset = self._load_dataset()
        self.tasks = []

    def _load_dataset(self):
        """加载完整数据集"""
        if self.dataset_name == 'cifar10':
            return datasets.CIFAR10(
                root='./data', train=True, download=True,
                transform=self.transform
            )
        elif self.dataset_name == 'cifar100':
            return datasets.CIFAR100(
                root='./data', train=True, download=True,
                transform=self.transform
            )
        elif self.dataset_name == 'mnist':
            return datasets.MNIST(
                root='./data', train=True, download=True,
                transform=self.transform
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def split_tasks(self, num_tasks: int, task_type: str = 'split'):
        """
        划分任务

        Args:
            num_tasks: 任务数量
            task_type: 任务类型 ('split', 'permuted', 'rotated')
        """
        self.tasks = []
        self.task_type = task_type

        if task_type == 'split':
            self._split_tasks_by_class(num_tasks)
        elif task_type == 'permuted':
            self._create_permuted_tasks(num_tasks)
        elif task_type == 'rotated':
            self._create_rotated_tasks(num_tasks)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _split_tasks_by_class(self, num_tasks: int):
        """按类别划分任务"""
        num_classes = len(self.full_dataset.classes)
        classes_per_task = num_classes // num_tasks

        # 关键修复：使用 .clone() 避免 UserWarning
        targets = torch.tensor(self.full_dataset.targets).clone()
        indices_by_class = {}

        # 按类别分组索引
        for cls in range(num_classes):
            mask = targets == cls
            indices = mask.nonzero().squeeze().tolist()
            random.shuffle(indices)
            indices_by_class[cls] = indices

        # 创建任务
        for task_id in range(num_tasks):
            start_class = task_id * classes_per_task
            end_class = start_class + classes_per_task

            task_indices = []
            for cls in range(start_class, end_class):
                task_indices.extend(indices_by_class[cls])

            self.tasks.append(task_indices)

    def _create_permuted_tasks(self, num_tasks: int):
        """创建排列任务"""
        indices = list(range(len(self.full_dataset)))
        self.tasks = [indices.copy() for _ in range(num_tasks)]

        # 对每个任务应用不同的排列
        for task_id in range(num_tasks):
            random.shuffle(self.tasks[task_id])

    def _create_rotated_tasks(self, num_tasks: int):
        """创建旋转任务"""
        num_tasks_rotations = 10
        angles = torch.linspace(0, 90, num_tasks_rotations).tolist()

        indices = list(range(len(self.full_dataset)))
        for task_id in range(min(num_tasks, num_tasks_rotations)):
            self.tasks.append(indices.copy())
            # 注意：实际旋转在数据加载时应用

    def get_task_loaders(self, task_id: int):
        """获取特定任务的数据加载器"""
        if task_id >= len(self.tasks):
            raise ValueError(f"Task {task_id} does not exist")

        task_indices = self.tasks[task_id]

        # 如果是旋转任务，应用旋转
        if self.task_type == 'rotated' and task_id < 10:
            angle = task_id * 9  # 0, 9, 18, ... 81, 90 度
            subset = RotatedMNIST(
                self.full_dataset,
                task_indices,
                angle=angle
            )
            train_loader = DataLoader(
                subset, batch_size=self.batch_size,
                shuffle=True, num_workers=self.num_workers
            )
        else:
            subset = Subset(self.full_dataset, task_indices)
            train_loader = DataLoader(
                subset, batch_size=self.batch_size,
                shuffle=True, num_workers=self.num_workers
            )

        return train_loader

    def get_test_loader(self):
        """获取测试集加载器"""
        test_dataset = self._load_test_dataset()
        return DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers
        )

    def _load_test_dataset(self):
        """加载测试集"""
        if self.dataset_name == 'cifar10':
            return datasets.CIFAR10(
                root='./data', train=False, download=True,
                transform=self.transform
            )
        elif self.dataset_name == 'cifar100':
            return datasets.CIFAR100(
                root='./data', train=False, download=True,
                transform=self.transform
            )
        elif self.dataset_name == 'mnist':
            return datasets.MNIST(
                root='./data', train=False, download=True,
                transform=self.transform
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")


class RotatedMNIST(Subset):
    """旋转MNIST数据集包装器"""
    def __init__(self, mnist_dataset, indices, angle=0):
        super().__init__(mnist_dataset, indices)
        self.angle = angle
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomRotation((angle, angle), fill=0)  # 固定旋转
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        img = self.transforms(img)
        return img, label, self.indices[idx]


def create_dataloaders(
    dataset_name: str,
    task_type: str,
    num_tasks: int,
    batch_size: int = 128
):
    """
    创建数据加载器的便捷函数

    Returns:
        (train_loaders, test_loader, num_classes)
    """
    continual_dataset = ContinualDataset(
        dataset_name=dataset_name,
        batch_size=batch_size
    )
    continual_dataset.split_tasks(num_tasks, task_type)

    train_loaders = []
    for task_id in range(num_tasks):
        train_loader = continual_dataset.get_task_loaders(task_id)
        train_loaders.append(train_loader)

    test_loader = continual_dataset.get_test_loader()

    # 获取类别数
    num_classes = len(continual_dataset.full_dataset.classes)

    return train_loaders, test_loader, num_classes
