import os
import torch
from torch_geometric.data import InMemoryDataset

class KuaiRand(InMemoryDataset):
    """
    一个用于加载预处理好的KuaiRand数据集的类。
    
    这个类假设数据已经被处理并保存为torch_geometric.data.HeteroData格式，
    位于 self.processed_dir/self.processed_file_names[0] 路径下。
    """
    def __init__(self, root: str, transform=None, pre_transform=None):
        """
        初始化
        :param root: 数据集根目录, e.g., 'dataset/kuairand'
        """
        super().__init__(root, transform, pre_transform)
        # InMemoryDataset的父类构造函数会检查processed_paths是否存在。
        # 如果存在，它会加载数据。如果不存在，它会调用self.process()。
        # 这里我们假设数据已存在，因此会直接加载。
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """不需要原始文件，因为我们不进行处理。"""
        return []

    @property
    def processed_file_names(self) -> str:
        """返回预处理好的数据文件名。"""
        return 'title_data_kuairand_5tags.pt'

    def download(self):
        """不需要下载。"""
        pass

    def process(self):
        """
        不进行任何处理。
        
        如果 `processed_file_names` 指向的文件不存在，
        torch_geometric的InMemoryDataset会调用此方法。
        我们在这里抛出一个错误，以明确告知用户文件缺失。
        """
        path = self.processed_paths[0]
        raise FileNotFoundError(
            f"预处理文件未找到: '{path}'. "
            f"请确保KuaiRand数据集已经处理完毕，并且文件位于正确的路径下。"
        )

if __name__ == '__main__':
    # 用于测试的示例代码
    # 假设您的项目根目录是当前工作目录
    dataset_root = os.path.join('dataset', 'kuairand')
    
    # 检查processed文件是否存在
    processed_file = os.path.join(dataset_root, 'processed', 'title_data_kuairand_5tags.pt')
    if not os.path.exists(processed_file):
        print(f"错误: 预处理文件不存在于 '{processed_file}'")
        print("请确保您已经生成了该文件。")
    else:
        print(f"找到预处理文件: {processed_file}")
        print("尝试加载数据集...")
        try:
            dataset = KuaiRand(root=dataset_root)
            print("数据集加载成功！")
            print("数据对象:", dataset.data)
        except Exception as e:
            print(f"加载数据集时发生错误: {e}")
