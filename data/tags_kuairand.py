import os
import torch
from torch_geometric.data import InMemoryDataset

class KuaiRand(InMemoryDataset):
    """
    A class to load the preprocessed KuaiRand dataset.
    
    This class assumes that the data has already been processed and saved
    in the torch_geometric.data.HeteroData format, located at
    self.processed_dir/self.processed_file_names[0].
    """
    def __init__(self, root: str, transform=None, pre_transform=None):
        """
        Initializes the dataset object.
        :param root: The root directory of the dataset, e.g., 'dataset/kuairand'
        """
        super().__init__(root, transform, pre_transform)
        # The constructor of the parent class InMemoryDataset checks for the existence of processed_paths.
        # If they exist, it loads the data. If not, it calls self.process().
        # Here, we assume the data already exists and will be loaded directly.
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """Raw files are not needed as we are not processing them."""
        return []

    @property
    def processed_file_names(self) -> str:
        """Returns the name of the preprocessed data file."""
        return 'title_data_kuairand_5tags.pt'

    def download(self):
        """No download is required."""
        pass

    def process(self):
        """
        Does not perform any processing.
        
        If the file pointed to by `processed_file_names` does not exist,
        the InMemoryDataset from torch_geometric will call this method.
        We raise an error here to explicitly inform the user that the file is missing.
        """
        path = self.processed_paths[0]
        raise FileNotFoundError(
            f"Processed file not found: '{path}'. "
            f"Please make sure the KuaiRand dataset has been processed and "
            f"the file is in the correct path."
        )

if __name__ == '__main__':
    # Example code for testing
    # Assume your project root is the current working directory
    dataset_root = os.path.join('dataset', 'kuairand')
    
    # Check if the processed file exists
    processed_file = os.path.join(dataset_root, 'processed', 'title_data_kuairand_5tags.pt')
    if not os.path.exists(processed_file):
        print(f"Error: Preprocessed file does not exist at '{processed_file}'")
        print("Please make sure you have generated this file.")
    else:
        print(f"Found preprocessed file: {processed_file}")
        print("Attempting to load the dataset...")
        try:
            dataset = KuaiRand(root=dataset_root)
            print("Dataset loaded successfully!")
            print("Data object:", dataset.data)
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}")
