from abc import ABC, abstractmethod

class CustomDataset(ABC):

    @abstractmethod
    def get_balanced_mask(self, train_size):
        pass
    
    @abstractmethod
    def get_indices_from_patient_mask(self, mask):
        pass

    @abstractmethod
    def get_patient_labels(self, preds, mask=None, dataset="test"):
        pass