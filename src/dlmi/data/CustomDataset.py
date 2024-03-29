from abc import ABC, abstractmethod

class CustomDataset(ABC):

    @abstractmethod
    def get_balanced_mask(self, train_size):
        """deprecated"""
        pass
    
    @abstractmethod
    def get_indices_from_patient_mask(self, mask):
        """convert a mask of patients to a mask of indices"""
        pass

    @abstractmethod
    def get_patient_labels(self, preds, mask=None, dataset="test", fold=0):
        """ Write the predictions to a csv file (function should be renamed)"""
        pass