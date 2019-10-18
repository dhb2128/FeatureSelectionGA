from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import f1_score

class FitenessFunction:
    
    def __init__(self, n_splits = 5, *args, **kwargs):
        """
            Parameters
            -----------
            n_splits :int, 
                Number of splits for cv
            
            verbose: 0 or 1
        """
        self.n_splits = n_splits
    

    def calculate_fitness(self,model,x,y):
        return cross_val_score(model, x, y, cv=self.n_splits, verbose=1).mean()