"""
Select train, validation and test sets
"""
# Seed
import random
random.seed(2)

from collections import namedtuple
from collections import Counter
from typing import List, Union, Optional, Tuple, Dict,NamedTuple
from sklearn.model_selection import train_test_split

# namedtuple to reorder datasets based on 'exlusive_on' attribute
VarIdSet = namedtuple("VarIdSet", ["var","id","set"])

class DataSelector:
    """Get train, val and test sets"""
    def __init__(self,
                    id_labels: List[Union[str,int]], 
                    labels: List[Union[str,int]],
                ):
        self._id = range(len(labels))
        self.id_labels  = id_labels
        self.labels = labels
        self.datasets = {}

    def __call__(self, 
                    train_size: float = 0.7,
                    test_size: float = None,
                    balanced_on: Optional[List[Union[str,int]]] = None,
                    exclusive_on: Optional[List[Union[str,int]]] = None,
                    print_summary: bool = True,
                    ):
        print("Generating train, validation and test sets...")
        # Generate train, val and test sets
        self.train_val_test_split(train_size, test_size, balanced_on)

        # Reorder train, val and test sets based on 'exclusive_on'
        if exclusive_on:
            print("Reordering dataset based on 'exclusive_on'")
            self.datasets_mutually_exclusive_on(exclusive_on)
        else:
            # Assing id_labels and  labels to train, test and val sets
            self.__update_datasets()
        print("Datasets successfully generated. See 'datasets' attribute.")
        
        # Print summary of labels if desired
        print(self.get_summary_labels()) if print_summary else None

    def train_val_test_split(self, 
                                train_size: float = 0.7, 
                                test_size: Optional[float] = None, 
                                balanced_on: List[Union[str,int]] = None,
                            ):
        """Get indexes for train, val and test sets"""
        X_dataset = self.id_labels if self.id_labels else self._id
        X = self._id
        y = self.labels if balanced_on is None else balanced_on
        test_size = test_size if test_size else (1 - train_size) / 2
        
        # train+val and test sets
        X_train_val, X_test, y_train_val, y_test = self.__split_dataset(X, y, test_size, balanced_on)
        # split train and val
        balanced_on = y_train_val if balanced_on else None
        X_train, X_val, y_train, y_val = self.__split_dataset(X_train_val, y_train_val, test_size / (1-test_size), balanced_on)
        
        # distribution of indexes in train, val and test sets
        self._id_datasets = {
            "train": [self._id[idx] for idx in X_train],
            "val"  : [self._id[idx] for idx in X_val],
            "test" : [self._id[idx] for idx in X_test]
        }

    def __split_dataset(self, 
                            X: List[Optional[Union[str,int]]], 
                            y: List[Optional[Union[str,int]]], 
                            perc: float, 
                            balanced_on: List[Optional[Union[str,int]]] = None
                        ) -> Tuple:
        """Split one dataset in 2 independent datasets
        Used in train_val_test_split function"""
        X1, X2, y1, y2 = train_test_split(X, y, 
                                        test_size = perc, 
                                        stratify=balanced_on
                                        )
        return X1, X2, y1, y2

    def __count_labels_on(self, labels: List[Union[int,str]]) -> Dict:
        """Count frequency of each label in a list"""
        return dict(Counter(labels))
    
    def get_summary_labels(self,):
        """Count frequency of labels in each dataset"""
        return self._get_summary_var(self.labels)

    def _split_var_on_datasets(self, list_var: List[Union[str,int]]) -> Dict[str,List]:
        """Split a list of variables into train, val and test sets
            using the order obtained in attribute _id_datasets.
        """
        datasets_var = {}
        for name, list_idx in self._id_datasets.items(): 
            datasets_var[name] = [list_var[idx] for idx in list_idx]
        return datasets_var

    def _get_summary_var(self, list_var: List[Union[str,int]]):
        """Count frequency of labels in each dataset"""
        new_dict = self._split_var_on_datasets(list_var)
        return {ds: self.__count_labels_on(new_dict.get(ds)) 
                    for ds in ["train","val","test"]
                }

    def __tuples_var_id_set(self, list_var: List[Union[str,int]]) -> List[NamedTuple]:
        # For datasets_mutually_exclusive_on
        """Generate namedtuples with VarIdSet(var, id, set)
        var : element of 'list_var'
        id  : the corresponding index in attribute _id_datasets for the element var
        set : the set where the element 'var' belongs (either 'train', 'test' or 'val')
        """
        #TODO: usar namedtuple
        new_list = []
        # Reorder list_var in train, test and val sets
        distribution_var = self._split_var_on_datasets(list_var)
        for set_ in ["train","test","val"]:
            set_dist = distribution_var.get(set_)
            set_id   = self._id_datasets.get(set_)
            new_list.extend([VarIdSet(var_, id_, set_) for var_, id_ in zip(set_dist,set_id)])
        return new_list

    def load_id_datasets(self, _id_datasets: Dict[str, List[int]]):
        """Cargar los id de los datasets desde un diccionario
        Util para reordenar una asignacion ya hecha.
        _id_datasets debe contener las llaves "train", "val" y "test"
        """
        assert all([set_ in ["train","test","val"] for set_ in _id_datasets.keys()])
        self._id_datasets = _id_datasets
        
        # Assing id_labels and  labels to train, test and val sets
        self.__update_datasets()

    def __update_datasets(self,):
        """Assing id_labels and labels to train, test and val sets
        based on attribute _id_datasets"""
        # id_labels and labels distributed in train, val and test sets
        self.datasets["id_labels"] = self._split_var_on_datasets(self.id_labels)
        self.datasets["labels"]    = self._split_var_on_datasets(self.labels)

    #FIXME: From here, work in progress ------------------
    def datasets_mutually_exclusive_on(self, exclusive_on: List[Union[str,int]]) -> Dict[str,int]:
        #TODO: find a better approach to do this
        """Redefine train, val, test datasets based on exclusive_on list.
        Each unique element of exclusive_on will be in only one set (either train, val or test)
        """
        dict_id = self.__reorder_datasets(exclusive_on)
        self._id_datasets = dict_id
        
        # Assing id_labels and  labels to train, test and val sets
        self.__update_datasets()
        return dict_id
        
    def __reorder_datasets(self, list_var):
        # For datasets_mutually_exclusive_on
        """what do you do"""
        # 
        new_assign = self.__reasign_id(list_var)
        dict_id = {}
        # Generate dictionary of ids with new order for train, test and val sets
        for set_ in ["train","test","val"]:
            dict_id[set_] = [x.id for x in new_assign if x.set == set_]
        return dict_id

    def __reasign_set(self, sets_of_var: List[str]) -> List[str]:
        # For datasets_mutually_exclusive_on
        """Reasign sets in order of preferences: test -> val -> train
        sets_of_var must be a list which elements are either 'train', 'test' or 'val'"""
        if 'test' in sets_of_var:
            sets_of_var = ['test' for i in sets_of_var]
        elif 'val' in sets_of_var:
            sets_of_var = ['val' for i in sets_of_var]
        return sets_of_var

    def __reasign_id(self, list_var: List[NamedTuple]):
        # For datasets_mutually_exclusive_on
        """Reasign id"""
        new_assign = []
        new_list_varidset   = self.__tuples_var_id_set(list_var)
        # Loop over each unique element of list_var
        for var in set(list_var):
            # Get all tuples related to the element var
            list_var = list(filter(lambda x: x.var == var, new_list_varidset))
            # If there are more than one element with 
            if len(list_var)>1:
                sets_of_var = [x.set for x in list_var]
                sets_of_var = self.__reasign_set(sets_of_var)
                list_var = [VarIdSet(old_.var, old_.id, new_set_) for old_ ,new_set_ in zip(list_var, sets_of_var)]
                new_assign.extend(list_var)
            else:
            # If there exists only one element 
                new_assign.extend(list_var)
        return new_assign