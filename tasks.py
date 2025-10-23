from textgrad.tasks.base import Dataset
import pandas as pd
from typing import Tuple, Callable
from textgrad.engine import EngineLM
from textgrad.variable import Variable
from typing import List, Union
import re
import textgrad as tg

def parse_tag_answer(answer: str, only_first_line: bool=False):
    try:
        if only_first_line:
            answer = answer.strip().split('\n')[0]
        answer = answer.strip()
        tags = re.findall(r'<[^<>]+>', answer)
        answer = tags[-1] if tags else ''
        answer = answer.lower().strip()
    except IndexError:
        answer = ''
    
    return answer

def tag_based_equality_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    pred_tag = parse_tag_answer(str(prediction.value))
    gt_tag = parse_tag_answer(str(ground_truth_answer.value))
    return int(pred_tag == gt_tag)


class TrafficSafeData(Dataset):
    def __init__(self, data_path, label_num=4):
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        self.label_num = label_num
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        label = row["label"]
        if self.label_num == 3 and (row["label"] in ['<NO APPARENT INJURY>', '<MINOR INJURY>']):
            label = '<NO APPARENT OR MINOR INJURY>'
        return "\n The crash data: "+row["prompt"], label

    def get_task_description(self):
        return self._task_description
    
class PandemicData(Dataset):
    def __init__(self, data_path, label_num=5):
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        self.label_num = label_num
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        label = row["t1"]
        if self.label_num == 3:
            label = row["t1_3"]
        return "\n The pandemic related information: "+row["prompt"], label

    def get_task_description(self):
        return self._task_description
    
class SwissData(Dataset):
    def __init__(self, data_path, prompt_col):
        self.prompt_col = prompt_col
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        return "\n The descriptions of the traveler: "+ row[self.prompt_col], row["answer"]

    def get_task_description(self):
        return self._task_description
    
    def get_causal_description(self):
        return self._causal_prompt
    

    

def load_task(task_name: str, evaluation_api: EngineLM, *args, **kwargs) -> Tuple[Dataset, Dataset, Callable]:
    if "trafficsafe" in task_name:
        from textgrad.autograd.string_based_ops import StringBasedFunction

        train_set = TrafficSafeData("./data/trafficsafe/train.csv", kwargs.get("label_num", 4))
        val_set = TrafficSafeData("./data/trafficsafe/val.csv", kwargs.get("label_num", 4))
        test_set = TrafficSafeData("./data/trafficsafe/test.csv", kwargs.get("label_num", 4))
        
        
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(tag_based_equality_fn, function_purpose=fn_purpose)
        
        return train_set, val_set, test_set, eval_fn
    
    if "swiss" in task_name:
        from textgrad.autograd.string_based_ops import StringBasedFunction
        
        train_set = SwissData("./data/swiss/train.csv", kwargs.get("prompt_col", "organized_prompt")) # random_prompt
        val_set = SwissData("./data/swiss/val.csv", kwargs.get("prompt_col", "organized_prompt"))
        test_set = SwissData("./data/swiss/test.csv", kwargs.get("prompt_col", "organized_prompt"))

        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(tag_based_equality_fn, function_purpose=fn_purpose)
        
        return train_set, val_set, test_set, eval_fn
    
    if "pandemic" in task_name:
        from textgrad.autograd.string_based_ops import StringBasedFunction
        
        train_set = PandemicData("./data/pandemic/train.csv", kwargs.get("label_num", 5))
        val_set = PandemicData("./data/pandemic/val.csv", kwargs.get("label_num", 5))
        test_set = PandemicData("./data/pandemic/test.csv", kwargs.get("label_num", 5))

        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(tag_based_equality_fn, function_purpose=fn_purpose)
        
        return train_set, val_set, test_set, eval_fn