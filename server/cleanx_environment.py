import pandas as pd
import numpy as np
import io
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CleanxAction, CleanxObservation
except (ModuleNotFoundError, ImportError):
    from models import CleanxAction, CleanxObservation

LATEST_CUSTOM_DATA = None
LATEST_CUSTOM_GOAL = None

class CleanxEnvironment(Environment):

    """
    CleanX AI Environment.
    An auto-grading dataset cleaner that supports 3 levels of difficulty.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.df = None
        self.original_df = None
        self.task_level = None
        self.goal = ""

    def reset(self) -> CleanxObservation:
        global LATEST_CUSTOM_DATA, LATEST_CUSTOM_GOAL
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        if LATEST_CUSTOM_DATA is not None:
            self.task_level = "custom"
            self.df = LATEST_CUSTOM_DATA.copy()
            self.goal = LATEST_CUSTOM_GOAL or "Custom Task: Clean this unknown dataset."
            LATEST_CUSTOM_DATA = None
            LATEST_CUSTOM_GOAL = None
        else:
            # We cycle through easy, medium, hard to ensure variety in baseline runs
            levels = ["easy", "medium", "hard"]
            self.task_level = levels[random.randint(0, 2)]
        
            if self.task_level == "easy":
                # Real-world: Simple contact list cleaning
                self.df = pd.DataFrame({
                    "Name": ["John Doe", "Jane Smith", None, "Alice Brown"],
                    "Email": ["john@example.com", "jane@example", "alice@base.com", "alice@base.com"],
                    "Phone": ["123-456", "987-654", "555-000", "123-456"]
                })
                self.goal = "Easy Task: 1. Rename 'Name' to 'full_name'. 2. Drop rows with missing names. 3. Remove duplicate rows based on all columns."
                
            elif self.task_level == "medium":
                # Real-world: E-commerce order logs
                self.df = pd.DataFrame({
                    "order_id": [1001, 1002, 1002, 1003],
                    "timestamp": ["2023-01-01 10:00", "2023-01-01 11:00", "2023-01-01 11:00", "01/02/2023"],
                    "amount": ["$10.50", "$20.00", "$20.00", "15.75"],
                    "is_shipped": ["yes", "no", "no", "yes"]
                })
                self.goal = "Medium Task: 1. Remove duplicate orders. 2. Rename 'timestamp' to 'order_date' and cast to datetime. 3. Clean 'amount' to float (remove '$'). 4. Cast 'is_shipped' to bool."
                
            else: # hard
                # Real-world: Messy sensor data / logs
                self.df = pd.DataFrame({
                    "SensorID": ["S1", "S2", "S3", "S1"],
                    "Value": ["10.5", "ERR", "12.2", "10.5"],
                    "Status": [1, 0, 1, 1],
                    "Entry": ["2023/05/01", "2023-05-01", "05-01-2023", "2023/05/01"]
                })
                self.goal = "Hard Task: 1. Drop duplicates. 2. Rename 'SensorID' to 'sensor_id', 'Value' to 'reading', 'Entry' to 'timestamp'. 3. Cast 'reading' to float (handle 'ERR' as NaN then drop). 4. Cast 'timestamp' to datetime. 5. Cast 'Status' to bool."

        self.original_df = self.df.copy()
        return self._make_observation(reward=0.0, done=False)

    def _make_observation(self, reward: float, done: bool, error_msg: str = None) -> CleanxObservation:
        preview = self.df.to_csv(index=False)
        return CleanxObservation(
            dataset_preview=preview,
            columns=list(self.df.columns),
            shape=list(self.df.shape),
            goal=self.goal,
            last_action_error=error_msg,
            progress=reward,
            done=done,
            reward=reward
        )
        
    def _evaluate(self) -> float:
        score = 0.0
        try:
            if self.task_level == "easy":
                # Goal: full_name exists, Name gone (0.3). dropna (0.4). drop_duplicates (0.3).
                if "full_name" in self.df.columns and "Name" not in self.df.columns: score += 0.3
                if self.df["full_name"].isna().sum() == 0: score += 0.4
                if len(self.df) == 3: score += 0.3 # Original 4, one NaN name dropped.

            elif self.task_level == "medium":
                # 1. duplicates (0.2). 2. rename/cast date (0.3). 3. amount float (0.3). 4. cast bool (0.2).
                if len(self.df) == 3: score += 0.2
                if "order_date" in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df.get("order_date")): score += 0.3
                if pd.api.types.is_float_dtype(self.df.get("amount")): score += 0.3
                if pd.api.types.is_bool_dtype(self.df.get("is_shipped")): score += 0.2

            elif self.task_level == "hard":
                # 1. dupes (0.1). 2. renames (0.2). 3. reading float/dropna (0.3). 4. timestamp (0.2). 5. status bool (0.2).
                if len(self.df) == 3: score += 0.1
                if all(c in self.df.columns for c in ["sensor_id", "reading", "timestamp"]): score += 0.2
                if pd.api.types.is_float_dtype(self.df.get("reading")) and self.df["reading"].isna().sum() == 0: score += 0.3
                if pd.api.types.is_datetime64_any_dtype(self.df.get("timestamp")): score += 0.2
                if pd.api.types.is_bool_dtype(self.df.get("Status")) or pd.api.types.is_bool_dtype(self.df.get("is_active")): # check both as agent might rename status to is_active
                    score += 0.2
        except Exception:
            pass
        return min(max(score, 0.0), 1.0)

    def step(self, action: CleanxAction) -> CleanxObservation:
        self._state.step_count += 1
        error_msg = None
        
        try:
            if action.operation == "drop_row":
                if action.args.get("dropna"):
                    self.df = self.df.dropna()
                elif action.args.get("drop_duplicates"):
                    self.df = self.df.drop_duplicates()
                    
            elif action.operation == "rename_column":
                old_name = action.args.get("old_name")
                new_name = action.args.get("new_name")
                if old_name in self.df.columns:
                    self.df = self.df.rename(columns={old_name: new_name})
                else:
                    error_msg = f"Column '{old_name}' not found."
                    
            elif action.operation == "cast_type":
                col = action.args.get("column")
                dtype = action.args.get("type", "").lower()
                if col in self.df.columns:
                    if dtype == "datetime":
                        self.df[col] = pd.to_datetime(self.df[col], format="mixed")
                    elif dtype == "float" or dtype == "float64":
                        if self.df[col].dtype == object and self.df[col].str.contains(r'[^\d.]', regex=True).any():
                           self.df[col] = self.df[col].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
                        else:
                           self.df[col] = self.df[col].astype(float)
                    elif dtype == "bool":
                        self.df[col] = self.df[col].astype(bool)
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                else:
                    error_msg = f"Column '{col}' not found."
                    
            elif action.operation == "submit":
                pass
                
        except Exception as e:
            error_msg = f"Action failed: {str(e)}"
            
        reward = self._evaluate()
        done = reward >= 1.0 or action.operation == "submit" or self._state.step_count >= 15
        
        return self._make_observation(reward, done, error_msg)

    @property
    def state(self) -> State:
        return self._state
