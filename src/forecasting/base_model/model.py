from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod 
    def run(self, time_offset):
        ...