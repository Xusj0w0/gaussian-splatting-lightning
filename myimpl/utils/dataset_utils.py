from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict


class ExtraDataContainer(dict):
    pass


class ExtraDataProcessor(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __call__(self, extra_data: Dict[str, Any], *args, **kwargs):
        pass

    def update_properties(self, *args, **kwargs):
        pass


class ExtraDataProcessorContainer:
    def __init__(self):
        self._processors: Dict[str, ExtraDataProcessor] = {}

    def add_processor(self, key: str, processor: ExtraDataProcessor):
        self._processors.update({key: processor})

    def update_properties(self, *args, **kwargs):
        for v in self._processors.values():
            v.update_properties(*args, **kwargs)

    def __call__(self, extra_data: ExtraDataContainer):
        res = {}
        for k, v in extra_data.items():
            if k in self._processors:
                res.update({k: self._processors[k](v)})
            else:
                res.update({k: v})
        return res
