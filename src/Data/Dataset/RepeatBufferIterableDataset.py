import torch
import random
import queue


class RepeatBufferIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, buffer_size, repeat_prob=.5,):
        self.dataset = dataset
        self.buffer = AutoDropQueue(maxsize=buffer_size)
        self.repeat_prob = repeat_prob

    def __iter__(self):
        for datum in self.dataset:
            while torch.rand(1) < self.repeat_prob and self.buffer.full():
                yield random.choice(self.buffer)

            self.buffer.put(datum)
            yield datum


# queue without constant time ops, but supports random.choice
class AutoDropQueue(list):
    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize

    def put(self, item,):
        while self.full():
            self.pop(0)
        return self.append(item)

    def full(self):
        return len(self) >= self.maxsize

