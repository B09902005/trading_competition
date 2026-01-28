from torch.utils.data import Sampler, BatchSampler, SequentialSampler, RandomSampler
from typing import Iterator, Sized
import torch

class PreBatchRandomSampler(Sampler[int]):
    r"""This sampler is designed to shuffle items within the original batch.
    The implementation is based on RandomSampler and BatchSampler.

    Args:
        data_source (Dataset): dataset to sample from
        batch_size (int): Size of mini-batch.
        generator (Generator): Generator used in sampling.
        maintain_order (bool): If True, only shuffle items within the original batch.

    Example:
        >>> sampler = PreBatchRandomSampler(range(10), batch_size=3)
        >>> list(BatchSampler(sampler, batch_size=3, drop_last=False))
        [[6, 8, 7], [0, 2, 1], [4, 3, 5], [9]]
        >>> list(BatchSampler(sampler, batch_size=3, drop_last=True))
        [[0, 2, 1], [8, 7, 6], [4, 5, 3]]
    """

    data_source: Sized
    batch_size: int

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        generator=None,
        maintain_order=False,
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.generator = generator
        self.maintain_order = maintain_order
        
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={self.batch_size}"
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )
    
    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        return len(self.data_source)
    
    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        iters = range(self.num_samples // self.batch_size)
        if self.maintain_order:
            iters = torch.randperm(self.num_samples // self.batch_size, generator=generator)
            

        for i in iters:
            result = torch.randperm(self.batch_size, generator=generator) + i * self.batch_size
            yield from result.tolist()
        result = torch.randperm(self.num_samples % self.batch_size, generator=generator) + (self.num_samples // self.batch_size) * self.batch_size
        yield from result.tolist()
    
    def __len__(self) -> int:
        return self.num_samples

if __name__ == '__main__':
    # no shuffle
    data = list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
    print(data) # [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    # default shuffle
    data = list(BatchSampler(RandomSampler(range(10)), batch_size=3, drop_last=True))
    print(data) # [[8, 4, 5], [2, 1, 3], [7, 0, 6]] -> Randomly shuffled, but the items within the original batch are mixed up.

    # batch_per_day shuffle
    sampler = PreBatchRandomSampler(range(10), batch_size=3)
    data = list(BatchSampler(sampler, batch_size=3, drop_last=True))
    print(data) # [[0, 2, 1], [8, 7, 6], [3, 4, 5]] -> Items within the original batch are maintained.

    # batch_per_day shuffle with maintain_order
    sampler = PreBatchRandomSampler(range(10), batch_size=3, maintain_order=True)
    data = list(BatchSampler(sampler, batch_size=3, drop_last=True))
    print(data) # [[0, 2, 1], [4, 3, 5], [6, 8, 7]] -> Batch order is maintained.
