from typing import Any, Iterator

import torch
import torch.distributed as dist
import torchdata.datapipes as dp


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


@dp.functional_datapipe("distributed_sharding_filter")
class DistributedShardingFilter(dp.iter.ShardingFilter):
    def __init__(self, source_datapipe: dp.iter.IterDataPipe) -> None:
        super().__init__(source_datapipe)

        self.rank = 0
        self.world_size = 1
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.apply_sharding(self.world_size, self.rank)

    def __iter__(self) -> Iterator[Any]:
        num_workers = self.world_size
        worker_id = self.rank
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_id + worker_info.id * num_workers
            num_workers *= worker_info.num_workers
        self.apply_sharding(num_workers, worker_id)

        yield from super().__iter__()
