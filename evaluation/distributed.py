import numpy as np
import torch.distributed as dist


class DistributedForecastCollector:
    def __init__(self, args):
        self.args = args
        self.enabled = bool(getattr(args, "distributed", False))

    def collect(self, reals, sample_ensemble, contexts, sample_count):
        if not self.enabled:
            return reals, sample_ensemble, contexts

        gathered = [None for _ in range(int(self.args.world_size))]
        dist.all_gather_object(gathered, (reals, sample_ensemble, contexts))
        if not self.args.is_main_process:
            return None, None, None

        real_parts = []
        sample_parts = []
        context_parts = []
        for rank_reals, rank_samples, rank_contexts in gathered:
            real_parts.append(rank_reals)
            sample_parts.append(rank_samples)
            if rank_contexts is not None:
                context_parts.append(rank_contexts)

        merged_reals = np.concatenate(real_parts, axis=0)[:sample_count]
        merged_samples = np.concatenate(sample_parts, axis=1)[:, :sample_count]
        merged_contexts = (
            np.concatenate(context_parts, axis=0)[:sample_count]
            if len(context_parts) == int(self.args.world_size)
            else None
        )
        return merged_reals, merged_samples, merged_contexts
