from typing import Callable, Mapping, Optional, Sequence, Union

from pytorch_lightning.loggers import WandbLogger as _WandbLogger


class WandbLogger(_WandbLogger):
    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        version: Optional[str] = None,
        project: Optional[str] = None,
        log_model: Union[str, bool] = False,
        experiment=None,
        prefix: Optional[str] = "",
        agg_key_funcs: Optional[
            Mapping[str, Callable[[Sequence[float]], float]]
        ] = None,
        agg_default_func: Optional[
            Callable[[Sequence[float]], float]
        ] = None,
        entity: Optional[str] = None,
        job_type: Optional[str] = None,
        tags: Optional[Sequence] = None,
        group: Optional[str] = None,
        notes: Optional[str] = None,
        mode: Optional[str] = None,
        sync_tensorboard: Optional[bool] = False,
        monitor_gym: Optional[bool] = False,
        save_code: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            save_dir=save_dir,
            offline=offline,
            id=id,
            anonymous=anonymous,
            version=version,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            agg_key_funcs=agg_key_funcs,
            agg_default_func=agg_default_func,
            entity=entity,
            job_type=job_type,
            tags=tags,
            group=group,
            notes=notes,
            mode=mode,
            sync_tensorboard=sync_tensorboard,
            monitor_gym=monitor_gym,
            save_code=save_code,
            **kwargs,
        )


