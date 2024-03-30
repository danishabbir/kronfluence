from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from kronfluence.task import Task
from kronfluence.abstract_task import AbstractTask

BATCH_DTYPE = Tuple[torch.Tensor, torch.Tensor]


class MnistExperimentModelOutput:
    # Copied from:
    # https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py#L87.
    softmax: nn.Module = torch.nn.Softmax(-1)
    loss_temperature: float = 1.0

    @staticmethod
    def get_output(
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.func.functional_call(
            model, (params, buffers), image.unsqueeze(0)
        )
        bindex = torch.arange(logits.shape[0]).to(
            device=logits.device, non_blocking=False
        )
        logits_correct = logits[bindex, label.unsqueeze(0)]

        cloned_logits = logits.clone()
        cloned_logits[bindex, label.unsqueeze(0)] = torch.tensor(
            -torch.inf, device=logits.device, dtype=logits.dtype
        )
        
        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum()

    def get_out_to_loss_grad(
        self,
        model: nn.Module,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        batch: BATCH_DTYPE,
    ) -> torch.Tensor:
        images, labels = batch
        logits = torch.func.functional_call(model, (params, buffers), images)
        ps = self.softmax(logits / self.loss_temperature)[
            torch.arange(logits.size(0)), labels
        ]
        return (1 - ps).clone().detach().unsqueeze(-1)


class MnistExperimentTask(AbstractTask):
    def __init__(
        self, device: torch.device = "cpu", generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__(device=device, generator=generator)

    def compute_train_loss(
        self,
        model: nn.Module,
        batch: BATCH_DTYPE,
        parameter_and_buffer_dicts: Optional[Union[Dict[str, torch.Tensor]]] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:
        images, labels = batch

        if parameter_and_buffer_dicts is None:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
        else:
            images = images.unsqueeze(0).to(self.device)
            labels = labels.unsqueeze(0).to(self.device)
            params, buffers = parameter_and_buffer_dicts
            outputs = torch.func.functional_call(model, (params, buffers), (images,))

        if not sample:
            return F.cross_entropy(outputs, labels.to(self.device), reduction=reduction)
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(outputs, dim=-1)
                sampled_labels = torch.multinomial(
                    probs, num_samples=1, generator=self.generator
                ).flatten()
            return F.cross_entropy(
                outputs, sampled_labels.detach(), reduction=reduction
            )

    def get_train_loss_with_wd(
        self,
        model: nn.Module,
        batch: Any,
        wd: float,
        parameter_and_buffer_dicts: Optional[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:
        assert not sample
        assert parameter_and_buffer_dicts is not None
        images, labels = batch

        images = images.unsqueeze(0).to(self.device)
        labels = labels.unsqueeze(0).to(self.device)
        params, buffers = parameter_and_buffer_dicts
        outputs = torch.func.functional_call(model, (params, buffers), (images,))
        sq_norm = 0
        for name, param in parameter_and_buffer_dicts[0].items():
            sq_norm += torch.sum(param**2.0)

        return (
            F.cross_entropy(outputs, labels.to(self.device), reduction=reduction)
            + 0.5 * wd * sq_norm
        )

    def compute_measurement(
        self,
        model: nn.Module,
        batch: BATCH_DTYPE,
        parameter_and_buffer_dicts: Optional[Union[Dict[str, torch.Tensor]]] = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        assert reduction == "sum"
        images, labels = batch

        if parameter_and_buffer_dicts is None:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = model(images)
        else:
            images = images.unsqueeze(0).to(self.device)
            labels = labels.unsqueeze(0).to(self.device)
            params, buffers = parameter_and_buffer_dicts
            logits = torch.func.functional_call(model, (params, buffers), (images,))

        bindex = torch.arange(logits.shape[0]).to(
            device=logits.device, non_blocking=False
        )
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(
            -torch.inf, device=logits.device, dtype=logits.dtype
        )

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()

    def get_batch_size(self, batch: BATCH_DTYPE) -> int:
        images, _ = batch
        return images.shape[0]

    def influence_modules(self) -> List[str]:
        return ["1", "3", "5", "7"]

    def representation_module(self) -> str:
        return "5"

    def get_model_output(self) -> Optional[Any]:
        return MnistExperimentModelOutput()


class MnistCorruptionExperimentTask(MnistExperimentTask):
    def get_measurement(
        self,
        model: nn.Module,
        batch: BATCH_DTYPE,
        parameter_and_buffer_dicts: Optional[Union[Dict[str, torch.Tensor]]] = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        assert reduction == "sum"
        return self.get_train_loss(
            model=model,
            batch=batch,
            parameter_and_buffer_dicts=parameter_and_buffer_dicts,
            sample=False,
            reduction=reduction,
        )
