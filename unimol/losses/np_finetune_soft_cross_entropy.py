# Soft-target cross-entropy for the organ-distribution task.
# Single classification head (num_classes = #organs), softmax over organs,
# trained against SOFT targets (a probability distribution per sample that
# sums to 1). At inference, softmax(logits) IS the predicted organ distribution.
#
# loss = -(target * log_softmax(logits)).sum(-1).mean()
#
# This file auto-registers via unimol/losses/__init__.py (it imports every
# non-underscore .py in this directory). Select it with:
#     --loss np_finetune_soft_cross_entropy

import math

import torch
import torch.nn.functional as F

from unimol.core import metrics
from unimol.core.losses import register_loss
from unimol.core.losses.cross_entropy import CrossEntropyLoss


def _pick(value, head_name):
    """The model output / target may be a bare tensor (single-head, no-schema
    path) or a dict keyed by task name (schema path). Return the tensor either
    way."""
    if isinstance(value, dict):
        if head_name in value:
            return value[head_name]
        # single-entry dict -> take the only value
        if len(value) == 1:
            return next(iter(value.values()))
        raise KeyError(
            f"soft_cross_entropy: head '{head_name}' not in {list(value.keys())}"
        )
    return value


@register_loss("np_finetune_soft_cross_entropy")
class NPFinetuneSoftCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, target_subclass="finetune_target",
                infer=False, output_cls_rep=False, **kwargs):
        head = self.args.classification_head_name
        net_output = model(
            sample,
            features_only=True,
            np_classification_head_name=head,
        )
        logits = _pick(net_output[0], head).float()
        logits = logits.view(-1, logits.size(-1))                  # [N, C]

        target = _pick(sample["target"][target_subclass], head).float()
        target = target.view(-1, logits.size(-1))                  # [N, C] distribution

        loss = self.compute_loss(logits, target, reduce=reduce)
        sample_size = target.size(0)

        if not self.training:
            # optionally surface the per-sample LNP CLS embeddings (--output-cls-rep)
            cls_rep_log_dict = {}
            if output_cls_rep:
                cls_representations = net_output[1]
                if isinstance(cls_representations, dict):
                    for cls_rep_name in cls_representations:
                        cls_rep_log_dict["cls_" + cls_rep_name] = cls_representations[cls_rep_name].cpu()
                else:
                    cls_rep_log_dict["cls_representations"] = cls_representations.cpu()

            logging_output = {
                "loss": loss.data,
                "lnp_ids": sample["net_input"].get("lnp_ids", []),
                "prob": F.softmax(logits, dim=-1).data.cpu(),
                "target": target.data.cpu(),
                "sample_size": sample_size,
                "bsz": sample_size,
                **cls_rep_log_dict,
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample_size,
            }
        return loss, sample_size, logging_output

    def compute_loss(self, logits, target, reduce=True):
        lprobs = F.log_softmax(logits, dim=-1)
        loss = -(target * lprobs).sum(dim=-1)                      # per-sample CE
        return loss.mean() if reduce else loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid", infer=False) -> dict:
        reduced_metrics_dict = {}

        # carry through LNP ids so predictions can be tied back to samples
        lnp_ids = []
        for log in logging_outputs:
            lnp_ids.extend(log.get("lnp_ids", []))
        reduced_metrics_dict["lnp_ids"] = lnp_ids

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        if sample_size:
            metrics.log_scalar(
                "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
            )

        if "valid" in split or "test" in split or infer:
            # compile any CLS representations emitted with --output-cls-rep
            for key in logging_outputs[0]:
                if "cls_" in key:
                    cls_rep_list = [log.get(key) for log in logging_outputs]
                    reduced_metrics_dict[key] = torch.cat(cls_rep_list, dim=0).cpu()

            # gather predicted distributions and soft targets
            probs = [log.get("prob") for log in logging_outputs if log.get("prob") is not None]
            tgts = [log.get("target") for log in logging_outputs if log.get("target") is not None]
            if probs:
                prob = torch.cat(probs, dim=0)
                reduced_metrics_dict["prob"] = prob.cpu()
            if tgts:
                tgt = torch.cat(tgts, dim=0)
                reduced_metrics_dict["target"] = tgt.cpu()

            # top-1 agreement: does argmax(prediction) match argmax(target)?
            if probs and tgts:
                correct = (prob.argmax(-1) == tgt.argmax(-1)).sum().item()
                total = prob.size(0)
                if total:
                    acc = correct / total
                    metrics.log_scalar(f"{split}_top1_acc", acc, total, round=4)
                    reduced_metrics_dict[f"{split}_top1_accuracy"] = acc

        return reduced_metrics_dict

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        return is_train
