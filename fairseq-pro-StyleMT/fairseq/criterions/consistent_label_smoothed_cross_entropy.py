# Coded by: Yuzhuang Xu
# Follows: https://github.com/protonish/fairseq-cipherdaug

import math
import logging
import torch

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


logger = logging.getLogger(__name__)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("consistent_label_smoothed_cross_entropy")
class ConsistLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        inter_dropout_alpha=0.0,
        intra_dropout_beta=0.0,
        kl_warmup=0,
    ):
        super().__init__(task, sentence_avg, label_smoothing)
        self.inter_dropout_alpha = torch.Tensor([inter_dropout_alpha]).to("cuda")
        self.intra_dropout_beta = torch.Tensor([intra_dropout_beta]).to("cuda")
        self.kl_warmup = kl_warmup
        logger.info("Alpha for KL Loss set to {} .".format(inter_dropout_alpha))
        logger.info("Beta for KL Loss set to {} .".format(intra_dropout_beta))
        logger.info("KL Loss will start after {} updates.".format(kl_warmup))

    @staticmethod
    def add_args(parser):
        # super().add_args(parser)
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

        parser.add_argument('--inter-dropout-alpha', default=0., type=float,
                            help='coefficient of loss in inter-dropout task')
        parser.add_argument('--intra-dropout-beta', default=0., type=float,
                            help='coefficient of loss in intra-dropout task')
        parser.add_argument('--kl-warmup', default=0, type=int,
                            help='warmUp model with regular x-ent for this many updates before computing KL loss')
        # fmt: on

    def compute_kl_loss(self, model, net_output, prime_net_output, pad_mask=None, reduce=True):
        # mean ouptut probs for the 2 forward passes
        # mean_net_output = (net_output[0] + prime_net_output[0]) / 2
        # mean_probs = model.get_normalized_probs((mean_net_output,), log_probs=False)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        prime_lprobs = model.get_normalized_probs(prime_net_output, log_probs=True)

        probs = model.get_normalized_probs(net_output, log_probs=False)
        prime_probs = model.get_normalized_probs(prime_net_output, log_probs=False)

        # p, q = torch.split(net_prob, net_prob.size(0) // 2, dim=0)
        # p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0) // 2, dim=0)

        # og
        # p_loss = torch.nn.functional.kl_div(lprobs, mean_probs, reduction="none")
        # q_loss = torch.nn.functional.kl_div(prime_lprobs, mean_probs, reduction="none")

        p_loss = torch.nn.functional.kl_div(lprobs, prime_probs, reduction="none")
        q_loss = torch.nn.functional.kl_div(prime_lprobs, probs, reduction="none")

        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.0)
            q_loss.masked_fill_(pad_mask, 0.0)

        if reduce:
            p_loss = p_loss.sum()
            q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    def forward(self, model, sample, reduce=True, inter_dropout=False, intra_dropout=False, num_updates=None):

        if num_updates is not None and num_updates < self.kl_warmup:
            net_output = model(**sample['net_input'])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
            logging_output = {
                'loss': loss.data,
                'nll_loss': nll_loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
            }
            return loss, sample_size, logging_output

        sample_input = sample["net_input"]

        # original outputs
        net_output = model(**sample_input)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        # inter outputs
        if inter_dropout is True:
            inter_net_output = model(**sample_input, inter_dropout=True)
            inter_lprobs = model.get_normalized_probs(inter_net_output, log_probs=True)
            inter_lprobs = inter_lprobs.view(-1, inter_lprobs.size(-1))

        # inter outputs
        if intra_dropout is True:
            intra_net_output = model(**sample_input, intra_dropout=True)
            intra_lprobs = model.get_normalized_probs(intra_net_output, log_probs=True)
            intra_lprobs = intra_lprobs.view(-1, intra_lprobs.size(-1))

        # # mean ouptut probs for the 2 forward passes
        # mean_net_output = (net_output[0] + prime_net_output[0]) / 2
        # mean_lprobs = model.get_normalized_probs(net_output, log_probs=False)

        target = model.get_targets(sample, net_output)
        pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
        # target = torch.cat([target, target.clone()], dim=0)

        # x-ent loss for original input
        og_loss, og_nll_loss = label_smoothed_nll_loss(
            lprobs,
            target.view(-1, 1),
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        # x-ent loss for inter input
        if inter_dropout is True:
            inter_loss, inter_nll_loss = label_smoothed_nll_loss(
                inter_lprobs,
                target.view(-1, 1),
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
            inter_kl_loss = self.compute_kl_loss(model, net_output, inter_net_output, pad_mask=pad_mask)
        else:
            inter_loss = inter_nll_loss = inter_kl_loss = torch.Tensor([0.0]).to("cuda")

        # x-ent loss for intra input
        if intra_dropout is True:
            intra_loss, intra_nll_loss = label_smoothed_nll_loss(
                intra_lprobs,
                target.view(-1, 1),
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
            intra_kl_loss = self.compute_kl_loss(model, net_output, intra_net_output, pad_mask=pad_mask)
        else:
            intra_loss = intra_nll_loss = intra_kl_loss = torch.Tensor([0.0]).to("cuda")

        # js_loss = torch.zeros(1).to(og_loss.device)
        loss = og_loss + self.inter_dropout_alpha * (inter_loss + inter_kl_loss) + self.intra_dropout_beta * (intra_loss + intra_kl_loss)

        ntokens = sample["ntokens"]
        nsentences = sample["target"].size(0)
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        # sample_size = sample_size * 2

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(og_nll_loss.data) if reduce else og_nll_loss.data,
            "inter_nll_loss": utils.item(inter_nll_loss.data) if reduce else inter_nll_loss.data,
            "inter_kl_loss": utils.item(inter_kl_loss.data) if reduce else inter_kl_loss.data,
            "intra_nll_loss": utils.item(intra_nll_loss.data) if reduce else intra_nll_loss.data,
            "intra_kl_loss": utils.item(intra_kl_loss.data) if reduce else intra_kl_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))

        # don't log for valid
        #if sample_size == 2 * ntokens:
        inter_kl_loss = utils.item(sum(log.get("inter_kl_loss", 0) for log in logging_outputs))
        metrics.log_scalar(
            "inter_kl_loss",
            inter_kl_loss / sample_size,
            sample_size,
            round=3,
        )
        inter_nll_loss = utils.item(sum(log.get("inter_nll_loss", 0) for log in logging_outputs))
        metrics.log_scalar(
            "inter_nll_loss",
            inter_nll_loss / ntokens / math.log(2),
            ntokens,
            round=3,
        )

        intra_kl_loss = utils.item(sum(log.get("intra_kl_loss", 0) for log in logging_outputs))
        metrics.log_scalar(
            "intra_kl_loss",
            intra_kl_loss / sample_size,
            sample_size,
            round=3,
        )
        intra_nll_loss = utils.item(sum(log.get("intra_nll_loss", 0) for log in logging_outputs))
        metrics.log_scalar(
            "intra_nll_loss",
            intra_nll_loss / ntokens / math.log(2),
            ntokens,
            round=3,
        )
