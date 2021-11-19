from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

import pdb

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRTextAdaptModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        phn_vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        a2p: AbsEncoder,
        p2w: AbsEncoder,
        phn_ctc: CTC,
        ctc: CTC,
        phn_ctc_weight: float = 0.5,
        ignore_id: int = -1,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        p2w_use_posterior: bool = True,
    ):
        assert check_argument_types()

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.phn_vocab_size = phn_vocab_size
        self.ignore_id = ignore_id
        self.phn_ctc_weight = phn_ctc_weight
        self.token_list = token_list.copy()
        self.ctc = ctc
        self.phn_ctc = phn_ctc
        self.p2w_use_posterior = p2w_use_posterior

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.a2p = a2p
        self.p2w = p2w
        # we set self.decoder = None in the CTC mode since
        # self.decoder parameters were never used and PyTorch complained
        # and threw an Exception in the multi-GPU experiment.
        # thanks Jeff Farris for pointing out the issue.
        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor=None,
        speech_lengths: torch.Tensor=None,
        text_phn: torch.Tensor=None,
        text_phn_lengths: torch.Tensor=None,
        text: torch.Tensor=None,
        text_lengths: torch.Tensor=None,
        text_phn_in: torch.Tensor=None,
        text_phn_in_lengths: torch.Tensor=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if speech is not None:
            return self.forward_speech(speech, speech_lengths, text_phn, text_phn_lengths, text, text_lengths)
        else:
            return self.forward_text(text_phn_in, text_phn, text, text_lengths)

    def forward_text(
        self, 
        text_phn_in: torch.Tensor,
        text_phn_in_lengths,
        text,
        text_lengths):
        pass

    def forward_speech(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text_phn: torch.Tensor,
        text_phn_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # pre_encode
        feats, feats_lengths = self.pre_encode(speech, speech_lengths)

        # 1. a2p
        a2p_out, a2p_out_lens, _ = self.a2p(feats, feats_lengths)

        # pdb.set_trace()

        if self.phn_ctc_weight == 0.0:
            loss_phn_ctc, cer_phn_ctc = None, None
        else:
            loss_phn_ctc, cer_phn_ctc = self._calc_ctc_loss(
                a2p_out, a2p_out_lens, text_phn, text_phn_lengths, self.phn_ctc
            )

        # 2. p2w
        if self.p2w_use_posterior:
            a2p_out = self.phn_ctc.softmax(a2p_out)
            p2w_out, p2w_out_lens, _ = self.p2w(a2p_out, a2p_out_lens)
        else:
            p2w_out, p2w_out_lens, _ = self.p2w(a2p_out, a2p_out_lens)

        loss_ctc, cer_ctc = self._calc_ctc_loss(
            p2w_out, p2w_out_lens, text, text_lengths, self.ctc
        )

        if self.phn_ctc_weight == 0.0:
            loss = loss_ctc
        else:
            loss = self.phn_ctc_weight * loss_phn_ctc + (1 - self.phn_ctc_weight) * loss_ctc

        stats = dict(
            loss=loss.detach(),
            loss_phn_ctc=loss_phn_ctc.detach() if loss_phn_ctc is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            cer_ctc=cer_ctc,
            cer_phn_ctc=cer_phn_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text_phn: torch.Tensor,
        text_phn_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # pre_encode
        feats, feats_lengths = self.pre_encode(speech, speech_lengths)

        # 1. a2p
        a2p_out, a2p_out_lens, _ = self.a2p(feats, feats_lengths)

        # 2. p2w
        if self.p2w_use_posterior:
            a2p_out = self.phn_ctc.softmax(a2p_out)
            p2w_out, p2w_out_lens, _ = self.p2w(a2p_out, a2p_out_lens)
        else:
            p2w_out, p2w_out_lens, _ = self.p2w(a2p_out, a2p_out_lens)

        return p2w_out, p2w_out_lens

    def pre_encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        return feats, feats_lengths

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths


    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        ctc: CTC,
    ):
        # Calc CTC loss
        loss_ctc = ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        raise NotImplementedError
