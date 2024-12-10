from typing import List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from ical.utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder

from ptflops import get_model_complexity_info
def test_encoder(model, img, img_mask):
    def encoder_input(resolution):
        return dict(img = img, img_mask = img_mask)
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(),
                                                 (1, 64, 64),
                                                 as_strings = True,
                                                 print_per_layer_stat = False,
                                                 input_constructor = encoder_input)
    print('{:<30} {:<8}'.format('Computational complexity: ', macs))
    print('{:<30} {:<8}'.format('Number of parameters: ', params))

def test_decoder(model, feature, mask, tgt):
    def decoder_input(resolution):
        return dict(src = feature, src_mask = mask, tgt = tgt)
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(),
                                                 (1, 64, 64),
                                                 as_strings = True,
                                                 print_per_layer_stat = False,
                                                 input_constructor = decoder_input)
    print('{:<30} {:<8}'.format('Computational complexity: ', macs))
    print('{:<30} {:<8}'.format('Number of parameters: ', params))
class ICAL(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        vocab_size: int = 114,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            vocab_size=vocab_size,
        )

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """

        # test_encoder(self.encoder, img, img_mask)
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
        mask = torch.cat((mask, mask), dim=0)

        # test_decoder(self.decoder, feature, mask, tgt)
        exp_out, imp_out, fusion_out = self.decoder(feature, mask, tgt)

        return exp_out, imp_out, fusion_out

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        return self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature
        )
