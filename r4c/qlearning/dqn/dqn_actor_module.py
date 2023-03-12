import torch
from torchness.types import TNS, DTNS
from torchness.motorch import Module
from torchness.layers import LayDense
from typing import Optional



class DQNModel(Module):

    def __init__(
            self,
            num_actions: int=   4,
            observation_width=  4,
            hidden_layers=      (12,),
            seed=               121):

        torch.nn.Module.__init__(self)

        lay_shapeL = []
        next_in = observation_width
        for hl in hidden_layers:
            lay_shapeL.append((next_in,hl))
            next_in = hl

        self.ln = torch.nn.LayerNorm(observation_width) # input layer norm

        self.linL = [LayDense(*shape) for shape in lay_shapeL]
        self.lnL = [torch.nn.LayerNorm(shape[-1]) for shape in lay_shapeL]

        lix = 0
        for lin,ln in zip(self.linL, self.lnL):
            self.add_module(f'lay_lin{lix}', lin)
            self.add_module(f'lay_ln{lix}', ln)
            lix += 1

        self.logits = LayDense(
            in_features=    next_in,
            out_features=   num_actions,
            activation=     None)

        # TODO: try with Hubner
        self.loss_fn = torch.nn.MSELoss(reduction='none')


    def forward(self, observations:TNS) -> DTNS:
        out = self.ln(observations.to(torch.float32)) # + safety convert
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            out = ln(out)
        logits = self.logits(out)
        return {'logits': logits}


    def loss(
            self,
            observations: TNS,
            labels: TNS,
            mask: Optional[TNS]=    None
    ) -> DTNS:
        out = self(observations)
        loss = self.loss_fn(out['logits'], labels.to(torch.float32)) # + safety convert
        if mask is not None:
            loss *= mask                        # mask
        loss = torch.sum(loss, dim=-1)          # reduce over samples
        out['loss'] = torch.mean(loss)          # average
        return out