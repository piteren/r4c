import torch
from torchness.types import TNS, DTNS
from torchness.motorch import Module
from torchness.layers import LayDense, zeroes


class DQNModule(Module):
    """ Deep QNetwork Module """

    def __init__(
            self,
            observation_width: int,
            num_actions: int,
            n_hidden: int=      2,
            hidden_width: int=  12,
            lay_norm=           False,
            use_huber: bool=    True, # MSE / Huber loss
            seed=               121,
            **kwargs):

        super().__init__(**kwargs)

        lay_shapeL = []
        next_in = observation_width
        for _ in range(n_hidden):
            lay_shapeL.append((next_in,hidden_width))
            next_in = hidden_width

        self.ln = torch.nn.LayerNorm(observation_width) if lay_norm else None # input layer norm

        self.linL = [LayDense(*shape) for shape in lay_shapeL]
        self.lnL = [torch.nn.LayerNorm(shape[-1]) if lay_norm else None for shape in lay_shapeL]

        lix = 0
        for lin,ln in zip(self.linL, self.lnL):
            self.add_module(f'lay_lin{lix}', lin)
            self.add_module(f'lay_ln{lix}', ln)
            lix += 1

        self.logits = LayDense(
            in_features=    next_in,
            out_features=   num_actions,
            activation=     None)

        loss_class = torch.nn.HuberLoss if use_huber else torch.nn.MSELoss
        self.loss_fn = loss_class(reduction='none')

    def forward(self, observation:TNS) -> DTNS:

        out = observation
        if self.ln:
            out = self.ln(observation)

        zsL = []
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            zsL.append(zeroes(out))
            if ln:
                out = ln(out)

        return {
            'qvs':      self.logits(out),
            'zeroes':   torch.cat(zsL).detach()}

    def loss(
            self,
            observation: TNS,
            action: TNS,
            new_qv: TNS,
    ) -> DTNS:
        out = self(observation)
        qv_pred = out['qvs'][range(len(action)),action]
        loss = self.loss_fn(qv_pred, new_qv)
        out['loss'] = torch.mean(loss)
        return out