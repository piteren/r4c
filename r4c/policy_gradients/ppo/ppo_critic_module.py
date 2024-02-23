import torch
from torchness.motorch import Module
from torchness.types import TNS, DTNS
from torchness.layers import LayDense, zeroes


class PPOCriticModule(Module):
    """ baseline PPO Critic Module """

    def __init__(
            self,
            observation_width: int,
            discount: float,
            n_hidden: int=      2,
            hidden_width: int=  12,
            lay_norm=           False,
            seed=               121,
            logger=             None,
            loglevel=           20,
    ):

        Module.__init__(self, logger=logger, loglevel=loglevel)

        lay_shapeL = []
        next_in = observation_width
        for _ in range(n_hidden):
            lay_shapeL.append((next_in,hidden_width))
            next_in = hidden_width

        self.lay_norm = lay_norm

        self.ln = torch.nn.LayerNorm(observation_width) # input layer norm

        self.linL = [LayDense(*shape) for shape in lay_shapeL]
        self.lnL = [torch.nn.LayerNorm(shape[-1]) for shape in lay_shapeL]

        lix = 0
        for lin,ln in zip(self.linL, self.lnL):
            self.add_module(f'lay_lin{lix}', lin)
            self.add_module(f'lay_ln{lix}', ln)
            lix += 1

        self.value = LayDense(
            in_features=    next_in,
            out_features=   1,
            activation=     None)

        self.discount = discount

    def forward(self, observation:TNS) -> DTNS:
        out = self.ln(observation) if self.lay_norm else observation
        zsL = []
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            zsL.append(zeroes(out))
            if self.lay_norm:
                out = ln(out)
        return {'value':self.value(out), 'zeroes':torch.cat(zsL).detach()}

    def loss(
            self,
            observation: TNS,
            dreturn: TNS,
    ) -> DTNS:
        out = self(observation)
        diff = dreturn - out['value']
        # TODO: add PPO Critic loss clipping (cleanrl ppo.py #268)
        loss = torch.mean(diff * diff) # MSE
        out.update({'loss': loss})
        return out