import torch
from torchness.motorch import Module
from torchness.types import TNS, DTNS
from torchness.layers import LayDense, zeroes


# baseline AC Critic Module
class ACCriticModule(Module):

    def __init__(
            self,
            observation_width=  4,
            gamma=              0.99,  # discount factor (gamma)
            num_actions: int=   2,
            hidden_layers=      (24,24),
            lay_norm=           False,
            seed=               121,
            logger=             None,
            loglevel=           20,
    ):

        Module.__init__(self, logger=logger, loglevel=loglevel)

        self.gamma = gamma

        lay_shapeL = []
        next_in = observation_width
        for hl in hidden_layers:
            lay_shapeL.append((next_in,hl))
            next_in = hl

        self.lay_norm = lay_norm

        self.ln = torch.nn.LayerNorm(observation_width) # input layer norm

        self.linL = [LayDense(*shape) for shape in lay_shapeL]
        self.lnL = [torch.nn.LayerNorm(shape[-1]) for shape in lay_shapeL]

        lix = 0
        for lin,ln in zip(self.linL, self.lnL):
            self.add_module(f'lay_lin{lix}', lin)
            self.add_module(f'lay_ln{lix}', ln)
            lix += 1

        self.qvs = LayDense(
            in_features=    next_in,
            out_features=   num_actions,
            activation=     None)

    def forward(self, observations:TNS) -> DTNS:

        out = self.ln(observations) if self.lay_norm else observations

        zsL = []
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            zsL.append(zeroes(out))
            if self.lay_norm: out = ln(out)

        qvs = self.qvs(out)

        return {
            'qvs':      qvs,
            'zeroes':   zsL}

    def loss(
            self,
            observations: TNS,
            actions_taken: TNS,
            next_observations_qvs: TNS,
            next_actions_probs: TNS,
            rewards: TNS
    ) -> DTNS:

        out = self(observations)

        next_state_V = torch.sum(next_observations_qvs * next_actions_probs, dim=-1)
        target_QV = rewards + self.gamma * next_state_V
        qv = out['qvs'][range(len(actions_taken)),actions_taken]
        diff = target_QV - qv
        loss = torch.mean(diff * diff) # MSE

        out.update({'loss': loss})
        return out