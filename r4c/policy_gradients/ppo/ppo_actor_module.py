import torch
from torchness.motorch import Module
from torchness.types import TNS, DTNS
from torchness.layers import LayDense, zeroes
from torchness.base_elements import select_with_indices


class PPOActorModule(Module):
    """ baseline PPO Actor Module """

    def __init__(
            self,
            observation_width: int,
            num_actions: int,
            minibatch_num: int,
            n_epochs_ppo: int,
            clip_coef: float,
            entropy_coef: float,
            n_hidden: int=      2,
            hidden_width: int=  12,
            lay_norm=           False,
            seed=               121,
            **kwargs):

        super().__init__(**kwargs)

        hidden_layers = [hidden_width] * n_hidden
        lay_shapeL = []
        next_in = observation_width
        for hl in hidden_layers:
            lay_shapeL.append((next_in,hl))
            next_in = hl

        self.lay_norm = lay_norm

        self.ln = torch.nn.LayerNorm(observation_width) if self.lay_norm else None # input layer norm

        self.linL = [LayDense(*shape) for shape in lay_shapeL]
        self.lnL = [torch.nn.LayerNorm(shape[-1]) if self.lay_norm else None for shape in lay_shapeL]

        lix = 0
        for lin,ln in zip(self.linL, self.lnL):
            self.add_module(f'lay_lin{lix}', lin)
            if ln: self.add_module(f'lay_ln{lix}', ln)
            lix += 1

        self.logits = LayDense(
            in_features=    next_in,
            out_features=   num_actions,
            activation=     None)

        self.clip_coef = clip_coef
        self.entropy_coef = entropy_coef

    def forward(self, observation:TNS) -> DTNS:

        out = observation
        if self.lay_norm:
            out = self.ln(observation)

        zsL = []
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            zsL.append(zeroes(out))
            if self.lay_norm:
                out = ln(out)

        logits = self.logits(out)
        dist = torch.distributions.Categorical(logits=logits)

        return {
            'logits':       logits,
            'probs':        dist.probs,
            'entropy':      dist.entropy().mean(),
            'zeroes':       zsL}

    def fwd_logprob(self, observation:TNS, action:TNS) -> DTNS:
        """ FWD
        + preparation of logprob (ln(prob) of selected move) """
        out = self(observation)
        prob_move = select_with_indices(source=out['probs'], indices=action)
        out['logprob'] = torch.log(prob_move)
        return out

    def fwd_logprob_ratio(self, observation:TNS, action:TNS, old_logprob:TNS) -> DTNS:
        """ FWD
        + logprob (current)
        + ratio of current logprob vs given old
        + prepares additional metrics """

        logrpob_out = self.fwd_logprob(observation=observation, action=action)
        new_logrpob = logrpob_out.pop('logprob')
        logratio = new_logrpob - old_logprob
        ratio = logratio.exp()

        out = logrpob_out
        out['ratio'] = ratio

        # stats
        with torch.no_grad():
            out.update({
                'approx_kl':    ((ratio - 1) - logratio).mean(),
                'clipfracs':    ((ratio - 1.0).abs() > self.clip_coef).float().mean()})

        return out

    def loss_actor(self, advantage:TNS, ratio:TNS) -> TNS:
        """ actor (policy) loss, clipped """
        pg_loss1 = -advantage * ratio
        pg_loss2 = -advantage * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        return torch.max(pg_loss1, pg_loss2)

    def loss(
            self,
            observation: TNS,
            action: TNS,
            advantage: TNS,
            logprob: TNS,
    ) -> DTNS:

        out = self.fwd_logprob_ratio(observation=observation, action=action, old_logprob=logprob)

        loss_actor = self.loss_actor(advantage=advantage, ratio=out['ratio'])
        loss_actor = torch.mean(loss_actor)

        entropy = out['entropy']
        loss_entropy_factor = self.entropy_coef * entropy

        loss = loss_actor - loss_entropy_factor

        out.update({
            'entropy':      entropy,
            'loss':         loss,
            'loss_actor':   loss_actor})
        return out