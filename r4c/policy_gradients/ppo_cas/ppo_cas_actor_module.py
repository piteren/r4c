import torch
from torchness.motorch import Module
from torchness.types import TNS, DTNS
from torchness.layers import LayDense, zeroes


class PPOCASActorModule(Module):
    """ baseline PPO for continuous action space (CAS) Actor Module """

    def __init__(
            self,
            observation_width: int,
            action_width: int,
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
            if ln: self.add_module(f'lay_ln{lix}', ln)
            lix += 1

        self.action_mean = LayDense(
            in_features=    next_in,
            out_features=   action_width,
            activation=     None)

        self.action_logstd = torch.nn.Parameter(torch.zeros(action_width))

        self.clip_coef = clip_coef
        self.entropy_coef = entropy_coef

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

        action_mean = self.action_mean(out)
        action_logstd = self.action_logstd.expand(action_mean.size())
        action_std = torch.exp(action_logstd)

        dist = torch.distributions.normal.Normal(action_mean, action_std)

        action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return {
            'dist':     dist,
            'action':   action,
            'logprob':  logprob,
            'entropy':  entropy,
            'zeroes':   torch.cat(zsL).detach()}

    def fwd_logprob(self, observation:TNS, action:TNS) -> DTNS:
        """ FWD
        + preparation of logprob of given action) """
        out = self(observation)
        out['logprob'] = out['dist'].log_prob(action).sum(-1)
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
                'approx_kl': ((ratio - 1) - logratio).mean(),
                'clipfracs': ((ratio - 1.0).abs() > self.clip_coef).float().mean()})

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
        #print('loss_actor:',loss_actor)
        loss_actor = torch.mean(loss_actor)
        #print('loss_actor mean:', loss_actor)

        entropy = torch.mean(out['entropy'])
        loss_entropy_factor = self.entropy_coef * entropy
        #print('loss_entropy_factor:', loss_entropy_factor)

        loss = loss_actor - loss_entropy_factor

        out.update({
            'entropy':      entropy,
            'loss':         loss,
            'loss_actor':   loss_actor})
        return out