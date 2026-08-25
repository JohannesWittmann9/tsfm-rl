"""The trained neural baseline: a 3-layer MLP on the one-step increment."""

import numpy as np
import torch
from torch import nn


class MLPDynamics:
    """Trained baseline (3-layer MLP, 128/64, Adam, early stopping).

    Input is ``[s_{t-lag+1..t}, a_{t-lag+1..t}, a_t]`` -- the last term matters:
    under this study's alignment (a_t produced s_t) the action that drives the
    predicted step would otherwise be missing. Target is the increment.

    ``lag`` is swept in the grid rather than pinned: these environments are Markov
    in the observation, so one step suffices in principle, and a longer history
    says what a flexible model buys from it. A smaller lag also lowers the smallest
    usable budget, since the number of training windows is T - lag.
    """

    def __init__(
        self,
        spec,
        states,
        actions,
        lag=1,
        hidden=(128, 64),
        lr=1e-3,
        batch_size=256,
        max_epochs=300,
        patience=20,
        val_frac=0.2,
        seed=0,
    ):
        self.spec, self.lag = spec, lag
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        states = np.asarray(states, np.float32)
        actions = np.asarray(actions, np.float32)
        x, y = self._windows(states, actions)
        rng = np.random.default_rng(seed)
        n_ep, n_win = x.shape[0], x.shape[1]
        if n_win == 0:
            raise ValueError(
                f"need more than lag={lag} transitions to build a "
                f"single training window"
            )
        if n_ep == 1:
            # A single (possibly truncated) episode has to be split along time.
            # Splitting by episode would hand the whole thing to training and
            # leave validation empty, silently disabling early stopping -- exactly
            # the small-sample case this has to get right.
            if n_win < 2:
                raise ValueError("too few windows to hold out a validation set")
            cut = min(n_win - 1, max(1, int((1 - val_frac) * n_win)))
            xt, yt, xv, yv = x[0, :cut], y[0, :cut], x[0, cut:], y[0, cut:]
        else:
            order = rng.permutation(n_ep)
            n_val = max(1, round(val_frac * n_ep))
            va, tr = order[:n_val], order[n_val:]

            def flat(a, e):
                return a[e].reshape(-1, a.shape[-1])

            xt, yt, xv, yv = flat(x, tr), flat(y, tr), flat(x, va), flat(y, va)
        self.xm, self.xs = self._stats(xt)
        self.ym, self.ys = self._stats(yt)
        torch.manual_seed(seed)
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(xt.shape[1], h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, spec.n_obs),
        ).to(self.device)
        self._train(xt, yt, xv, yv, lr, batch_size, max_epochs, patience)
        self.net.eval()
        self.name = f"MLP(lag {lag})"

    def _windows(self, states, actions):
        lag, n, T = self.lag, self.spec.n_obs, actions.shape[1]
        k = np.arange(lag, T)
        ws = states[:, k[:, None] + np.arange(-lag + 1, 1)]
        wa = actions[:, k[:, None] + np.arange(-lag, 0), 0]
        x = np.concatenate(
            [ws.reshape(*ws.shape[:2], lag * n), wa, actions[:, k, 0][..., None]], -1
        )
        return x.astype(np.float32), (states[:, k + 1] - states[:, k]).astype(
            np.float32
        )

    @staticmethod
    def _stats(a):
        sd = a.std(0)
        return a.mean(0), np.where(sd > 1e-8, sd, 1.0).astype(np.float32)

    def _train(self, xt, yt, xv, yv, lr, bs, max_epochs, patience):
        def to(a, m, s):
            return torch.as_tensor((a - m) / s, device=self.device)

        xt, yt = to(xt, self.xm, self.xs), to(yt, self.ym, self.ys)
        xv, yv = to(xv, self.xm, self.xs), to(yv, self.ym, self.ys)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        best, state, bad = float("inf"), None, 0
        g = torch.Generator().manual_seed(0)
        for _ in range(max_epochs):
            self.net.train()
            perm = torch.randperm(len(xt), generator=g).to(self.device)
            for i in range(0, len(perm), bs):
                b = perm[i : i + bs]
                opt.zero_grad()
                loss_fn(self.net(xt[b]), yt[b]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(loss_fn(self.net(xv), yv))
            if v < best - 1e-9:
                best, bad = v, 0
                state = {
                    k: t.detach().clone() for k, t in self.net.state_dict().items()
                }
            else:
                bad += 1
                if bad >= patience:
                    break
        if state:
            self.net.load_state_dict(state)

    def predict(self, cs, ca, fa):
        lag, n = self.lag, self.spec.n_obs
        s = np.asarray(cs, np.float32)[:, -lag:]
        a = np.asarray(ca, np.float32)[:, -lag:, 0]
        if s.shape[1] < lag:  # edge-pad a short context
            p = lag - s.shape[1]
            s = np.concatenate([np.repeat(s[:, :1], p, 1), s], 1)
            a = np.concatenate([np.repeat(a[:, :1], p, 1), a], 1)
        fut = np.asarray(fa, np.float32)[:, :, 0]
        out = np.empty((len(s), fut.shape[1], n), np.float32)
        with torch.no_grad():
            for h in range(fut.shape[1]):
                x = np.concatenate([s.reshape(len(s), -1), a, fut[:, h : h + 1]], 1)
                xn = torch.as_tensor((x - self.xm) / self.xs, device=self.device)
                d = self.net(xn).cpu().numpy() * self.ys + self.ym
                nxt = (s[:, -1] + d).astype(np.float32)
                out[:, h] = nxt
                s = np.concatenate([s[:, 1:], nxt[:, None]], 1)
                a = np.concatenate([a[:, 1:], fut[:, h : h + 1]], 1)
        return out
