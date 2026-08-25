"""The two presentation transforms, shared by every foundation model.

``difference``  integrated channels handed over as increments, re-integrated after
``stretch``     r-1 interpolated steps between real steps, so the same motion
                spans r times more of the model's patch grid

Both change the representation only; neither adds nor removes information.
"""

import numpy as np

FROM_CFG = "from_cfg"  # sentinel: take `difference` from the env config


def resolve_difference(spec, difference):
    """``FROM_CFG`` -> the env's integrated channels; ``()`` -> no differencing.

    Explicit channel indices need no ``spec`` at all, which is what lets a model
    run on data that is not one of this study's environments.
    """
    if isinstance(difference, str):
        if difference != FROM_CFG:
            raise ValueError(f"unknown sentinel {difference!r}")
        if spec is None:
            raise ValueError(
                "difference=FROM_CFG needs an env spec; pass the channel indices "
                "directly instead, e.g. difference=(1,)"
            )
        difference = spec["difference"]
    return tuple(difference or ())


def to_increments(ctx_s, difference):
    """Level context -> the array the model sees, with ``difference`` channels as
    increments. The first row has no increment and is dropped, which is why the
    caller also drops the first action."""
    m = ctx_s[:, 1:].copy()
    for c in difference:
        m[:, :, c] = np.diff(ctx_s[:, :, c], axis=1)
    return m


class UpsampledDynamics:
    """Run any model on a time-stretched copy of the same trajectory.

    States are interpolated linearly, actions held. That is a zero-order hold, not
    a repeated application: each sub-step increment shrinks by the same factor. The
    model forecasts r*H sub-steps and every r-th is kept.
    """

    def __init__(self, inner, factor=2):
        self.inner, self.factor = inner, int(factor)
        self.spec = inner.spec
        self.difference = getattr(inner, "difference", ())
        self.name = inner.name if factor == 1 else f"{inner.name} x{factor}"

    def predict(self, cs, ca, fa):
        r = self.factor
        if r == 1:
            return self.inner.predict(cs, ca, fa)
        cs, ca, fa = (np.asarray(x, np.float32) for x in (cs, ca, fa))
        L, H = cs.shape[1], fa.shape[1]
        dst = np.arange((L - 1) * r + 1)
        idx = np.minimum(dst // r, L - 2)
        frac = ((dst - idx * r) / r)[None, :, None]
        up_s = ((1 - frac) * cs[:, idx] + frac * cs[:, idx + 1]).astype(np.float32)
        up_a = ca[:, np.ceil(dst / r).astype(int)]  # ca[i] produced cs[i]
        out = self.inner.predict(up_s, up_a, np.repeat(fa, r, axis=1))
        return out[:, r - 1 :: r][:, :H]
