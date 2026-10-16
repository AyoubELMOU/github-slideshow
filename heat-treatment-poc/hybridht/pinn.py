"""Physics-informed neural network for 1D carburizing with a mass-transfer surface condition.

Dimensionless form (xi = x / Lc, tau = t / t_end, u = (c - c0) / (cp - c0)):
    u_tau = kappa u_xixi,                   kappa = D t_end / Lc^2
    -u_xi(0, tau) = Bi (1 - u(0, tau)),      Bi = beta Lc / D
    u(1, tau) = 0,  u(xi, 0) = 0  (hard-constrained via u = sqrt(tau) N(xi, sqrt(tau)))

Forward mode : D, beta known -> learn u, compare with analytical solution and FreeFEM.
Inverse mode : D, beta unknown -> identify them from sparse concentration "measurements".
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax


def init_mlp(key, sizes):
    params = []
    for m, n in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        params.append((jax.random.normal(k, (m, n)) * jnp.sqrt(2.0 / (m + n)), jnp.zeros(n)))
    return params


def mlp(params, z):
    for W, b in params[:-1]:
        z = jnp.tanh(z @ W + b)
    W, b = params[-1]
    return z @ W + b


# Reference kappa for the similarity feature eta = xi / (2 sqrt(kappa_ref tau)); a known
# scaling of diffusion problems that lets a small network resolve the thin early-time layer.
KAPPA_REF = 0.024


def u_fn(net, xi, tau):
    s = jnp.sqrt(tau)
    eta = xi / (2 * jnp.sqrt(KAPPA_REF) * s)
    z = jnp.stack([2 * jnp.tanh(eta / 2) - 1, 2 * xi - 1, 2 * s - 1], axis=-1)
    return s * mlp(net, z)[..., 0]


def _derivs(net, xi, tau):
    u = lambda a, b: u_fn(net, a, b)
    u_x = jax.grad(u, 0)
    return (jax.vmap(jax.grad(u, 1))(xi, tau),
            jax.vmap(jax.grad(u_x, 0))(xi, tau),
            jax.vmap(u_x)(xi, tau))


class Problem:
    def __init__(self, D, beta, t_end, Lc, c0, cp):
        self.D, self.beta, self.t_end, self.Lc, self.c0, self.cp = D, beta, t_end, Lc, c0, cp

    def groups(self, D, beta):
        return D * self.t_end / self.Lc**2, beta * self.Lc / D


def loss_terms(net, kappa, Bi, xi_c, tau_c, tau_b):
    u_t, u_xx, _ = _derivs(net, xi_c, tau_c)
    r_pde = u_t - kappa * u_xx
    z = jnp.zeros_like(tau_b)
    u0 = jax.vmap(lambda t: u_fn(net, 0.0, t))(tau_b)
    _, _, ux0 = _derivs(net, z, tau_b)
    r_rob = ux0 / Bi + (1.0 - u0)
    r_far = jax.vmap(lambda t: u_fn(net, 1.0, t))(tau_b)
    return jnp.mean(r_pde**2), jnp.mean(r_rob**2), jnp.mean(r_far**2)


def train(prob, mode="forward", data=None, steps=20000, seed=0, width=48, depth=4,
          n_col=2048, n_bnd=256, lr=2e-3, log_every=2000, D_guess=None, beta_guess=None):
    """Train the PINN. `data` (inverse mode) = (x [m], t [s], c [wt%]) arrays."""
    key = jax.random.PRNGKey(seed)
    net = init_mlp(key, [3] + [width] * depth + [1])
    theta = {"net": net}
    if mode == "inverse":
        # Optimise the O(1) dimensionless groups (float32-safe), report D and beta.
        k0, b0 = prob.groups(D_guess, beta_guess)
        theta["logkappa"] = jnp.asarray(np.log(k0), jnp.float32)
        theta["logBi"] = jnp.asarray(np.log(b0), jnp.float32)
        xd = jnp.asarray(data[0] / prob.Lc)
        td = jnp.asarray(data[1] / prob.t_end)
        ud = jnp.asarray((data[2] - prob.c0) / (prob.cp - prob.c0))

    def groups_of(th):
        if mode == "inverse":
            return jnp.exp(th["logkappa"]), jnp.exp(th["logBi"])
        return prob.groups(prob.D, prob.beta)

    def params_of(th):
        kappa, Bi = (float(v) for v in groups_of(th))
        D = kappa * prob.Lc**2 / prob.t_end
        return D, Bi * D / prob.Lc

    def loss(th, key):
        k1, k2, k3 = jax.random.split(key, 3)
        xi_c = jax.random.uniform(k1, (n_col,))
        tau_c = jax.random.uniform(k2, (n_col,), minval=1e-3, maxval=1.0)
        tau_b = jax.random.uniform(k3, (n_bnd,), minval=1e-3, maxval=1.0)
        kappa, Bi = groups_of(th)
        l_pde, l_rob, l_far = loss_terms(th["net"], kappa, Bi, xi_c, tau_c, tau_b)
        total = l_pde / kappa**2 * 1e-2 + 10.0 * l_rob + l_far
        l_dat = 0.0
        if mode == "inverse":
            l_dat = jnp.mean((u_fn(th["net"], xd, td) - ud) ** 2)
            total = total + 100.0 * l_dat
        return total, (l_pde, l_rob, l_far, l_dat)

    sched = optax.exponential_decay(lr, transition_steps=steps // 4, decay_rate=0.5)
    opt = optax.adam(sched)
    state = opt.init(theta)

    @jax.jit
    def step(th, state, key):
        (l, aux), g = jax.value_and_grad(loss, has_aux=True)(th, key)
        upd, state = opt.update(g, state, th)
        return optax.apply_updates(th, upd), state, l, aux

    history = []
    t0 = time.time()
    for i in range(steps):
        key, sk = jax.random.split(key)
        theta, state, l, aux = step(theta, state, sk)
        if i % log_every == 0 or i == steps - 1:
            D, beta = params_of(theta)
            history.append(dict(step=i, loss=float(l), pde=float(aux[0]), robin=float(aux[1]),
                                far=float(aux[2]), data=float(aux[3]), D=float(D), beta=float(beta),
                                wall_s=time.time() - t0))
    D, beta = params_of(theta)
    return dict(theta=theta, history=history, D=float(D), beta=float(beta),
                train_s=time.time() - t0)


def predict(result, prob, x, t):
    """Concentration [wt%] at depths x [m] and times t [s] (broadcastable)."""
    x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
    u = u_fn(result["theta"]["net"], jnp.asarray(x / prob.Lc), jnp.asarray(t / prob.t_end))
    return prob.c0 + (prob.cp - prob.c0) * np.asarray(u)
