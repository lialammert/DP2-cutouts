# optim.py
# Anna Preto
# AstroDeep Project
# Optimisation of the deblending stages (2026-11-28): learning-rate schedule, Adam with global gradient clipping
# and an exponential moving average of the weights, and the callbacks that read them.
#
#   lr_schedule        linear warm-up from 1 % of the peak over the first 2 % of the updates, then cosine decay to
#                      1 % of the peak at the last update (keras CosineDecay); the peak is the stage's learning rate
#                      (x sqrt(B / 128) for the data-parallel stages, train.scale_for_batch)
#   DeblendAdam        Adam (beta_1 0.9, beta_2 0.999, epsilon 1e-8), clipping of the global gradient norm (all
#                      variables together; under keras.distribution.DataParallel the gradients are those of the
#                      whole batch, already averaged over the devices), running statistics of that norm before
#                      clipping, and an optional EMA of the trainable weights that copies them until ema_start (the
#                      end of the warm-up) and averages them afterwards
#   GradNormLogger     per epoch: mean, max and quantiles (50, 90, 95, 99 %) of the gradient norm before clipping,
#                      fraction of clipped updates, non-finite norms, learning rate (history.csv); the histograms of
#                      the whole run in <run>/grad_norms.json (scripts/inspect/clip_thresholds.py reads them)
#
# Clipping threshold (2026-11-28, as AutoClip, Seetharaman et al. 2020, and the Deep Learning Tuning Playbook): the
# summed-pixel likelihoods give gradient norms of ~1e6, so a threshold of 1 clipped every update (Adam on normalised
# gradients). The pilot runs without clipping (deblend.train.global_clipnorm: null) and records the norm of every
# update; each stage's threshold is then the 95 % quantile of its norms after the warm-up (C2: after lambda_D1 is
# fixed), deblend.<stage>.train.global_clipnorm. Gradients are those of the batch-mean loss (every loss term is a
# mean over the examples); under keras.distribution.DataParallel the global batch is one sharded program, so the
# norm is that of the whole batch's gradient, before any clipping (tests: one and two devices give the same norm).
#   EMAValidation      validation on the EMA weights (then restored); train.BestSubmodels saves them too
#   LambdaD1Protocol   C2: the gradient norms of its two objectives on the shared Rubin backbone, measured during
#                      training, set the target of lambda_D1 (models.C2Model)

import json
import time
from pathlib import Path

import keras
import numpy as np
from keras import ops


def lr_schedule(lr, total_steps, warmup_frac=0.02, final_frac=0.01, start_frac=0.01):
    # (schedule, warm-up steps). Keras CosineDecay with warm-up: linear from initial_learning_rate to warmup_target
    # over warmup_steps, then warmup_target x ((1 - alpha) (1 + cos(pi u)) / 2 + alpha), u from 0 to 1 over
    # decay_steps; constant at alpha x peak afterwards.
    total = max(int(total_steps), 1)
    warm = int(round(float(warmup_frac or 0) * total))
    if warm > 0:
        warm = max(warm, 1)
        sched = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=float(lr) * float(start_frac), decay_steps=max(total - warm, 1),
            alpha=float(final_frac), warmup_target=float(lr), warmup_steps=warm)
    else:
        sched = keras.optimizers.schedules.CosineDecay(initial_learning_rate=float(lr), decay_steps=total,
                                                       alpha=float(final_frac))
    return sched, warm


# Histogram of the gradient norms: 280 logarithmic bins from 1e-4 to 1e10 (12 % wide); below / above go to the end
# bins. Quantiles are read at the geometric centre of a bin.
NORM_EDGES = np.logspace(-4.0, 10.0, 281)


def hist_quantiles(counts, qs=(0.5, 0.9, 0.95, 0.99)):
    counts = np.asarray(counts, float)
    n = counts.sum()
    if n <= 0:
        return {f"q{int(round(q * 100))}": None for q in qs}
    cum = np.cumsum(counts) / n
    centres = np.sqrt(NORM_EDGES[:-1] * NORM_EDGES[1:])
    return {f"q{int(round(q * 100))}": float(centres[min(int(np.searchsorted(cum, q)), len(centres) - 1)])
            for q in qs}


class DeblendAdam(keras.optimizers.Adam):
    # stats_from: first update counted in the "steady" histogram (end of the warm-up; C2: end of its lambda_D1
    # protocol), from which the clipping threshold is read.
    def __init__(self, learning_rate, global_clipnorm=None, ema=None, ema_start=0, stats_from=0, **kw):
        kw.setdefault("beta_1", 0.9)
        kw.setdefault("beta_2", 0.999)
        kw.setdefault("epsilon", 1e-8)
        g = float(global_clipnorm) if global_clipnorm else None
        super().__init__(learning_rate=learning_rate, global_clipnorm=g, use_ema=bool(ema),
                         ema_momentum=float(ema) if ema else 0.99, **kw)
        self.clip_threshold = g or 0.0
        self.ema_start = int(ema_start)
        self.stats_from = int(stats_from)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.gn_sum = self.add_variable((), name="grad_norm_sum")
        self.gn_max = self.add_variable((), name="grad_norm_max")
        self.gn_count = self.add_variable((), name="grad_norm_count")
        self.gn_clipped = self.add_variable((), name="grad_norm_clipped")
        self.gn_nonfinite = self.add_variable((), name="grad_norm_nonfinite")
        nb = len(NORM_EDGES) - 1
        self.gn_hist_epoch = self.add_variable((nb,), name="grad_norm_hist_epoch")
        self.gn_hist_all = self.add_variable((nb,), name="grad_norm_hist_all")
        self.gn_hist_steady = self.add_variable((nb,), name="grad_norm_hist_steady")

    def _clip_gradients(self, grads):
        # Norm of the whole gradient before clipping, recorded at every update (sum, max, count, clipped count,
        # non-finite count, and three histograms: this epoch, the whole run, from stats_from on).
        sq = [ops.sum(ops.square(ops.cast(g, "float32"))) for g in grads if g is not None]
        n = ops.sqrt(sum(sq)) if sq else ops.convert_to_tensor(0.0)
        finite = ops.isfinite(n)
        fin = ops.cast(finite, "float32")
        nf = ops.where(finite, n, 0.0)
        self.gn_sum.assign(self.gn_sum + nf)
        self.gn_max.assign(ops.maximum(self.gn_max, nf))
        self.gn_count.assign(self.gn_count + fin)
        self.gn_nonfinite.assign(self.gn_nonfinite + (1.0 - fin))
        if self.clip_threshold > 0:
            self.gn_clipped.assign(self.gn_clipped + fin * ops.cast(nf > self.clip_threshold, "float32"))
        nb = len(NORM_EDGES) - 1
        lo, hi = float(np.log10(NORM_EDGES[0])), float(np.log10(NORM_EDGES[-1]))
        k = ops.floor((ops.log(ops.maximum(nf, 1e-30)) / np.log(10.0) - lo) / (hi - lo) * nb)
        onehot = ops.one_hot(ops.cast(ops.clip(k, 0, nb - 1), "int32"), nb) * fin
        self.gn_hist_epoch.assign(self.gn_hist_epoch + onehot)
        self.gn_hist_all.assign(self.gn_hist_all + onehot)
        steady = ops.cast(ops.greater_equal(self.iterations, self.stats_from), "float32")
        self.gn_hist_steady.assign(self.gn_hist_steady + onehot * steady)
        return super()._clip_gradients(grads)

    def _update_model_variables_moving_average(self, trainable_variables):
        if not self.use_ema:
            return
        started = ops.cast(ops.greater_equal(self.iterations, self.ema_start), "float32")
        for var, avg in zip(trainable_variables, self._model_variables_moving_average):
            if avg is not None:
                m = ops.cast(started * self.ema_momentum, var.dtype)
                avg.assign(m * avg + (1.0 - m) * var)

    def get_config(self):
        c = super().get_config()
        c.update(ema_start=self.ema_start)
        return c

    def stats(self):
        # This epoch (since reset_stats).
        n = float(ops.convert_to_numpy(self.gn_count))
        out = dict(grad_norm_mean=float(ops.convert_to_numpy(self.gn_sum)) / max(n, 1.0),
                   grad_norm_max=float(ops.convert_to_numpy(self.gn_max)),
                   clipped_frac=float(ops.convert_to_numpy(self.gn_clipped)) / max(n, 1.0), updates=n,
                   grad_norm_nonfinite=float(ops.convert_to_numpy(self.gn_nonfinite)))
        out.update({f"grad_norm_{k}": v for k, v in hist_quantiles(ops.convert_to_numpy(self.gn_hist_epoch)).items()})
        return out

    def run_histograms(self):
        h_all = np.asarray(ops.convert_to_numpy(self.gn_hist_all))
        h_st = np.asarray(ops.convert_to_numpy(self.gn_hist_steady))
        return dict(edges=NORM_EDGES.tolist(), threshold=self.clip_threshold or None, stats_from=self.stats_from,
                    all=dict(counts=h_all.tolist(), n=int(h_all.sum()), **hist_quantiles(h_all)),
                    steady=dict(counts=h_st.tolist(), n=int(h_st.sum()), **hist_quantiles(h_st)))

    def reset_stats(self):
        for v in (self.gn_sum, self.gn_max, self.gn_count, self.gn_clipped, self.gn_nonfinite):
            v.assign(0.0)
        self.gn_hist_epoch.assign(ops.zeros(self.gn_hist_epoch.shape))


class EMAWeights:
    # Context: the trainable variables tracked by the optimizer take their EMA values (copies: the JAX train and
    # test steps donate their input buffers), restored on exit. No-op without EMA or before the first update.
    def __init__(self, optimizer):
        self.opt = optimizer
        self.saved = None

    def active(self):
        o = self.opt
        return o is not None and getattr(o, "use_ema", False) and o.built and int(ops.convert_to_numpy(o.iterations)) > 0

    def __enter__(self):
        if self.active():
            pairs = list(zip(self.opt._trainable_variables, self.opt._model_variables_moving_average))
            self.saved = [(v, ops.copy(v.value)) for v, a in pairs if a is not None]
            for v, a in pairs:
                if a is not None:
                    v.assign(ops.copy(a.value))
        return self

    def __exit__(self, *a):
        if self.saved is not None:
            for v, val in self.saved:
                v.assign(val)
            self.saved = None


def _sync(model):
    # JAX: write the training state of the running fit back into the variables (keras purges them between steps).
    if hasattr(model, "jax_state_sync"):
        model.jax_state_sync()


class GradNormLogger(keras.callbacks.Callback):
    # Adds the gradient-norm statistics of the epoch and lr to the epoch logs (before CSVLogger in the list), and
    # writes the run's histograms to <run>/grad_norms.json at every epoch end.
    def __init__(self, run_dir=None):
        super().__init__()
        self.path = None if run_dir is None else Path(run_dir) / "grad_norms.json"

    def on_epoch_begin(self, epoch, logs=None):
        opt = self.model.optimizer
        if isinstance(opt, DeblendAdam) and opt.built:
            opt.reset_stats()

    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer
        if logs is None or not isinstance(opt, DeblendAdam) or not opt.built:
            return
        s = opt.stats()
        logs.update({k: (np.nan if v is None else v) for k, v in s.items() if k != "updates"})
        lr = opt.learning_rate
        logs["lr"] = float(ops.convert_to_numpy(lr(opt.iterations) if callable(lr) else lr))
        if self.path is not None:
            h = opt.run_histograms()
            h.update(epoch=int(epoch) + 1, updates=int(ops.convert_to_numpy(opt.iterations)),
                     time=time.strftime("%Y-%m-%d %H:%M:%S"))
            self.path.write_text(json.dumps(h))


class EMAValidation(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.ctx = None

    def on_test_begin(self, logs=None):
        self.ctx = EMAWeights(self.model.optimizer).__enter__()

    def on_test_end(self, logs=None):
        if self.ctx is not None:
            self.ctx.__exit__()
            self.ctx = None


def branch_grad_norms(model, x):
    # Unweighted gradients of C2's D2 objective (E_r[-log p_E] + KL(r || s)) and of its D1 term
    # (1/2 KL(q_A || s) + 1/2 KL(q_A || r)) with respect to the shared Rubin backbone: norms and cosine. JAX.
    import jax
    tv = [v.value for v in model.trainable_variables]
    ntv = [v.value for v in model.non_trainable_variables]
    shared = {id(v) for v in model.branches.r_embed.trainable_variables}
    sel = [k for k, v in enumerate(model.trainable_variables) if id(v) in shared]
    if not hasattr(model, "_grad_norm_fn"):
        def fn(tv_, ntv_, x_):
            def term(p, which):
                out, _ = model.stateless_call(p, ntv_, x_, training=True)
                t = model.loss_terms(x_, out, training=False)
                return t["d2_nll_e"] + t["d2_kl_rs"] if which == "d2" else 0.5 * t["d1_kl_s"] + 0.5 * t["d1_kl_r"]
            g2 = jax.grad(lambda p: term(p, "d2"))(tv_)
            g1 = jax.grad(lambda p: term(p, "d1"))(tv_)
            n2 = sum(jax.numpy.sum(g2[k] ** 2) for k in sel)
            n1 = sum(jax.numpy.sum(g1[k] ** 2) for k in sel)
            dot = sum(jax.numpy.sum(g2[k] * g1[k]) for k in sel)
            return n2, n1, dot
        model._grad_norm_fn = jax.jit(fn)
    n2, n1, dot = model._grad_norm_fn(tv, ntv, {k: jax.numpy.asarray(v) for k, v in x.items()})
    n2, n1, dot = float(np.sqrt(n2)), float(np.sqrt(n1)), float(dot)
    return dict(d2_grad=n2, d1_grad=n1, cosine=dot / max(n2 * n1, 1e-30))


class LambdaD1Protocol(keras.callbacks.Callback):
    # models.LAMBDA_PROTOCOL in training updates: after `warm` updates at lambda_D1 = 1, every `every` updates
    # during the next `measure`, the gradient norms on `batches` training batches (unweighted, before clipping) give
    # the ratio ||g_D2|| / (||g_D1|| + eps); at the end of the window the median ratio becomes models.C2Model
    # .lambda_target (lambda_D1 then ramps to it over `ramp` updates and stays there). Without a finite ratio the
    # target stays 1. Everything is written to <run>/lambda_d1.json. A run resumed after the window keeps the
    # target restored from its backup.
    def __init__(self, ds, run_dir, protocol, eps=1e-12):
        super().__init__()
        self.ds, self.path, self.p, self.eps = ds, Path(run_dir) / "lambda_d1.json", dict(protocol), float(eps)
        self.records, self.step, self.done = [], 0, False

    def on_train_begin(self, logs=None):
        self.step = int(round(float(ops.convert_to_numpy(self.model.n_updates))))
        self.done = self.step >= self.p["warm"] + self.p["measure"]

    def _measure(self, s):
        _sync(self.model)
        for j in range(int(self.p.get("batches", 2))):
            i = (s * 7919 + j * 104729) % len(self.ds)
            x, _ = self.ds[i]
            r = branch_grad_norms(self.model, x)
            r.update(step=s, batch=int(i), ratio=r["d2_grad"] / (r["d1_grad"] + self.eps))
            self.records.append(r)
            print(f"lambda_d1 protocol, update {s}: ||g_D2|| {r['d2_grad']:.4g}, ||g_D1|| {r['d1_grad']:.4g}, "
                  f"ratio {r['ratio']:.4g}, cosine {r['cosine']:.3f}", flush=True)

    def on_train_batch_end(self, batch, logs=None):
        if self.done:
            return
        self.step += 1
        s, w, m = self.step, int(self.p["warm"]), int(self.p["measure"])
        if w < s <= w + m and (s - w) % max(int(self.p.get("every", 10)), 1) == 0:
            self._measure(s)
        if s >= w + m:
            ratios = [r["ratio"] for r in self.records if np.isfinite(r["ratio"]) and r["ratio"] > 0]
            target = float(np.median(ratios)) if ratios else 1.0
            _sync(self.model)
            self.model.lambda_target.assign(target)
            self.done = True
            self.path.write_text(json.dumps(dict(
                target=target, fallback=not ratios, protocol=self.p, n_measurements=len(self.records),
                median_cosine=float(np.median([r["cosine"] for r in self.records])) if self.records else None,
                records=self.records, time=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
            print(f"lambda_d1 protocol: target {target:.4g} from {len(ratios)} ratios"
                  + (" (none finite: 1)" if not ratios else "") + f"; ramp over {self.p['ramp']} updates", flush=True)
