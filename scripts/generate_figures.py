#!/usr/bin/env python3
"""
generate_figures.py — Figure generation for the BRG paper.

Generates all 9 figures:
  Fig 1: Self-consistent fixed point convergence
  Fig 2: KL(α) landscape
  Fig 3: Perturbation response (with dynamic sentinel)
  Fig 4: Habituation dynamics (with dynamic sentinel)
  Fig 5: Free energy landscape FE(α, H)
  Fig 6: Dynamic sentinel response
  Fig 7: Sentinel parameter sensitivity
  Fig 8: Reservoir dimension sweep
  Fig 9: Phase transition analysis

Usage:
  python generate_figures.py          # generate all
  python generate_figures.py 1 3 6    # generate specific figures
"""

import os
import sys
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PAPER_DIR = os.path.join(ROOT_DIR, "paper")

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

COLORS = {
    "dynamic": "#9C27B0",
    "alpha1": "#2196F3",
    "alpha085": "#00BCD4",
    "alpha07": "#4CAF50",
    "alpha05": "#FF9800",
    "alpha0": "#F44336",
    "allc": "#9E9E9E",
    "standard": "#4CAF50",
}


def read_lines(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping")
        return []
    with open(path, "r") as f:
        return f.readlines()


def parse_selfconsistent(lines):
    """Parse the two-part CSV from selfconsistent experiment."""
    timeseries = []
    kl_landscape = []
    mode = "ts"

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("---KL_LANDSCAPE---"):
            mode = "kl"
            continue
        if line.startswith("experiment,") or line.startswith("alpha,seed,kl"):
            continue
        if mode == "ts":
            parts = line.split(",")
            if len(parts) >= 7:
                timeseries.append({
                    "experiment": parts[0],
                    "alpha": float(parts[1]),
                    "seed": int(parts[2]),
                    "t": int(parts[3]),
                    "body_output": float(parts[4]),
                    "action": float(parts[5]),
                    "state_norm": float(parts[6]),
                })
        elif mode == "kl":
            parts = line.split(",")
            if len(parts) >= 8:
                kl_landscape.append({
                    "alpha": float(parts[0]),
                    "seed": int(parts[1]),
                    "kl_noisy": float(parts[2]),
                    "kl_pure": float(parts[3]),
                    "mean_body_out": float(parts[4]),
                    "mean_action": float(parts[5]),
                    "mean_payoff": float(parts[6]),
                    "act_var": float(parts[7]),
                })

    return timeseries, kl_landscape


# ======================================================================
# Fig 1: Self-consistent fixed point convergence
# ======================================================================
def fig1_selfconsistent(timeseries):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    seed_0 = [r for r in timeseries if r["seed"] == 0]

    for exp, color, label in [
        ("coupled", None, None),
        ("allc", COLORS["allc"], "AllC"),
    ]:
        subset = [r for r in seed_0 if r["experiment"] == exp]
        if exp == "coupled":
            by_alpha = defaultdict(list)
            for r in subset:
                by_alpha[r["alpha"]].append(r)
            for alpha in sorted(by_alpha.keys()):
                rows = sorted(by_alpha[alpha], key=lambda x: x["t"])
                ts = [r["t"] for r in rows]
                bos = [r["body_output"] for r in rows]
                if alpha == 1.0:
                    c = COLORS["alpha1"]
                elif alpha == 0.5:
                    c = COLORS["alpha05"]
                elif alpha == 0.0:
                    c = COLORS["alpha0"]
                else:
                    c = "#000000"
                ax.plot(ts, bos, color=c, alpha=0.8,
                       label=f"$\\alpha={alpha:.1f}$ (coupled)")
        else:
            rows = sorted(subset, key=lambda x: x["t"])
            ts = [r["t"] for r in rows]
            bos = [r["body_output"] for r in rows]
            ax.plot(ts, bos, color=color, alpha=0.6, linestyle="--",
                   label=label)

    ax.axhline(y=0.95, color="black", linestyle=":", alpha=0.3, label="target")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("Body output $a^*(t)$")
    ax.set_title("(a) Body output convergence")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim([-0.05, 1.15])

    ax = axes[1]
    by_alpha_exp = defaultdict(lambda: defaultdict(list))
    for r in timeseries:
        if r["experiment"] == "coupled":
            by_alpha_exp[r["alpha"]][r["t"]].append(r["body_output"])

    for alpha in sorted(by_alpha_exp.keys()):
        data = by_alpha_exp[alpha]
        ts = sorted(data.keys())
        means = [np.mean(data[t]) for t in ts]
        stds = [np.std(data[t]) for t in ts]
        if alpha == 1.0:
            c = COLORS["alpha1"]
        elif alpha == 0.5:
            c = COLORS["alpha05"]
        else:
            c = COLORS["alpha0"]
        ax.plot(ts, means, color=c, label=f"$\\alpha={alpha:.1f}$")
        ax.fill_between(ts,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=c, alpha=0.15)

    ax.axhline(y=0.95, color="black", linestyle=":", alpha=0.3)
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("Body output $a^*(t)$")
    ax.set_title("(b) Mean $\\pm$ std across seeds")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim([-0.05, 1.15])

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, "fig_selfconsistent.pdf"))
    plt.close()
    print("  Generated: fig_selfconsistent.pdf")


# ======================================================================
# Fig 2: KL(α) landscape
# ======================================================================
def fig2_kl_landscape(kl_data):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    by_alpha_noisy = defaultdict(list)
    by_alpha_pure = defaultdict(list)
    by_alpha_var = defaultdict(list)
    by_alpha_act = defaultdict(list)
    by_alpha_pay = defaultdict(list)
    for r in kl_data:
        by_alpha_noisy[r["alpha"]].append(r["kl_noisy"])
        by_alpha_pure[r["alpha"]].append(r["kl_pure"])
        by_alpha_var[r["alpha"]].append(r["act_var"])
        by_alpha_act[r["alpha"]].append(r["mean_action"])
        by_alpha_pay[r["alpha"]].append(r["mean_payoff"])

    alphas = sorted(by_alpha_noisy.keys())

    ax = axes[0]
    means = [np.mean(by_alpha_noisy[a]) for a in alphas]
    sems = [np.std(by_alpha_noisy[a]) / np.sqrt(len(by_alpha_noisy[a]))
            for a in alphas]
    ax.errorbar(alphas, means, yerr=sems, fmt="o-", color="#2196F3",
               capsize=3, markersize=4, linewidth=1.5, label="Noisy ($\\varepsilon=0.1$)")

    means_pure = [np.mean(by_alpha_pure[a]) for a in alphas]
    sems_pure = [np.std(by_alpha_pure[a]) / np.sqrt(len(by_alpha_pure[a]))
                for a in alphas]
    ax.errorbar(alphas, means_pure, yerr=sems_pure, fmt="s--", color="#9E9E9E",
               capsize=3, markersize=3, linewidth=1, label="Pure cooperation")

    ax.set_xlabel("Metacognitive receptivity $\\alpha$")
    ax.set_ylabel("$D_{KL}(q_{\\alpha}^{\\mathrm{noisy}} \\| q^{\\mathrm{hab}})$")
    ax.set_title("(a) Complexity cost vs $\\alpha$")
    ax.set_xlim([-0.05, 1.05])
    ax.legend(fontsize=8)

    ax = axes[1]
    var_means = [np.mean(by_alpha_var[a]) for a in alphas]
    var_sems = [np.std(by_alpha_var[a]) / np.sqrt(len(by_alpha_var[a]))
                for a in alphas]
    ax.errorbar(alphas, var_means, yerr=var_sems, fmt="o-", color="#F44336",
               capsize=3, markersize=4, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Metacognitive receptivity $\\alpha$")
    ax.set_ylabel("Action variance $\\mathrm{Var}[a(t)]$")
    ax.set_title("(b) Reservoir smoothing effect")
    ax.set_xlim([-0.05, 1.05])

    ax = axes[2]
    pay_means = [np.mean(by_alpha_pay[a]) for a in alphas]
    pay_sems = [np.std(by_alpha_pay[a]) / np.sqrt(len(by_alpha_pay[a]))
                for a in alphas]
    act_means = [np.mean(by_alpha_act[a]) for a in alphas]

    ax.errorbar(alphas, pay_means, yerr=pay_sems, fmt="s-", color="#4CAF50",
               capsize=3, markersize=4, linewidth=1.5, label="Payoff")
    ax2 = ax.twinx()
    ax2.plot(alphas, act_means, "o-", color="#FF9800",
            markersize=3, linewidth=1, label="Mean action")
    ax.set_xlabel("Metacognitive receptivity $\\alpha$")
    ax.set_ylabel("Mean payoff $\\bar{u}$", color="#4CAF50")
    ax2.set_ylabel("Mean action $\\bar{a}$", color="#FF9800")
    ax.set_title("(c) Payoff and cooperation level")

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, "fig_kl_landscape.pdf"))
    plt.close()
    print("  Generated: fig_kl_landscape.pdf")


# ======================================================================
# Fig 3: Perturbation response (with dynamic sentinel)
# ======================================================================
def fig3_perturbation():
    lines = read_lines("perturbation.csv")
    if not lines:
        return

    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("agent,"):
            continue
        parts = line.split(",")
        if len(parts) >= 6:
            data.append({
                "agent": parts[0],
                "seed": int(parts[1]),
                "t": int(parts[2]),
                "body_output": float(parts[3]),
                "action": float(parts[4]),
                "opp_action": float(parts[5]),
            })

    if not data:
        print("  Warning: no perturbation data parsed")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    agents = ["alpha1", "dynamic", "alpha05", "alpha0", "allc"]
    labels = ["$\\alpha=1$ (body)", "Dynamic sentinel", "$\\alpha=0.5$ (mixed)",
              "$\\alpha=0$ (TfT)", "AllC"]
    colors = [COLORS["alpha1"], COLORS["dynamic"], COLORS["alpha05"],
              COLORS["alpha0"], COLORS["allc"]]

    for panel, field, ylabel, title in [
        (0, "action", "Agent action $a(t)$", "(a) Action response"),
        (1, "body_output", "Body output $a^*(t)$", "(b) Body readout response"),
    ]:
        ax = axes[panel]
        for agent, label, color in zip(agents, labels, colors):
            by_t = defaultdict(list)
            for r in data:
                if r["agent"] == agent:
                    by_t[r["t"]].append(r[field])
            if not by_t:
                continue
            ts = sorted(by_t.keys())
            means = [np.mean(by_t[t]) for t in ts]
            stds = [np.std(by_t[t]) for t in ts]
            lw = 2.0 if agent == "dynamic" else 1.5
            ax.plot(ts, means, color=color, label=label, linewidth=lw)
            ax.fill_between(ts,
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           color=color, alpha=0.1)

        ax.axvspan(200, 300, alpha=0.08, color="red")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)

    axes[0].set_ylim([-0.15, 1.15])
    axes[1].set_xlabel("Time step $t$")

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, "fig_perturbation.pdf"))
    plt.close()
    print("  Generated: fig_perturbation.pdf")


# ======================================================================
# Fig 4: Habituation dynamics (with dynamic sentinel)
# ======================================================================
def fig4_habituation():
    lines = read_lines("habituation_alpha.csv")
    if not lines:
        return

    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("mode,"):
            continue
        parts = line.split(",")
        if len(parts) >= 8:
            data.append({
                "mode": parts[0],
                "alpha": float(parts[1]),
                "seed": int(parts[2]),
                "H": int(parts[3]),
                "kl_noisy": float(parts[4]),
                "mean_body_out": float(parts[5]),
                "rho_current": float(parts[6]),
                "action_var": float(parts[7]),
            })

    if not data:
        print("  Warning: no habituation data parsed")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    groups = defaultdict(lambda: defaultdict(list))
    for r in data:
        key = (r["mode"], r["alpha"])
        groups[key][r["H"]].append(r)

    configs = [
        (("coupled", 1.0), COLORS["alpha1"], "$\\alpha=1$"),
        (("coupled", -1.0), COLORS["dynamic"], "Dynamic sentinel"),
        (("coupled", 0.5), COLORS["alpha05"], "$\\alpha=0.5$"),
        (("coupled", 0.0), COLORS["alpha0"], "$\\alpha=0$ (TfT)"),
    ]

    for panel, field, ylabel, title, use_log in [
        (0, "kl_noisy", "$D_{KL}$", "(a) Noise resilience", False),
        (1, "action_var", "Action variance", "(b) Smoothing effect", True),
        (2, "mean_body_out", "Mean body output", "(c) Body readout quality", False),
    ]:
        ax = axes[panel]
        for key, color, label in configs:
            if key not in groups:
                continue
            hs = sorted(groups[key].keys())
            means = [np.mean([r[field] for r in groups[key][h]]) for h in hs]
            stds = [np.std([r[field] for r in groups[key][h]]) for h in hs]
            sems = [s / max(np.sqrt(len(groups[key][h])), 1) for h, s in zip(hs, stds)]
            lw = 2.0 if "dynamic" in label.lower() or key[1] == -1.0 else 1.5
            ax.plot(hs, means, color=color, label=label, linewidth=lw)
            ax.fill_between(hs,
                           [max(1e-8, m - s) for m, s in zip(means, sems)],
                           [m + s for m, s in zip(means, sems)],
                           color=color, alpha=0.15)
        if use_log:
            ax.set_yscale("log")
        ax.set_xlabel("Habituation epochs $H$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, "fig_habituation.pdf"))
    plt.close()
    print("  Generated: fig_habituation.pdf")


# ======================================================================
# Fig 5: Free energy landscape
# ======================================================================
def fig5_free_energy():
    lines = read_lines("free_energy.csv")
    if not lines:
        return

    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("H,"):
            continue
        parts = line.split(",")
        if len(parts) >= 10:
            data.append({
                "H": int(parts[0]),
                "alpha": float(parts[1]),
                "seed": int(parts[2]),
                "kl": float(parts[3]),
                "mean_payoff": float(parts[4]),
                "mean_action": float(parts[5]),
                "mean_body_out": float(parts[6]),
                "fe_low": float(parts[7]),
                "fe_mid": float(parts[8]),
                "fe_high": float(parts[9]),
            })

    if not data:
        print("  Warning: no free energy data parsed")
        return

    h_values = sorted(set(r["H"] for r in data))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    cmap = plt.cm.viridis
    h_colors = {h: cmap(i / max(len(h_values) - 1, 1))
                for i, h in enumerate(h_values)}

    ax = axes[0]
    for h in h_values:
        by_alpha = defaultdict(list)
        for r in data:
            if r["H"] == h:
                by_alpha[r["alpha"]].append(r["kl"])
        alphas = sorted(by_alpha.keys())
        means = [np.mean(by_alpha[a]) for a in alphas]
        sems = [np.std(by_alpha[a]) / np.sqrt(len(by_alpha[a])) for a in alphas]
        ax.errorbar(alphas, means, yerr=sems, fmt="o-", color=h_colors[h],
                   capsize=2, markersize=3, label=f"$H={h}$", linewidth=1.2)

    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$D_{KL}(q_\\alpha \\| p_H)$")
    ax.set_title("(a) Complexity cost")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for h in h_values:
        by_alpha = defaultdict(list)
        for r in data:
            if r["H"] == h:
                by_alpha[r["alpha"]].append(r["fe_mid"])
        alphas = sorted(by_alpha.keys())
        means = [np.mean(by_alpha[a]) for a in alphas]
        sems = [np.std(by_alpha[a]) / np.sqrt(len(by_alpha[a])) for a in alphas]
        ax.errorbar(alphas, means, yerr=sems, fmt="o-", color=h_colors[h],
                   capsize=2, markersize=3, label=f"$H={h}$", linewidth=1.2)

    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$\\mathcal{F}(\\alpha)$")
    ax.set_title("(b) Free energy ($\\lambda=3$)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    for fe_col, lambda_val, color, marker in [
        ("fe_low", 1.0, "#4CAF50", "o"),
        ("fe_mid", 3.0, "#2196F3", "s"),
        ("fe_high", 8.0, "#F44336", "^"),
    ]:
        opt_alphas = []
        opt_h = []
        for h in h_values:
            by_alpha = defaultdict(list)
            for r in data:
                if r["H"] == h:
                    by_alpha[r["alpha"]].append(r[fe_col])
            alphas = sorted(by_alpha.keys())
            means = [np.mean(by_alpha[a]) for a in alphas]
            if means:
                best_idx = np.argmin(means)
                opt_alphas.append(alphas[best_idx])
                opt_h.append(h)
        ax.plot(opt_h, opt_alphas, f"{marker}-", color=color,
               label=f"$\\lambda={lambda_val}$", markersize=6, linewidth=1.5)

    ax.set_xlabel("Habituation epochs $H$")
    ax.set_ylabel("Optimal $\\alpha^*$")
    ax.set_title("(c) Optimal receptivity")
    ax.legend(fontsize=8)
    ax.set_ylim([-0.05, 1.15])

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, "fig_free_energy.pdf"))
    plt.close()
    print("  Generated: fig_free_energy.pdf")


# ======================================================================
# Fig 6: Dynamic sentinel response
# ======================================================================
def fig6_dynamic_sentinel():
    lines = read_lines("dynamic_alpha.csv")
    if not lines:
        return

    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("agent,") or line.startswith("---"):
            if line.startswith("---"):
                break
            continue
        parts = line.split(",")
        if len(parts) >= 13:
            data.append({
                "agent": parts[0],
                "seed": int(parts[1]),
                "t": int(parts[2]),
                "body_output": float(parts[3]),
                "a_cog": float(parts[4]),
                "alpha_t": float(parts[5]),
                "d_state": float(parts[6]),
                "d_output": float(parts[7]),
                "d_disagree": float(parts[8]),
                "d_total": float(parts[9]),
                "action": float(parts[10]),
                "opp_action": float(parts[11]),
                "payoff": float(parts[12]),
            })

    if not data:
        print("  Warning: no dynamic alpha data parsed")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    by_t = defaultdict(list)
    for r in data:
        if r["agent"] == "dynamic":
            by_t[r["t"]].append(r["alpha_t"])
    ts = sorted(by_t.keys())
    means = [np.mean(by_t[t]) for t in ts]
    stds = [np.std(by_t[t]) for t in ts]
    ax.plot(ts, means, color=COLORS["dynamic"], linewidth=1.5, label="$\\alpha(t)$ (dynamic)")
    ax.fill_between(ts,
                    [max(0, m - s) for m, s in zip(means, stds)],
                    [min(1, m + s) for m, s in zip(means, stds)],
                    color=COLORS["dynamic"], alpha=0.15)
    ax.axhline(y=0.85, color="grey", linestyle=":", alpha=0.5, label="$\\alpha_0=0.85$")

    ax.axvspan(500, 550, alpha=0.12, color="red", label="defection")
    ax.axvspan(1050, 1250, alpha=0.08, color="orange", label="noisy ($\\varepsilon=0.3$)")
    ax.set_ylabel("$\\alpha(t)$")
    ax.set_title("(a) Dynamic metacognitive receptivity")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.set_ylim([-0.05, 1.05])

    ax = axes[1]
    agents_plot = [
        ("dynamic", COLORS["dynamic"], "Dynamic sentinel"),
        ("alpha085", COLORS["alpha085"], "$\\alpha=0.85$"),
        ("alpha07", COLORS["alpha07"], "$\\alpha=0.7$"),
        ("alpha0", COLORS["alpha0"], "$\\alpha=0$ (TfT)"),
        ("alpha1", COLORS["alpha1"], "$\\alpha=1$ (body)"),
    ]

    for agent, color, label in agents_plot:
        by_t = defaultdict(list)
        for r in data:
            if r["agent"] == agent:
                by_t[r["t"]].append(r["action"])
        if not by_t:
            continue
        ts = sorted(by_t.keys())
        means = [np.mean(by_t[t]) for t in ts]
        ax.plot(ts, means, color=color, linewidth=1.2, label=label, alpha=0.8)

    ax.axvspan(500, 550, alpha=0.08, color="red")
    ax.axvspan(1050, 1250, alpha=0.05, color="orange")
    ax.set_ylabel("Agent action $a(t)$")
    ax.set_title("(b) Action response comparison")
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    ax.set_ylim([-0.1, 1.1])

    ax = axes[2]
    for agent, color, label in agents_plot:
        by_seed_t = defaultdict(lambda: defaultdict(float))
        for r in data:
            if r["agent"] == agent:
                by_seed_t[r["seed"]][r["t"]] = r["payoff"]

        if not by_seed_t:
            continue

        all_cum = []
        for seed_data in by_seed_t.values():
            ts = sorted(seed_data.keys())
            cum = np.cumsum([seed_data[t] for t in ts])
            all_cum.append(cum)

        if all_cum:
            cum_arr = np.array(all_cum)
            ts = sorted(list(by_seed_t.values())[0].keys())
            mean_cum = np.mean(cum_arr, axis=0)
            ax.plot(ts, mean_cum, color=color, linewidth=1.2, label=label, alpha=0.8)

    ax.axvspan(500, 550, alpha=0.08, color="red")
    ax.axvspan(1050, 1250, alpha=0.05, color="orange")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("Cumulative payoff")
    ax.set_title("(c) Cumulative payoff comparison")
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, "fig_dynamic_sentinel.pdf"))
    plt.close()
    print("  Generated: fig_dynamic_sentinel.pdf")


# ======================================================================
# Fig 7: Sentinel parameter sensitivity
# ======================================================================
def fig7_sentinel_sensitivity():
    lines = read_lines("dynamic_alpha.csv")
    if not lines:
        return

    data = []
    in_part2 = False
    for line in lines:
        line = line.strip()
        if line.startswith("---PARAM_SENSITIVITY---"):
            in_part2 = True
            continue
        if not in_part2:
            continue
        if line.startswith("---"):
            break
        if line.startswith("param,") or line.startswith("defect_length,"):
            continue
        parts = line.split(",")
        if len(parts) >= 7:
            data.append({
                "param": parts[0],
                "value": float(parts[1]),
                "seed": int(parts[2]),
                "mean_payoff": float(parts[3]),
                "mean_alpha": float(parts[4]),
                "mean_d_total": float(parts[5]),
                "action_var": float(parts[6]),
            })

    if not data:
        print("  Warning: no sensitivity data parsed")
        return

    params = ["alpha_0", "eta_up", "eta_down", "theta"]
    param_labels = {
        "alpha_0": "$\\alpha_0$ (baseline trust)",
        "eta_up": "$\\eta_{\\uparrow}$ (recovery rate)",
        "eta_down": "$\\eta_{\\downarrow}$ (intervention sharpness)",
        "theta": "$\\theta$ (threshold)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes_flat[idx]
        subset = [r for r in data if r["param"] == param]
        if not subset:
            continue

        by_val = defaultdict(lambda: {"payoff": [], "alpha": [], "var": []})
        for r in subset:
            by_val[r["value"]]["payoff"].append(r["mean_payoff"])
            by_val[r["value"]]["alpha"].append(r["mean_alpha"])
            by_val[r["value"]]["var"].append(r["action_var"])

        vals = sorted(by_val.keys())
        pay_means = [np.mean(by_val[v]["payoff"]) for v in vals]
        pay_sems = [np.std(by_val[v]["payoff"]) / np.sqrt(len(by_val[v]["payoff"])) for v in vals]
        alpha_means = [np.mean(by_val[v]["alpha"]) for v in vals]

        ax.errorbar(vals, pay_means, yerr=pay_sems, fmt="o-", color="#2196F3",
                   capsize=3, markersize=5, linewidth=1.5, label="Payoff")
        ax2 = ax.twinx()
        ax2.plot(vals, alpha_means, "s--", color="#FF9800",
                markersize=4, linewidth=1, label="Mean $\\alpha$")
        ax2.set_ylabel("Mean $\\alpha$", color="#FF9800", fontsize=9)
        ax2.set_ylim([0, 1])

        ax.set_xlabel(param_labels.get(param, param))
        ax.set_ylabel("Mean payoff", color="#2196F3", fontsize=9)
        ax.set_title(f"({chr(97+idx)}) {param}", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, "fig_sensitivity.pdf"))
    plt.close()
    print("  Generated: fig_sensitivity.pdf")


# ======================================================================
# Fig 8: Reservoir dimension sweep
# ======================================================================
def fig8_dimension_sweep():
    lines = read_lines("dimension_sweep.csv")
    if not lines:
        return

    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("regime,"):
            continue
        parts = line.split(",")
        if len(parts) >= 11:
            data.append({
                "regime": parts[0],
                "d": int(parts[1]),
                "alpha": float(parts[2]),
                "seed": int(parts[3]),
                "kl_noisy": float(parts[4]),
                "mean_payoff": float(parts[5]),
                "mean_action": float(parts[6]),
                "mean_body_out": float(parts[7]),
                "action_var": float(parts[8]),
                "convergence_time": int(parts[9]),
                "w_out_norm2": float(parts[10]),
            })

    if not data:
        print("  Warning: no dimension sweep data parsed")
        return

    scaled = [r for r in data if r["regime"] == "scaled"]
    fixed = [r for r in data if r["regime"] == "fixed"]
    dims = sorted(set(r["d"] for r in scaled))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    kl_at_1 = defaultdict(list)
    kl_at_0 = defaultdict(list)
    for r in scaled:
        if abs(r["alpha"] - 1.0) < 0.01:
            kl_at_1[r["d"]].append(r["kl_noisy"])
        if abs(r["alpha"] - 0.0) < 0.01:
            kl_at_0[r["d"]].append(r["kl_noisy"])

    dims_1 = sorted(kl_at_1.keys())
    means_1 = [np.mean(kl_at_1[d]) for d in dims_1]
    sems_1 = [np.std(kl_at_1[d]) / np.sqrt(len(kl_at_1[d])) for d in dims_1]
    ax.errorbar(dims_1, means_1, yerr=sems_1, fmt="o-", color="#2196F3",
               capsize=3, markersize=5, linewidth=1.5, label="$\\alpha=1$ (body)")

    dims_0 = sorted(kl_at_0.keys())
    means_0 = [np.mean(kl_at_0[d]) for d in dims_0]
    sems_0 = [np.std(kl_at_0[d]) / np.sqrt(len(kl_at_0[d])) for d in dims_0]
    ax.errorbar(dims_0, means_0, yerr=sems_0, fmt="s-", color="#F44336",
               capsize=3, markersize=5, linewidth=1.5, label="$\\alpha=0$ (TfT)")

    ax.set_xlabel("Reservoir dimension $d$")
    ax.set_ylabel("$D_{KL}(q^{\\mathrm{noisy}} \\| q^{\\mathrm{hab}})$")
    ax.set_title("(a) Complexity cost vs reservoir richness")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    var_at_0 = defaultdict(list)
    var_at_1 = defaultdict(list)
    for r in scaled:
        if abs(r["alpha"] - 0.0) < 0.01:
            var_at_0[r["d"]].append(r["action_var"])
        if abs(r["alpha"] - 1.0) < 0.01:
            var_at_1[r["d"]].append(r["action_var"])

    dims_both = sorted(set(var_at_0.keys()) & set(var_at_1.keys()))
    ratios = []
    for d in dims_both:
        v0 = np.mean(var_at_0[d])
        v1 = np.mean(var_at_1[d])
        if v1 > 1e-12:
            ratios.append(v0 / v1)
        else:
            ratios.append(v0 / 1e-12)

    ax.plot(dims_both, ratios, "o-", color="#4CAF50", markersize=5, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Reservoir dimension $d$")
    ax.set_ylabel("Variance reduction ratio")
    ax.set_title("(b) Smoothing power vs dimension")

    ax = axes[1, 0]
    lambda_fe = 3.0
    opt_alpha = {}
    for d in dims:
        by_alpha = defaultdict(lambda: {"payoff": [], "kl": []})
        for r in scaled:
            if r["d"] == d:
                by_alpha[r["alpha"]]["payoff"].append(r["mean_payoff"])
                by_alpha[r["alpha"]]["kl"].append(r["kl_noisy"])

        best_fe = float("inf")
        best_a = 0.5
        for a in sorted(by_alpha.keys()):
            fe = -np.mean(by_alpha[a]["payoff"]) + lambda_fe * np.mean(by_alpha[a]["kl"])
            if fe < best_fe:
                best_fe = fe
                best_a = a
        opt_alpha[d] = best_a

    dims_opt = sorted(opt_alpha.keys())
    ax.plot(dims_opt, [opt_alpha[d] for d in dims_opt], "s-", color="#9C27B0",
           markersize=6, linewidth=1.5)
    ax.set_xlabel("Reservoir dimension $d$")
    ax.set_ylabel("Optimal $\\alpha^*$")
    ax.set_title("(c) Optimal receptivity vs dimension")
    ax.set_ylim([-0.05, 1.15])

    ax = axes[1, 1]
    wnorm_scaled = defaultdict(list)
    wnorm_fixed = defaultdict(list)
    var_scaled = defaultdict(list)
    var_fixed = defaultdict(list)
    for r in scaled:
        if abs(r["alpha"] - 1.0) < 0.01:
            wnorm_scaled[r["d"]].append(r["w_out_norm2"])
            var_scaled[r["d"]].append(r["action_var"])
    for r in fixed:
        if abs(r["alpha"] - 1.0) < 0.01:
            wnorm_fixed[r["d"]].append(r["w_out_norm2"])
            var_fixed[r["d"]].append(r["action_var"])

    dims_s = sorted(wnorm_scaled.keys())
    dims_f = sorted(wnorm_fixed.keys())

    ax.errorbar(dims_s, [np.mean(wnorm_scaled[d]) for d in dims_s],
               yerr=[np.std(wnorm_scaled[d]) / np.sqrt(len(wnorm_scaled[d])) for d in dims_s],
               fmt="o-", color="#2196F3", capsize=3, markersize=5, linewidth=1.5,
               label="$\\|\\mathbf{w}\\|^2$ (scaled $\\lambda$)")
    ax.errorbar(dims_f, [np.mean(wnorm_fixed[d]) for d in dims_f],
               yerr=[np.std(wnorm_fixed[d]) / np.sqrt(len(wnorm_fixed[d])) for d in dims_f],
               fmt="s--", color="#F44336", capsize=3, markersize=5, linewidth=1.5,
               label="$\\|\\mathbf{w}\\|^2$ (fixed $\\lambda$)")

    ax2 = ax.twinx()
    ax2.errorbar(dims_s, [np.mean(var_scaled[d]) for d in dims_s],
                yerr=[np.std(var_scaled[d]) / np.sqrt(len(var_scaled[d])) for d in dims_s],
                fmt="^-", color="#4CAF50", capsize=3, markersize=4, linewidth=1,
                label="Var (scaled $\\lambda$)")
    ax2.errorbar(dims_f, [np.mean(var_fixed[d]) for d in dims_f],
                yerr=[np.std(var_fixed[d]) / np.sqrt(len(var_fixed[d])) for d in dims_f],
                fmt="v--", color="#FF9800", capsize=3, markersize=4, linewidth=1,
                label="Var (fixed $\\lambda$)")
    ax2.set_ylabel("Action variance ($\\alpha=1$)", fontsize=9)
    ax2.set_yscale("log")

    ax.set_xlabel("Reservoir dimension $d$")
    ax.set_ylabel("$\\|\\mathbf{w}_{\\mathrm{out}}\\|^2$")
    ax.set_title("(d) Ridge regularization decomposition")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, "fig_dimension_sweep.pdf"))
    plt.close()
    print("  Generated: fig_dimension_sweep.pdf")


# ======================================================================
# Fig 9: Phase transition analysis
# ======================================================================
def fig9_phase_transition():
    lines_da = read_lines("dynamic_alpha.csv")
    part3_data = []
    in_part3 = False
    for line in lines_da:
        line = line.strip()
        if line.startswith("---PHASE_TRANSITION_ENV---"):
            in_part3 = True
            continue
        if not in_part3:
            continue
        if line.startswith("defect_length,") or line.startswith("---"):
            if line.startswith("---"):
                break
            continue
        parts = line.split(",")
        if len(parts) >= 7:
            part3_data.append({
                "defect_length": int(parts[0]),
                "seed": int(parts[1]),
                "detection_time": int(parts[2]),
                "alpha_min_reached": float(parts[3]),
                "recovery_time": int(parts[4]),
                "sentinel_payoff": float(parts[5]),
                "tft_payoff": float(parts[6]),
            })

    lines_pt = read_lines("phase_transition.csv")
    pt_data = []
    for line in lines_pt:
        line = line.strip()
        if not line or line.startswith("d,"):
            continue
        parts = line.split(",")
        if len(parts) >= 8:
            pt_data.append({
                "d": int(parts[0]),
                "defect_length": int(parts[1]),
                "seed": int(parts[2]),
                "detection_time": int(parts[3]),
                "alpha_min_reached": float(parts[4]),
                "recovery_time": int(parts[5]),
                "sentinel_payoff": float(parts[6]),
                "tft_payoff": float(parts[7]),
            })

    if not pt_data and not part3_data:
        print("  Warning: no phase transition data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    lines_ds = read_lines("dimension_sweep.csv")
    ds_data = []
    for line in lines_ds:
        line = line.strip()
        if not line or line.startswith("regime,"):
            continue
        parts = line.split(",")
        if len(parts) >= 11 and parts[0] == "scaled":
            ds_data.append({
                "d": int(parts[1]),
                "alpha": float(parts[2]),
                "seed": int(parts[3]),
                "kl_noisy": float(parts[4]),
                "mean_payoff": float(parts[5]),
            })

    if ds_data:
        dims_ds = sorted(set(r["d"] for r in ds_data))
        lambda_fe = 3.0
        opt_alpha_ds = {}
        for d in dims_ds:
            by_alpha = defaultdict(lambda: {"payoff": [], "kl": []})
            for r in ds_data:
                if r["d"] == d:
                    by_alpha[r["alpha"]]["payoff"].append(r["mean_payoff"])
                    by_alpha[r["alpha"]]["kl"].append(r["kl_noisy"])
            best_fe = float("inf")
            best_a = 0.5
            for a in sorted(by_alpha.keys()):
                fe = -np.mean(by_alpha[a]["payoff"]) + lambda_fe * np.mean(by_alpha[a]["kl"])
                if fe < best_fe:
                    best_fe = fe
                    best_a = a
            opt_alpha_ds[d] = best_a

        dims_opt = sorted(opt_alpha_ds.keys())
        ax.plot(dims_opt, [opt_alpha_ds[d] for d in dims_opt], "s-", color="#9C27B0",
               markersize=6, linewidth=1.5)
        ax.axhline(y=0.5, color="grey", linestyle=":", alpha=0.5, label="$\\alpha^*=0.5$ (threshold)")
        d_c = None
        for d in dims_opt:
            if opt_alpha_ds[d] > 0.5:
                d_c = d
                break
        if d_c is not None:
            ax.axvline(x=d_c, color="red", linestyle="--", alpha=0.5, label=f"$d_c \\approx {d_c}$")
        ax.set_xlabel("Reservoir dimension $d$")
        ax.set_ylabel("Optimal $\\alpha^*$ (FE-minimizing)")
        ax.set_title("(a) Critical dimension $d_c$")
        ax.legend(fontsize=8)
        ax.set_ylim([-0.05, 1.15])

    ax = axes[0, 1]
    if part3_data:
        dl_set = sorted(set(r["defect_length"] for r in part3_data))
        alpha_min_by_dl = defaultdict(list)
        detect_by_dl = defaultdict(list)
        payoff_adv_by_dl = defaultdict(list)
        for r in part3_data:
            alpha_min_by_dl[r["defect_length"]].append(r["alpha_min_reached"])
            detect_by_dl[r["defect_length"]].append(r["detection_time"])
            payoff_adv_by_dl[r["defect_length"]].append(r["sentinel_payoff"] - r["tft_payoff"])

        means_am = [np.mean(alpha_min_by_dl[dl]) for dl in dl_set]
        sems_am = [np.std(alpha_min_by_dl[dl]) / np.sqrt(len(alpha_min_by_dl[dl])) for dl in dl_set]
        ax.errorbar(dl_set, means_am, yerr=sems_am, fmt="o-", color="#9C27B0",
                   capsize=3, markersize=5, linewidth=1.5, label="$\\alpha_{\\min}$")
        ax.set_xlabel("Defection block length $L$")
        ax.set_ylabel("Minimum $\\alpha$ reached")
        ax.set_title("(b) Sentinel collapse vs environment timescale")
        ax.legend(fontsize=8)

        ax2 = ax.twinx()
        means_pa = [np.mean(payoff_adv_by_dl[dl]) for dl in dl_set]
        sems_pa = [np.std(payoff_adv_by_dl[dl]) / np.sqrt(len(payoff_adv_by_dl[dl])) for dl in dl_set]
        ax2.errorbar(dl_set, means_pa, yerr=sems_pa, fmt="s--", color="#4CAF50",
                    capsize=3, markersize=4, linewidth=1, label="Payoff advantage")
        ax2.set_ylabel("Sentinel $-$ TfT payoff", color="#4CAF50", fontsize=9)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center right")

    ax = axes[1, 0]
    if part3_data:
        means_dt = [np.mean(detect_by_dl[dl]) for dl in dl_set]
        sems_dt = [np.std(detect_by_dl[dl]) / np.sqrt(len(detect_by_dl[dl])) for dl in dl_set]
        ax.errorbar(dl_set, means_dt, yerr=sems_dt, fmt="o-", color="#2196F3",
                   capsize=3, markersize=5, linewidth=1.5)
        ax.set_xlabel("Defection block length $L$")
        ax.set_ylabel("Detection time (steps)")
        ax.set_title("(c) Detection speed vs environment timescale")

    ax = axes[1, 1]
    if pt_data:
        dims_pt = sorted(set(r["d"] for r in pt_data))
        dls_pt = sorted(set(r["defect_length"] for r in pt_data))

        adv_grid = np.zeros((len(dims_pt), len(dls_pt)))
        for i, d in enumerate(dims_pt):
            for j, dl in enumerate(dls_pt):
                subset = [r for r in pt_data if r["d"] == d and r["defect_length"] == dl]
                if subset:
                    adv = np.mean([r["sentinel_payoff"] - r["tft_payoff"] for r in subset])
                    adv_grid[i, j] = adv

        im = ax.imshow(adv_grid, aspect="auto", origin="lower",
                       cmap="RdBu", interpolation="nearest",
                       extent=[0, len(dls_pt), 0, len(dims_pt)])
        ax.set_xticks(np.arange(len(dls_pt)) + 0.5)
        ax.set_xticklabels(dls_pt, fontsize=8)
        ax.set_yticks(np.arange(len(dims_pt)) + 0.5)
        ax.set_yticklabels(dims_pt, fontsize=8)
        ax.set_xlabel("Defection block length $\\tau_{\\mathrm{env}}$")
        ax.set_ylabel("Reservoir dimension $d$")
        ax.set_title("(d) Phase diagram: sentinel payoff advantage")
        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("Sentinel $-$ TfT payoff", fontsize=8)

        if adv_grid.shape[0] > 1 and adv_grid.shape[1] > 1:
            X, Y = np.meshgrid(np.arange(len(dls_pt)) + 0.5, np.arange(len(dims_pt)) + 0.5)
            try:
                ax.contour(X, Y, adv_grid, levels=[0], colors="black", linewidths=2, linestyles="--")
            except Exception:
                pass

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, "fig_phase_transition.pdf"))
    plt.close()
    print("  Generated: fig_phase_transition.pdf")


# ======================================================================
# Main
# ======================================================================
FIGURES = {
    1: ("Self-consistent fixed point", None),  # special handling
    2: ("KL landscape", None),                  # special handling
    3: ("Perturbation response", fig3_perturbation),
    4: ("Habituation dynamics", fig4_habituation),
    5: ("Free energy landscape", fig5_free_energy),
    6: ("Dynamic sentinel", fig6_dynamic_sentinel),
    7: ("Sentinel sensitivity", fig7_sentinel_sensitivity),
    8: ("Dimension sweep", fig8_dimension_sweep),
    9: ("Phase transition", fig9_phase_transition),
}


def main():
    print("BRG Paper — Figure Generation")
    print("=" * 50)

    # Determine which figures to generate
    if len(sys.argv) > 1:
        requested = [int(x) for x in sys.argv[1:]]
    else:
        requested = list(range(1, 10))

    # Fig 1 & 2 share the same data source
    if 1 in requested or 2 in requested:
        lines = read_lines("selfconsistent.csv")
        if lines:
            timeseries, kl_data = parse_selfconsistent(lines)
            if 1 in requested and timeseries:
                fig1_selfconsistent(timeseries)
            if 2 in requested and kl_data:
                fig2_kl_landscape(kl_data)

    # Fig 3-9
    for fig_num in requested:
        if fig_num in (1, 2):
            continue
        if fig_num in FIGURES and FIGURES[fig_num][1] is not None:
            FIGURES[fig_num][1]()

    print("=" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
