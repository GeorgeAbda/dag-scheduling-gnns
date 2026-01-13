from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tyro

from cogito.dataset_generator.core.models import Dataset
from cogito.dataset_generator.core.gen_vm import generate_hosts, generate_vms, allocate_vms
from cogito.dataset_generator.core.gen_dataset import (
    generate_dataset,
    generate_dataset_long_cp_queue_free,
    generate_dataset_wide_queue_free,
    classify_queue_regime,
)


@dataclass
class Args:
    # Single-config mode (legacy): select k representatives from one domain
    config_json: str | None = None
    """Training config JSON with train/eval dataset blocks and (optionally) seeds."""
    out_json: str = "representative_eval.json"
    """Output JSON with selected eval seeds and coverage stats (single-config mode)."""
    k: int = 10
    """Number of representative eval environments to select (single-config mode)."""
    save_plot_dir: str | None = "runs/representativeness"
    """Optional directory to save simple feature scatter plots and CSV."""

    # Dual-config mode: build a fixed dataset JSON with workflows from two domains (no FGW)
    wide_config: str | None = None
    longcp_config: str | None = None
    k_wide: int = 5
    k_long: int = 5
    out_dataset_json: str | None = None
    # Optional overrides for synthesized compute fabric
    host_count: int | None = None
    vm_count: int | None = None
    max_memory_gb: int | None = None
    min_cpu_speed: int | None = None
    max_cpu_speed: int | None = None
    # Queue regime controls for single-config mode
    target_regime: str = "any"
    regime_req_divisor: int = 20
    regime_tol: float = 0.05


def _gen_ds_for_seed(seed: int, cfg: Dict[str, Any]) -> Dataset:
    ds = cfg.get("dataset", {})
    style = str(ds.get("style", "generic"))
    p = ds.get("gnp_p", None)

    common_kwargs = dict(
        seed=int(seed),
        host_count=int(ds.get("host_count", 4)),
        vm_count=int(ds.get("vm_count", 10)),
        max_memory_gb=int(ds.get("max_memory_gb", 10)),
        min_cpu_speed_mips=int(ds.get("min_cpu_speed", 500)),
        max_cpu_speed_mips=int(ds.get("max_cpu_speed", 5000)),
        workflow_count=int(ds.get("workflow_count", 1)),
        task_length_dist=str(ds.get("task_length_dist", "normal")),
        min_task_length=int(ds.get("min_task_length", 500)),
        max_task_length=int(ds.get("max_task_length", 100_000)),
        task_arrival=str(ds.get("task_arrival", "static")),
        arrival_rate=float(ds.get("arrival_rate", 3.0)),
    )
    dag_method = str(ds.get("dag_method", "gnp"))
    gnp_kwargs = dict(
        gnp_min_n=int(ds.get("gnp_min_n", 10)),
        gnp_max_n=int(ds.get("gnp_max_n", 40)),
    )

    if style == "long_cp":
        pr = (float(p), float(p)) if p is not None else (0.70, 0.95)
        return generate_dataset_long_cp_queue_free(p_range=pr, **common_kwargs, **gnp_kwargs)
    if style == "wide":
        pr = (float(p), float(p)) if p is not None else (0.02, 0.20)
        return generate_dataset_wide_queue_free(p_range=pr, **common_kwargs, **gnp_kwargs)
    # generic fallback
    return generate_dataset(
        gnp_p=(None if p is None else float(p)),
        dag_method=dag_method,
        **common_kwargs,
        **gnp_kwargs,
    )


def _layers(children: Dict[int, List[int]]) -> List[List[int]]:
    indeg: Dict[int, int] = {u: 0 for u in children}
    for u, vs in children.items():
        for v in vs:
            indeg[v] = indeg.get(v, 0) + 1
    frontier = [u for u in children if indeg[u] == 0]
    layers: List[List[int]] = []
    seen: set[int] = set()
    while frontier:
        cur = list(frontier)
        layers.append(cur)
        frontier = []
        for u in cur:
            if u in seen:
                continue
            seen.add(u)
            for v in children.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    frontier.append(v)
    return layers if layers else [list(children.keys())]


def _critical_path_len(children: Dict[int, List[int]], lengths: Dict[int, int]) -> float:
    # parents map
    parents: Dict[int, List[int]] = {u: [] for u in children}
    for u, vs in children.items():
        for v in vs:
            parents.setdefault(v, []).append(u)
    # topo order using layers order
    layers = _layers(children)
    order = [u for L in layers for u in L]
    dp: Dict[int, float] = {u: float(lengths.get(u, 1)) for u in children}
    for u in order:
        best = 0.0
        for p in parents.get(u, []):
            best = max(best, dp.get(p, 0.0))
        dp[u] = float(lengths.get(u, 1)) + best
    return float(max(dp.values()) if dp else 0.0)


def _extract_features(ds: Dataset) -> Tuple[np.ndarray, List[str]]:
    # single workflow expected (our generator uses 1), but support multiple
    feats: List[float] = []
    # aggregate over workflows
    Ws: List[int] = []
    Ds: List[int] = []
    Wtot = 0.0
    Csum = 0.0
    Nsum = 0
    Esum = 0
    widths: List[int] = []
    for wf in ds.workflows:
        children: Dict[int, List[int]] = {t.id: list(t.child_ids) for t in wf.tasks}
        n = len(children)
        e = sum(len(vs) for vs in children.values())
        layers = _layers(children)
        depth = len(layers)
        width_peak = max((len(L) for L in layers), default=0)
        lengths = {t.id: int(t.length) for t in wf.tasks}
        C = _critical_path_len(children, lengths)
        W = float(sum(lengths.values()))
        Ws.append(width_peak)
        Ds.append(depth)
        Wtot += W
        Csum += C
        Nsum += n
        Esum += e
        widths.extend(len(L) for L in layers)
    width_avg = (np.mean(widths) if widths else 0.0) if widths else (Nsum / max(1, np.mean(Ds)))
    width_peak_all = int(max(Ws) if Ws else 0)
    depth_avg = float(np.mean(Ds) if Ds else 0.0)
    Pbar = (Wtot / Csum) if Csum > 0 else 0.0
    burst = (float(width_peak_all) / float(width_avg)) if width_avg > 0 else 0.0
    cp_frac = (Csum / Wtot) if Wtot > 0 else 0.0
    feats = [
        float(Nsum),                # tasks
        float(Esum),                # edges
        float(width_peak_all),      # peak width
        float(depth_avg),           # avg depth
        float(Pbar),                # avg parallelism
        float(burst),               # burstiness
        float(cp_frac),             # CP share of work
    ]
    names = [
        "tasks",
        "edges",
        "width_peak",
        "depth_avg",
        "Pbar",
        "burstiness",
        "cp_frac",
    ]
    return np.array(feats, dtype=float), names


def _standardize(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = F.mean(axis=0)
    sigma = F.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (F - mu) / sigma, mu, sigma


def _select_k_representatives(Fz: np.ndarray, k: int) -> List[int]:
    # k-center greedy (coverage); start from point nearest to mean, then farthest-first
    n = Fz.shape[0]
    mean = Fz.mean(axis=0)
    start = int(np.argmin(np.sum((Fz - mean) ** 2, axis=1)))
    centers = [start]
    dmin = np.sum((Fz - Fz[start]) ** 2, axis=1)
    for _ in range(1, min(k, n)):
        i = int(np.argmax(dmin))
        centers.append(i)
        dmin = np.minimum(dmin, np.sum((Fz - Fz[i]) ** 2, axis=1))
    return centers


def _select_k_representatives_restricted(Fz: np.ndarray, candidates: List[int], k: int) -> List[int]:
    n = Fz.shape[0]
    cand = sorted({int(i) for i in candidates if 0 <= int(i) < n})
    if not cand:
        return _select_k_representatives(Fz, k)
    k = int(min(max(1, k), len(cand)))
    mean = Fz.mean(axis=0)
    start = int(min(cand, key=lambda i: float(np.sum((Fz[i] - mean) ** 2))))
    centers = [start]
    dmin = np.sum((Fz - Fz[start]) ** 2, axis=1)
    for _ in range(1, k):
        best_i = start
        best_val = -1.0
        for i in cand:
            val = float(np.min(np.vstack([dmin, np.sum((Fz - Fz[i]) ** 2, axis=1)]), axis=0).mean())
            if val > best_val and i not in centers:
                best_val = val
                best_i = i
        centers.append(int(best_i))
        dmin = np.minimum(dmin, np.sum((Fz - Fz[best_i]) ** 2, axis=1))
    return centers


def main(a: Args) -> None:
    # If dual-config requested and out_dataset_json provided, build fixed dataset JSON
    if a.wide_config and a.longcp_config and a.out_dataset_json:
        cfg_w = json.loads(Path(a.wide_config).read_text())
        cfg_l = json.loads(Path(a.longcp_config).read_text())

        def _seeds_and_cfg(cfg: dict) -> tuple[list[int], dict]:
            tr = cfg.get("train", {})
            seeds = list(tr.get("seeds", []))
            if not seeds:
                rng = np.random.RandomState(12345)
                seeds = [int(rng.randint(1, 10_000_000)) for _ in range(100)]
            return [int(s) for s in seeds], tr

        seeds_w, tr_w = _seeds_and_cfg(cfg_w)
        seeds_l, tr_l = _seeds_and_cfg(cfg_l)

        # Compute features per seed and select k via k-center in feature space
        def _select_k(seeds: list[int], tr_cfg: dict, k: int) -> list[int]:
            feats: list[np.ndarray] = []
            for s in seeds:
                ds = _gen_ds_for_seed(int(s), tr_cfg)
                f, _ = _extract_features(ds)
                feats.append(f)
            F = np.stack(feats, axis=0)
            Fz, _, _ = _standardize(F)
            idxs = _select_k_representatives(Fz, int(k))
            return [int(seeds[i]) for i in idxs]

        sel_w = _select_k(seeds_w, tr_w, int(a.k_wide))
        sel_l = _select_k(seeds_l, tr_l, int(a.k_long))
        print(f"[representative_eval] Selected WIDE seeds: {sel_w}")
        print(f"[representative_eval] Selected LONGCP seeds: {sel_l}")

        # Reconstruct workflows for selected seeds (each dataset has workflow_count=1)
        workflows = []
        wf_id = 0
        workflow_domains: list[str] = []
        for s in sel_w:
            ds = _gen_ds_for_seed(int(s), tr_w)
            if ds.workflows:
                wf = ds.workflows[0]
                wf.id = wf_id; wf.arrival_time = 0
                # ensure task.workflow_id and contiguous task ids
                for i, t in enumerate(wf.tasks):
                    t.id = i; t.workflow_id = wf_id
                workflows.append(wf)
                workflow_domains.append("wide")
                wf_id += 1
        for s in sel_l:
            ds = _gen_ds_for_seed(int(s), tr_l)
            if ds.workflows:
                wf = ds.workflows[0]
                wf.id = wf_id; wf.arrival_time = 0
                for i, t in enumerate(wf.tasks):
                    t.id = i; t.workflow_id = wf_id
                workflows.append(wf)
                workflow_domains.append("longcp")
                wf_id += 1

        # Synthesize hosts and VMs using host_specs and simple VM generator
        # Infer defaults from both configs if overrides not provided
        def _infer_int(keys: list[str], default: int) -> int:
            def _get(ds: dict, k: str, d: int) -> int:
                return int(ds.get(k, d))
            ds_w = tr_w.get("dataset", {})
            ds_l = tr_l.get("dataset", {})
            vals = []
            for k in keys:
                vals.append(_get(ds_w, k, default))
                vals.append(_get(ds_l, k, default))
            return int(max(vals) if vals else default)

        H = int(a.host_count) if a.host_count is not None else _infer_int(["host_count"], 4)
        V = int(a.vm_count) if a.vm_count is not None else _infer_int(["vm_count"], 12)
        max_mem_gb = int(a.max_memory_gb) if a.max_memory_gb is not None else _infer_int(["max_memory_gb"], 128)
        min_mips = int(a.min_cpu_speed) if a.min_cpu_speed is not None else _infer_int(["min_cpu_speed"], 500)
        max_mips = int(a.max_cpu_speed) if a.max_cpu_speed is not None else _infer_int(["max_cpu_speed"], 5000)

        rng = np.random.RandomState(12345)
        hosts = generate_hosts(H, rng)
        vms = generate_vms(V, max_mem_gb, min_mips, max_mips, rng)
        allocate_vms(vms, hosts, rng)

        ds_out = Dataset(workflows=workflows, vms=vms, hosts=hosts)
        outp = Path(a.out_dataset_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        base = ds_out.to_json()
        # Attach metadata with selected seeds and per-workflow domains
        base["meta"] = {
            "selected": {
                "wide_seeds": sel_w,
                "longcp_seeds": sel_l,
            },
            "workflow_domains": workflow_domains,
        }
        outp.write_text(json.dumps(base, indent=2))
        print(f"[representative_eval] Wrote fixed dataset JSON with {len(workflows)} workflows to: {outp}")
        return

    # Legacy single-config selection mode: select k seeds and write stats JSON
    cfg = json.loads(Path(a.config_json).read_text())
    tr_cfg = cfg.get("train", {})
    tr_seeds: List[int] = list(tr_cfg.get("seeds", []))
    if not tr_seeds:
        # fallback: generate 100 deterministic seeds
        rng = np.random.RandomState(12345)
        tr_seeds = [int(rng.randint(1, 10_000_000)) for _ in range(100)]
    # build features (structure-only, independent of regime)
    feats_list: List[np.ndarray] = []
    for s in tr_seeds:
        ds = _gen_ds_for_seed(int(s), tr_cfg)
        f, _ = _extract_features(ds)
        feats_list.append(f)
    F = np.stack(feats_list, axis=0)
    Fz, mu, sigma = _standardize(F)

    # Optional: restrict representatives to seeds matching a target queue regime
    target = str(a.target_regime).lower().strip()
    idxs: List[int]
    if target != "any":
        ds_conf = tr_cfg.get("dataset", {})
        dag_method = str(ds_conf.get("dag_method", "gnp"))
        gnp_min_n = int(ds_conf.get("gnp_min_n", 12))
        gnp_max_n = int(ds_conf.get("gnp_max_n", 30))
        gnp_p = ds_conf.get("gnp_p", None)
        # fabric and arrival settings
        h = int(ds_conf.get("host_count", 4))
        v = int(ds_conf.get("vm_count", 10))
        max_mem_gb = int(ds_conf.get("max_memory_gb", 10))
        min_mips = int(ds_conf.get("min_cpu_speed", 500))
        max_mips = int(ds_conf.get("max_cpu_speed", 5000))
        wf_count = int(ds_conf.get("workflow_count", 1))
        tdist = str(ds_conf.get("task_length_dist", "normal"))
        tmin = int(ds_conf.get("min_task_length", 500))
        tmax = int(ds_conf.get("max_task_length", 100_000))
        tarr = str(ds_conf.get("task_arrival", "static"))
        arate = float(ds_conf.get("arrival_rate", 3.0))

        cand_idx: List[int] = []
        for i, s in enumerate(tr_seeds):
            ds_seed = generate_dataset(
                seed=int(s),
                host_count=h,
                vm_count=v,
                max_memory_gb=max_mem_gb,
                min_cpu_speed_mips=min_mips,
                max_cpu_speed_mips=max_mips,
                workflow_count=wf_count,
                dag_method=dag_method,
                gnp_min_n=gnp_min_n,
                gnp_max_n=gnp_max_n,
                task_length_dist=tdist,
                min_task_length=tmin,
                max_task_length=tmax,
                task_arrival=tarr,
                arrival_rate=arate,
                gnp_p=(None if gnp_p is None else float(gnp_p)),
                req_divisor=int(a.regime_req_divisor),
            )
            label, _ = classify_queue_regime(ds_seed.workflows, ds_seed.vms, alpha_divisor=float(a.regime_req_divisor), tol=float(a.regime_tol))
            if label == target:
                cand_idx.append(i)

        if cand_idx:
            centers = _select_k_representatives_restricted(Fz, cand_idx, int(a.k))
            # Fill up to k if not enough candidates by adding global centers
            if len(centers) < int(a.k):
                extra = _select_k_representatives(Fz, int(a.k))
                for e in extra:
                    if e not in centers:
                        centers.append(e)
                    if len(centers) >= int(a.k):
                        break
            idxs = centers[: int(a.k)]
        else:
            idxs = _select_k_representatives(Fz, int(a.k))
    else:
        idxs = _select_k_representatives(Fz, int(a.k))

    selected_seeds = [int(tr_seeds[i]) for i in idxs]

    # coverage metrics
    centers = Fz[idxs]
    d = np.sqrt(((Fz[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    dmin = d.min(axis=1)
    cov_radius = float(dmin.max())
    cov_avg = float(dmin.mean())
    # cluster sizes by nearest center
    assign = d.argmin(axis=1)
    cluster_sizes = [int((assign == j).sum()) for j in range(len(idxs))]

    out = {
        "selected_eval_seeds": selected_seeds,
        "coverage": {
            "radius": cov_radius,
            "avg_min_dist": cov_avg,
            "cluster_sizes": cluster_sizes,
        },
        "train_seeds": [int(s) for s in tr_seeds],
        "feature_means": [float(x) for x in mu.tolist()],
        "feature_stds": [float(x) for x in sigma.tolist()],
        "feature_names": _extract_features(_gen_ds_for_seed(int(tr_seeds[0]), tr_cfg))[1] if tr_seeds else [],
    }
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(out, indent=2))

    # optional plots and CSV
    if a.save_plot_dir:
        import pandas as pd
        import matplotlib.pyplot as plt

        outd = Path(a.save_plot_dir)
        outd.mkdir(parents=True, exist_ok=True)
        names = out["feature_names"]
        df = pd.DataFrame(F, columns=names)
        df["seed"] = tr_seeds
        df.to_csv(outd / "train_features.csv", index=False)
        # simple 2D scatter of (Pbar, burstiness)
        try:
            x = df["Pbar"].values
            y = df["burstiness"].values
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(x, y, s=25, alpha=0.6, label="train")
            sel_mask = df["seed"].isin(selected_seeds)
            ax.scatter(x[sel_mask], y[sel_mask], s=80, alpha=0.95, marker="X", edgecolors="black", label="selected (eval)")
            ax.set_xlabel("Average parallelism P̄")
            ax.set_ylabel("Burstiness B")
            ax.set_title("Representativeness: train vs selected eval")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(outd / "representative_scatter.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

    print(f"Selected eval seeds: {selected_seeds}")
    print(f"Coverage radius={cov_radius:.3f} avg_min_dist={cov_avg:.3f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
