"""
多起源 + 空间 真值模拟器 (multi-origin spatial trajectory simulator)
====================================================================

为验证 causal-Hodge 的核心创新——【多起源发现 + 空间 TME 整合】——而造的真值数据。
现有 dyngen 风格基准都是单起源, 无法考"找几个根、找对哪几个根"。

生成模型 (完全可控、真值明确):
  - K 个独立起源 (progenitor), 每个长出一棵小的分叉谱系树;
  - 每个里程碑有一个隐空间程序向量, 子节点 = 父节点 + 随机方向×边长 (轨迹平滑);
    不同起源加大的隐空间偏移 => 起源身份可区分;
  - 细胞沿边插值采样, 真值伪时间 = 从本起源根累积的边长;
  - 表达 = softplus(隐向量 @ 基因载荷) 经 Poisson 采样 (适配 scanpy 标准流程);
  - 空间: 每个起源占一个 TME 区室中心 (sep 控分离度), 谱系内细胞按伪时间向外成扇形铺开
    => 空间结构携带轨迹信息 (causal-Hodge 可利用); sep 小 => 区室交叠 (鲁棒性考验)。

真值写入 AnnData:
  uns['timecourse']            每细胞真值伪时间 (各起源根均为 0)
  uns['milestone_network']     森林边表 from/to/length/directed (跨所有起源)
  uns['milestone_percentages'] 每细胞对所在边两端里程碑的隶属 (cell_id, milestone_id, percentage)
  uns['roots']                 真值根里程碑 id 列表 (多个!)
  uns['start_ids']             同上 (兼容)
  uns['n_origins']             K
  obs['origin']                每细胞真值起源 0..K-1
  obs['branch']                每细胞所在边 (段) 标签
  obsm['spatial']              2D 坐标

用法:
  conda activate cascat
  python simulate/make_multiorigin_spatial.py --out datasets_multiroot
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad


def make_origin_tree(rng, n_branches):
    """造一棵小分叉树: 返回 milestones(节点), edges[(u,v,length)]。节点 0 为根。"""
    # 根 -> 干 -> 分叉; n_branches 控制叶数
    edges = []
    nodes = [0]
    # 干
    edges.append((0, 1, rng.uniform(0.8, 1.2))); nodes.append(1)
    stem = 1
    leaves = max(2, n_branches)
    # 从干分出 leaves 条分支, 每条 1-2 段
    nid = 2
    for _ in range(leaves):
        seg = rng.integers(1, 3)
        parent = stem
        for _s in range(seg):
            edges.append((parent, nid, rng.uniform(0.8, 1.2)))
            nodes.append(nid); parent = nid; nid += 1
    return nodes, edges


def depth_map(edges, root=0):
    """每节点从根的累积边长 (用于真值伪时间)。"""
    adj = {}
    for u, v, l in edges:
        adj.setdefault(u, []).append((v, l))
    depth = {root: 0.0}
    stack = [root]
    while stack:
        u = stack.pop()
        for v, l in adj.get(u, []):
            depth[v] = depth[u] + l
            stack.append(v)
    return depth


def simulate(name, K=3, n_genes=300, latent_dim=12, cells_per_edge=70,
             sep=8.0, spatial_time_scale=3.0, noise_lat=0.12, lib=400.0,
             branches=(2, 3, 2, 3), id_gap=4.0, seed=0,
             conflict=False, confound=False, n_niche=3, n_confound_genes=20,
             confound_strength=2.0, condition=None):
    """
    conflict : 空间-谱系冲突 (对抗条件)。各分支空间方向收窄到同一扇区 -> 不同分支细胞
               空间交织, 纯空间 MST 会误连; 考验 Φ 单调+流向是否纠正。
    confound : 注入 niche 混杂变量 (考验 env-CMI)。每细胞按空间分箱得 niche; 末 n_confound_genes
               个基因额外受 niche 驱动 (与谱系无关) -> 产生 niche 引起的伪基因相关, env-adjust 应去除。
    condition: 条件标签 (写入 uns, 供分层统计)。
    """
    rng = np.random.default_rng(seed)
    angular_spread = 0.6 if conflict else 2 * np.pi

    # 基因载荷 (隐 -> 表达), 非负
    W = np.abs(rng.normal(0, 1, size=(n_genes, latent_dim)))
    gene_base = rng.uniform(0.05, 0.3, size=n_genes)
    # 共享分化 marker: 隐 dim0 = 谱系内进展(late), dim1 = 1-进展(early), 跨所有起源可比。
    # g0..g4 主载 dim0 (late markers), g5..g9 主载 dim1 (early markers)。
    LATE = [f"g{j}" for j in range(5)]
    EARLY = [f"g{j}" for j in range(5, 10)]
    W[:10, :2] *= 0.05                       # 这 10 个基因几乎只受 marker 轴驱动
    for j in range(5):
        W[j, 0] = 3.0
    for j in range(5, 10):
        W[j, 1] = 3.0

    cell_latent, cell_time, cell_origin, cell_branch = [], [], [], []
    cell_spatial = []
    mn_rows = []          # milestone_network
    mp_rows = []          # milestone_percentages
    roots_global = []
    # 起源区室中心 (圆周分布)
    centers = np.array([[sep * np.cos(2 * np.pi * k / K),
                         sep * np.sin(2 * np.pi * k / K)] for k in range(K)])

    cell_id = 0
    for k in range(K):
        nb = branches[k % len(branches)]
        nodes, edges = make_origin_tree(rng, nb)
        depth = depth_map(edges, root=0)
        # 每起源在隐空间一个偏移 (表达身份差异, id_gap 控大小; 小 -> 表达模糊, 需靠空间区分)
        origin_offset = rng.normal(0, id_gap, size=latent_dim)
        origin_offset[:2] = 0.0
        zpos = {0: origin_offset.copy()}
        for u, v, l in edges:
            direction = rng.normal(0, 1, size=latent_dim)
            direction[:2] = 0.0                  # 身份维不含 marker 轴
            direction /= (np.linalg.norm(direction) + 1e-9)
            zpos[v] = zpos[u] + direction * l
        # marker 轴: dim0=谱系内进展(归一), dim1=1-进展; 跨所有起源同尺度
        dmax = max(depth.values()) or 1.0
        for m in depth:
            prog = depth[m] / dmax
            zpos[m][0] = 3.0 * prog
            zpos[m][1] = 3.0 * (1.0 - prog)
        mid = lambda m: f"O{k}_M{m}"
        roots_global.append(mid(0))
        for u, v, l in edges:
            mn_rows.append((mid(u), mid(v), float(l), True))
        # 每条边方向角 (空间扇形铺开); conflict 时收窄到同一扇区 -> 分支空间交织
        base_ang = rng.uniform(0, 2 * np.pi)
        for ei, (u, v, l) in enumerate(edges):
            ang = base_ang + rng.uniform(-angular_spread / 2, angular_spread / 2)
            edir = np.array([np.cos(ang), np.sin(ang)])
            for i in range(cells_per_edge):
                t = (i + 0.5) / cells_per_edge
                z = zpos[u] * (1 - t) + zpos[v] * t + rng.normal(0, noise_lat, latent_dim)
                tt = depth[u] + t * l
                cell_latent.append(z)
                cell_time.append(tt)
                cell_origin.append(k)
                br = f"O{k}_E{ei}"
                cell_branch.append(br)
                # 空间: 起源中心 + 沿边方向×(伪时间) + 噪声
                xy = centers[k] + edir * (tt * spatial_time_scale / max(1.0, K)) \
                     + rng.normal(0, 0.6, 2)
                cell_spatial.append(xy)
                # 隶属: 边两端里程碑 (1-t, t)
                mp_rows.append((f"cell{cell_id}", mid(u), float(1 - t)))
                mp_rows.append((f"cell{cell_id}", mid(v), float(t)))
                cell_id += 1

    Z = np.vstack(cell_latent)
    n = Z.shape[0]
    sp = np.vstack(cell_spatial)
    raw = np.log1p(np.exp(Z @ W.T)) + gene_base           # softplus + base
    niche = None
    confound_genes = []
    if confound:
        # niche = 空间分箱 (与空间相关、跨谱系); 末 n_confound_genes 个基因受同一 per-niche
        # 加性驱动 -> 这些基因因 niche 而共变 (相互混杂), 与谱系 latent 无关。
        bx = (sp[:, 0] > np.median(sp[:, 0])).astype(int)
        by = (sp[:, 1] > np.median(sp[:, 1])).astype(int)
        niche = ((bx * 2 + by) % n_niche).astype(int)
        cg = np.arange(n_genes - n_confound_genes, n_genes)
        confound_genes = [f"g{j}" for j in cg]
        # 乘性 niche 驱动 (库归一化前): 同组基因随 niche 同向放大 -> 因 niche 共变。
        niche_mult = np.linspace(0.0, confound_strength, n_niche)
        for nb in range(n_niche):
            idx = np.where(niche == nb)[0]
            if idx.size:
                raw[np.ix_(idx, cg)] *= (1.0 + niche_mult[nb])
    rate = raw / raw.sum(1, keepdims=True) * lib          # 库大小归一
    counts = rng.poisson(rate).astype(np.float32)

    obs_d = {
        "origin": pd.Categorical([str(o) for o in cell_origin]),
        "branch": pd.Categorical(cell_branch),
    }
    if niche is not None:
        obs_d["niche"] = pd.Categorical([str(x) for x in niche])
    obs = pd.DataFrame(obs_d, index=[f"cell{i}" for i in range(n)])
    A = ad.AnnData(X=counts, obs=obs)
    A.var_names = [f"g{j}" for j in range(n_genes)]
    A.obsm["spatial"] = np.vstack(cell_spatial)
    A.uns["timecourse"] = np.asarray(cell_time, float)
    A.uns["milestone_network"] = pd.DataFrame(mn_rows, columns=["from", "to", "length", "directed"])
    A.uns["milestone_percentages"] = pd.DataFrame(mp_rows, columns=["cell_id", "milestone_id", "percentage"])
    A.uns["roots"] = roots_global
    A.uns["start_ids"] = roots_global
    A.uns["n_origins"] = int(K)
    A.uns["late_markers"] = LATE
    A.uns["early_markers"] = EARLY
    A.uns["sim_name"] = name
    A.uns["condition"] = condition if condition is not None else name
    A.uns["conflict"] = bool(conflict)
    A.uns["confound"] = bool(confound)
    A.uns["confound_genes"] = confound_genes
    A.uns["seed"] = int(seed)
    return A


PANEL = [
    # name, K, sep(分离度), id_gap(表达身份差异), seed
    dict(name="sim_K2_sep", K=2, sep=10.0, id_gap=4.0, seed=1, branches=(2, 3)),
    dict(name="sim_K3_sep", K=3, sep=10.0, id_gap=4.0, seed=2, branches=(2, 3, 2)),
    dict(name="sim_K4_sep", K=4, sep=11.0, id_gap=4.0, seed=3, branches=(2, 3, 2, 3)),
    dict(name="sim_K2_mix", K=2, sep=3.0,  id_gap=4.0, seed=4, branches=(3, 2)),    # 空间交叠
    dict(name="sim_K3_mix", K=3, sep=3.5,  id_gap=4.0, seed=5, branches=(2, 2, 3)), # 空间交叠
    # 表达模糊 + 空间分离: 同种程序在不同区室各发生一次, 只有空间能分开起源 (空间优势测例)
    dict(name="sim_K2_ambig", K=2, sep=12.0, id_gap=0.6, seed=6, branches=(2, 3)),
    dict(name="sim_K3_ambig", K=3, sep=13.0, id_gap=0.6, seed=7, branches=(2, 3, 2)),
]


def grid_conditions():
    """Tier-1 条件网格 (每条件 × N 复制): 用于 powered + 分层统计。
    返回 [(condition_name, base_cfg_without_seed), ...]。"""
    conds = []
    for K in (2, 3, 4):
        br = tuple((2, 3, 2, 3)[:K])
        # 规则区室 (清晰可分)
        conds.append((f"sep_K{K}", dict(K=K, sep=10.0, id_gap=4.0, branches=br)))
        # 空间交叠 (mix, TME 最常见)
        conds.append((f"mix_K{K}", dict(K=K, sep=3.0, id_gap=4.0, branches=br)))
        # 表达模糊 + 空间可分 (空间优势测例)
        conds.append((f"ambig_K{K}", dict(K=K, sep=12.0, id_gap=0.6, branches=br)))
    # 对抗条件 (专为证 Φ 约束价值): 空间-谱系冲突 / 低 SNR
    for K in (3,):
        br = tuple((2, 3, 2, 3)[:K])
        conds.append((f"conflict_K{K}", dict(K=K, sep=10.0, id_gap=4.0, branches=br, conflict=True)))
        conds.append((f"lowsnr_K{K}", dict(K=K, sep=10.0, id_gap=4.0, branches=br, noise_lat=0.45)))
        conds.append((f"confound_K{K}", dict(K=K, sep=10.0, id_gap=4.0, branches=br, confound=True, confound_strength=3.0)))
    return conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets_multiroot")
    ap.add_argument("--cells_per_edge", type=int, default=70)
    ap.add_argument("--grid", action="store_true", help="生成条件网格 × N 复制 (Tier-1 powered)")
    ap.add_argument("--reps", type=int, default=30, help="每条件复制数")
    ap.add_argument("--seed0", type=int, default=1000, help="复制 seed 起点")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if not args.grid:
        for cfg in PANEL:
            A = simulate(cells_per_edge=args.cells_per_edge, **cfg)
            p = out / f"{cfg['name']}.h5ad"
            A.write_h5ad(p)
            print(f"[sim] {cfg['name']}: n={A.n_obs} K={A.uns['n_origins']} -> {p}")
        return

    manifest = []
    for cname, base in grid_conditions():
        for r in range(args.reps):
            seed = args.seed0 + r
            name = f"{cname}_rep{r:02d}"
            A = simulate(name=name, condition=cname, seed=seed,
                         cells_per_edge=args.cells_per_edge, **base)
            A.write_h5ad(out / f"{name}.h5ad")
            manifest.append(dict(dataset=name, condition=cname, rep=r, seed=seed,
                                 K=A.uns["n_origins"], n_cells=A.n_obs,
                                 conflict=A.uns["conflict"], confound=A.uns["confound"]))
        print(f"[grid] {cname}: {args.reps} reps written")
    pd.DataFrame(manifest).to_csv(out / "manifest.csv", index=False)
    print(f"[grid] {len(manifest)} datasets + manifest.csv -> {out}")


if __name__ == "__main__":
    main()
