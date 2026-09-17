"""AEDPC: single-file implementation of the proposed clustering method.

The implementation includes adaptive density estimation, density-flow routing,
local ellipsoid construction, contact evidence, and certified reconstruction.
Internal statistical and geometric tests belong to the clustering algorithm.
No baseline algorithms, experiment runners, or result-analysis scripts are included.

Use Python 3.12 with the accompanying requirements and license records.

Entry points:
    AEDPC / AutomaticAEDPCv2: automatic parent/fusion certification.
    AEDPCCore / AEDPCv2: core engine, including the dedicated STL10 settings.
    AEDPCConfig / AEDPCv2Config: algorithm configuration.

Example (X is a samples-by-features NumPy array):
    from AEDPC import AEDPC, AEDPCConfig
    config = AEDPCConfig(neighbor_backend="exact")
    labels = AEDPC(alpha_n=0.01, alpha_m=0.05, config=config).fit_predict(X)
"""

from __future__ import annotations


# ============================================================================
# Original module: aedpcv2/config.py
# ============================================================================

"""Frozen configuration for the AEDPCv2 research implementation."""


from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class AEDPCv2Config:
    """Configuration with two statistical controls and disclosed fixed conventions."""

    alpha_n: float = 0.01
    alpha_m: float = 0.05
    neighbor_backend: Literal["auto", "exact", "hnsw"] = "auto"
    random_seed: int = 20260824
    exact_max_n: int = 3000
    max_neighbors: int = 350
    abide_iterations: int = 5
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 400
    boundary_coverage: float = 0.95
    null_permutations: int = 499
    null_swaps_per_edge: int = 10
    n_jobs: int = 1
    epsilon: float = 1e-12
    adaptive_neighborhood: bool = True
    prominence_mode: Literal["confidence", "lower_bound", "raw"] = "lower_bound"
    routing_scale: Literal["adaptive", "stabilized", "calibrated", "pointwise"] = "pointwise"
    reciprocity_mode: Literal["mutual", "directed"] = "mutual"
    parent_mode: Literal["topology", "nearest"] = "topology"
    geometry_mode: Literal["adaptive", "intrinsic", "spherical", "full"] = "adaptive"
    evidence: Literal["B", "BG", "BGF", "BGS"] = "BG"
    fusion_mode: Literal[
        "automatic", "correction", "stability", "hybrid", "null", "greedy", "none"
    ] = "correction"
    density_mode: Literal["pak"] = "pak"
    pak_solver: Literal["auto", "optimized", "pointwise"] = "auto"
    low_id_cutoff: float = 2.5
    optional_repair: Literal[False] = False
    shared_neighbor_evidence: bool = True
    topology_supported_ellipsoid: bool = False
    ellipsoid_contact_mode: Literal["cross", "center"] = "cross"
    markov_time_mode: Literal["adaptive", "one_step"] = "adaptive"
    density_topography: bool = True
    sequential_null: bool = True
    persistent_topography_time: bool = True
    persistent_point_manifold: bool = True
    partition_lineage_persistence: bool = True
    crossfit_ellipsoidal_speciation: bool = True
    speciation_conditioned_topography: bool = True
    certified_collapse_guard: bool = True

    def __post_init__(self) -> None:
        if self.density_mode != "pak":
            raise ValueError("AEDPC uses PAk density; density_mode must be 'pak'")
        if self.optional_repair is not False:
            raise ValueError("optional_repair is not part of the AEDPC method")
        if not 0 < self.alpha_n < 1:
            raise ValueError("alpha_n must lie in (0, 1)")
        if not 0 < self.alpha_m < 1:
            raise ValueError("alpha_m must lie in (0, 1)")
        if not 0 < self.boundary_coverage < 1:
            raise ValueError("boundary_coverage must lie in (0, 1)")
        if self.max_neighbors < 2 or self.abide_iterations < 1:
            raise ValueError("max_neighbors >= 2 and abide_iterations >= 1 are required")
        if self.null_permutations < 0 or self.null_swaps_per_edge < 0:
            raise ValueError("null settings must be non-negative")
        if self.n_jobs != 1:
            raise ValueError("the frozen reproducibility protocol requires n_jobs=1")
        if self.low_id_cutoff <= 0:
            raise ValueError("low_id_cutoff must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

# ============================================================================
# Original module: aedpcv2/utils.py
# ============================================================================

"""Numerical helpers shared by AEDPCv2 modules."""


from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_float_matrix(X: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("X must be a two-dimensional numeric array")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("X must contain at least two rows and one feature")
    return array


def mean_impute_zscore(X: ArrayLike) -> tuple[NDArray[np.float64], dict[str, NDArray]]:
    """Mean-impute then apply ddof=0 z-scoring; constant scales are set to one."""
    array = as_float_matrix(X).copy()
    finite = np.isfinite(array)
    if np.any(np.all(~finite, axis=0)):
        bad = np.flatnonzero(np.all(~finite, axis=0)).tolist()
        raise ValueError(f"features with no finite observations: {bad}")
    means = np.nanmean(np.where(finite, array, np.nan), axis=0)
    rows, cols = np.where(~finite)
    array[rows, cols] = means[cols]
    scales = np.std(array, axis=0, ddof=0)
    constant = scales <= np.finfo(np.float64).eps
    scales[constant] = 1.0
    standardized = (array - means) / scales
    return standardized, {"mean": means, "scale": scales, "constant": constant}






def relabel_consecutive(labels: ArrayLike) -> NDArray[np.int64]:
    labels = np.asarray(labels)
    unique = sorted(np.unique(labels).tolist())
    mapping = {label: idx for idx, label in enumerate(unique)}
    return np.asarray([mapping[label] for label in labels], dtype=np.int64)


def connected_components(nodes: Iterable[int], edges: Iterable[tuple[int, int]]) -> list[set[int]]:
    adjacency = {int(node): set() for node in nodes}
    for left, right in edges:
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    components: list[set[int]] = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            unseen.discard(node)
            stack.extend(sorted(adjacency[node] - component, reverse=True))
        components.append(component)
    return components

# ============================================================================
# Original module: aedpcv2/components.py
# ============================================================================

"""Topology-first higher-prominence routing and density-flow components."""


import numpy as np
from numpy.typing import NDArray



def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def route_parents(
    prominence: NDArray[np.float64],
    reciprocal: list[set[int]],
    distance_lookup: list[dict[int, float]],
    radius: NDArray[np.float64],
    epsilon: float = 1e-12,
    mode: str = "topology",
) -> NDArray[np.int64]:
    """Apply the deterministic lexicographic parent rule; roots point to -1."""
    parent, _ = route_parents_with_ranks(
        prominence,
        reciprocal,
        distance_lookup,
        radius,
        epsilon,
        mode,
    )
    return parent


def route_parents_with_ranks(
    prominence: NDArray[np.float64],
    reciprocal: list[set[int]],
    distance_lookup: list[dict[int, float]],
    radius: NDArray[np.float64],
    epsilon: float = 1e-12,
    mode: str = "topology",
) -> tuple[NDArray[np.int64], list[dict[int, int]]]:
    """Return the parent forest and every higher-prominence candidate's one-based rank.

    The first-ranked candidate is the actual parent.  Lower-ranked candidates are retained as a
    counterfactual ascent diagnostic for component-boundary transport; they never rewrite the
    parent forest.
    """

    n = prominence.size
    parent = np.full(n, -1, dtype=np.int64)
    ranks: list[dict[int, int]] = []
    for i in range(n):
        higher = [j for j in reciprocal[i] if prominence[j] > prominence[i]]
        if not higher:
            ranks.append({})
            continue

        def key(j: int) -> tuple[float, float, float, int]:
            distance = distance_lookup[i].get(j, distance_lookup[j].get(i))
            if distance is None:
                raise RuntimeError("closed reciprocal edge has no computed neighbor distance")
            scaled = distance / np.sqrt(max(radius[i] * radius[j], epsilon))
            if mode == "nearest":
                return (0.0, -scaled, prominence[j], -j)
            return (jaccard(reciprocal[i], reciprocal[j]), -scaled, prominence[j], -j)

        ordered = sorted(higher, key=key, reverse=True)
        parent[i] = ordered[0]
        ranks.append({int(node): rank for rank, node in enumerate(ordered, start=1)})
    return parent, ranks


def validate_parent_forest(parent: NDArray[np.int64], prominence: NDArray[np.float64]) -> None:
    n = parent.size
    for i, node in enumerate(parent):
        if node >= 0 and not prominence[node] > prominence[i]:
            raise RuntimeError("parent must have strictly higher prominence")
    state = np.zeros(n, dtype=np.int8)
    for start in range(n):
        node = start
        trail: list[int] = []
        while node >= 0 and state[node] == 0:
            state[node] = 1
            trail.append(node)
            node = int(parent[node])
        if node >= 0 and state[node] == 1:
            raise RuntimeError("parent graph contains a cycle")
        for visited in trail:
            state[visited] = 2


def labels_from_parent(parent: NDArray[np.int64]) -> NDArray[np.int64]:
    roots = np.full(parent.size, -1, dtype=np.int64)
    for i in range(parent.size):
        node = i
        trail: list[int] = []
        while parent[node] >= 0:
            trail.append(node)
            node = int(parent[node])
        roots[i] = node
        for visited in trail:
            roots[visited] = node
    return relabel_consecutive(roots)


def distance_maps(
    distances: NDArray[np.float64], indices: NDArray[np.int64]
) -> list[dict[int, float]]:
    return [
        {int(j): float(d) for d, j in zip(distances[i, 1:], indices[i, 1:])}
        for i in range(indices.shape[0])
    ]

# ============================================================================
# Original module: aedpcv2/density.py
# ============================================================================

"""Official DADApy ABIDE/PAk adapter and confidence-aware prominence."""


import contextlib
import io
from importlib.metadata import version
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray



@dataclass(frozen=True, slots=True)
class DensityResult:
    intrinsic_dimension: float
    intrinsic_dimension_path: NDArray[np.float64]
    intrinsic_dimension_error_path: NDArray[np.float64]
    kstar: NDArray[np.int64]
    kstar_path: NDArray[np.int64]
    log_density: NDArray[np.float64]
    log_density_error: NDArray[np.float64]
    radius: NDArray[np.float64]
    pvalue_path: NDArray[np.float64]
    dadapy_version: str
    pak_optimized: bool
    pak_equal_shells: int




def run_abide_pak(
    distances: NDArray[np.float64],
    indices: NDArray[np.int64],
    config: AEDPCv2Config,
) -> DensityResult:
    """Run DADApy 0.3.4 ABIDE iterations followed by official PAk density."""
    try:
        import dadapy
        from dadapy import Data
    except ImportError as exc:  # pragma: no cover - installation failure
        raise RuntimeError("official DADApy 0.3.4 is required") from exc

    safe_distances = np.asarray(distances, dtype=np.float64).copy()
    positive = safe_distances[:, 1:]
    machine_epsilon = np.finfo(np.float64).eps
    zero_mask = positive < machine_epsilon
    # DADApy's public ABIDE implementation is undefined when all first-neighbor
    # radii are identical zeros. A deterministic, rank-ordered machine-scale
    # regularization keeps duplicate observations while leaving every nonzero
    # geometric distance unchanged at working precision.
    nonzero = positive[~zero_mask]
    tie_scale = max(
        machine_epsilon,
        (float(np.min(nonzero)) if nonzero.size else 1.0) * 1e-9,
    )
    rank_floor = tie_scale * np.arange(1, positive.shape[1] + 1, dtype=np.float64)
    positive[zero_mask] = np.broadcast_to(rank_floor, positive.shape)[zero_mask]
    maxk = safe_distances.shape[1] - 1
    data = Data(
        distances=(safe_distances, np.asarray(indices, dtype=np.int64)),
        maxk=maxk,
        verbose=False,
        n_jobs=1,
    )
    # DADApy 0.3.4 prints one line per ABIDE iteration even with verbose=False.
    legacy_state = np.random.get_state()
    np.random.seed(config.random_seed)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ids, ids_err, kstars, pvalues = data.return_ids_kstar_binomial(
                n_iter=config.abide_iterations,
                alpha=config.alpha_n,
                bonferroni_deloc=False,
                bonferroni_loc=False,
                plot_mv=False,
                k_bootstrap=1,
            )
            active = safe_distances[:, 1 : int(np.max(data.kstar)) + 1]
            if active.shape[1] > 1:
                tolerance = 10.0 * np.finfo(np.float64).resolution
                equal_shells = int(
                    np.count_nonzero(
                        np.abs(active[:, 1:] / np.maximum(active[:, :-1], machine_epsilon) - 1.0)
                        < tolerance
                    )
                )
            else:
                equal_shells = 0
            if config.pak_solver == "optimized":
                pak_optimized = True
            elif config.pak_solver == "pointwise":
                pak_optimized = False
            else:
                # DADApy's vectorized Newton solver can stall on singular Hessians
                # induced by repeated non-zero distance shells.  Its official
                # point-wise solver is mathematically equivalent and terminates
                # promptly on those data.
                pak_optimized = equal_shells == 0
            log_density, log_density_error = data.compute_density_PAk(
                alpha=config.alpha_n,
                optimized=pak_optimized,
                bonferroni_deloc=False,
                bonferroni_loc=False,
            )
    finally:
        np.random.set_state(legacy_state)
    kstar = np.asarray(data.kstar, dtype=np.int64)
    kstar = np.clip(kstar, 1, maxk)
    radius = safe_distances[np.arange(safe_distances.shape[0]), kstar]
    return DensityResult(
        intrinsic_dimension=float(data.intrinsic_dim),
        intrinsic_dimension_path=np.asarray(ids, dtype=np.float64),
        intrinsic_dimension_error_path=np.asarray(ids_err, dtype=np.float64),
        kstar=kstar,
        kstar_path=np.asarray(kstars, dtype=np.int64),
        log_density=np.asarray(log_density, dtype=np.float64),
        log_density_error=np.asarray(log_density_error, dtype=np.float64),
        radius=np.maximum(radius, config.epsilon),
        pvalue_path=np.asarray(pvalues, dtype=np.float64),
        dadapy_version=version("dadapy"),
        pak_optimized=pak_optimized,
        pak_equal_shells=equal_shells,
    )


def confidence_prominence(
    log_density: NDArray[np.float64],
    log_density_error: NDArray[np.float64],
    candidate_neighbors: list[set[int]],
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Compute a density-shift-invariant, uncertainty-aware local prominence."""
    log_density = np.asarray(log_density, dtype=np.float64)
    error = np.asarray(log_density_error, dtype=np.float64)
    prominence = np.empty_like(log_density)
    for i, neighbors in enumerate(candidate_neighbors):
        local = np.fromiter(sorted(neighbors | {i}), dtype=np.int64)
        values = log_density[local]
        values = values[np.isfinite(values)]
        if values.size == 0 or not np.isfinite(log_density[i]) or not np.isfinite(error[i]):
            prominence[i] = np.nan
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        denominator = np.sqrt((1.4826 * mad) ** 2 + error[i] ** 2 + epsilon**2)
        prominence[i] = (log_density[i] - median) / denominator
    return prominence


def repair_nonfinite_prominence(
    prominence: NDArray[np.float64],
    candidate_neighbors: list[set[int]],
) -> tuple[NDArray[np.float64], int]:
    """Deterministically repair isolated non-finite official-density outputs.

    PAk can return a non-finite value at a repeated or numerically singular shell.  A single
    such value previously poisoned the global BGF continuity scale.  The repair uses only
    feature-derived neighbor topology: a bad point receives the median finite prominence in
    its candidate neighborhood, falling back to the global finite median.  Labels are never
    consulted.
    """
    repaired = np.asarray(prominence, dtype=np.float64).copy()
    finite = np.isfinite(repaired)
    count = int(np.count_nonzero(~finite))
    if count == 0:
        return repaired, 0
    if not np.any(finite):
        raise RuntimeError("density estimation produced no finite prominence values")
    global_median = float(np.median(repaired[finite]))
    original = repaired.copy()
    for i in np.flatnonzero(~finite):
        local = np.fromiter(sorted(candidate_neighbors[int(i)] | {int(i)}), dtype=np.int64)
        local_values = original[local]
        local_values = local_values[np.isfinite(local_values)]
        repaired[i] = float(np.median(local_values)) if local_values.size else global_median
    if not np.all(np.isfinite(repaired)):
        raise RuntimeError("non-finite prominence repair failed")
    return repaired, count


def lower_confidence_prominence(
    log_density: NDArray[np.float64],
    log_density_error: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """One-sided density lower confidence bound used for acyclic routing.

    Unlike a point-wise local z-score, this ordering remains comparable along
    an entire manifold while conservatively demoting uncertain density peaks.
    Adding a constant to all log densities leaves every parent decision intact.
    """
    critical_value = NormalDist().inv_cdf(1.0 - alpha)
    return np.asarray(log_density, dtype=np.float64) - critical_value * np.asarray(
        log_density_error, dtype=np.float64
    )

# ============================================================================
# Original module: aedpcv2/ellipsoids.py
# ============================================================================

"""Intrinsic-subspace shrinkage ellipsoids for density-flow micro-components."""


from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import OAS
from sklearn.decomposition import PCA


@dataclass(frozen=True, slots=True)
class Ellipsoid:
    component: int
    members: NDArray[np.int64]
    center: NDArray[np.float64]
    basis: NDArray[np.float64]
    covariance: NDArray[np.float64]
    precision: NDArray[np.float64]
    residual_variance: float
    isotropic_variance: float
    directional_reliability: float
    oas_shrinkage: float
    axis_ratio: float
    rank_deficient: bool
    geometry_kind: str
    zeta: float
    boundary_weights: NDArray[np.float64]
    small_component: bool

    def squared_distance(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        centered = np.atleast_2d(np.asarray(X, dtype=np.float64)) - self.center
        isotropic = np.sum(centered * centered, axis=1) / self.isotropic_variance
        if self.basis.shape[1] == 0:
            return np.maximum(isotropic, 0.0)
        projected = centered @ self.basis
        in_subspace = np.einsum("ni,ij,nj->n", projected, self.precision, projected)
        residual = centered - projected @ self.basis.T
        orthogonal = np.sum(residual * residual, axis=1) / self.residual_variance
        anisotropic = np.maximum(in_subspace + orthogonal, 0.0)
        blended = (
            self.directional_reliability * anisotropic
            + (1.0 - self.directional_reliability) * isotropic
        )
        return np.maximum(blended, 0.0)

    def distance(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sqrt(self.squared_distance(X))


def _positive_floor(X: NDArray[np.float64], epsilon: float) -> float:
    if X.shape[0] <= 1:
        return max(float(np.mean(np.var(X, axis=0))) if X.size else 0.0, epsilon)
    scale = float(np.trace(np.cov(X, rowvar=False, ddof=0))) / max(X.shape[1], 1)
    return max(scale * 1e-8, epsilon)


def fit_ellipsoids(
    X: NDArray[np.float64],
    labels: NDArray[np.int64],
    intrinsic_dimension: float,
    radius: NDArray[np.float64],
    boundary_coverage: float,
    epsilon: float,
    *,
    spherical: bool = False,
    full_space: bool = False,
    reliability_blend: bool = True,
    support_neighbors: list[set[int]] | None = None,
) -> tuple[list[Ellipsoid], NDArray[np.float64]]:
    n, dimension = X.shape
    q = int(np.clip(np.rint(intrinsic_dimension), 1, dimension))
    point_weights = np.ones(n, dtype=np.float64)
    ellipsoids: list[Ellipsoid] = []
    for component in range(int(labels.max()) + 1):
        members = np.flatnonzero(labels == component)
        values = X[members]
        center = np.mean(values, axis=0)
        originally_small = members.size < q + 2
        support_members = members
        if originally_small and support_neighbors is not None and not spherical:
            support = set(map(int, members))
            for member in members:
                support.update(support_neighbors[int(member)])
            support_members = np.fromiter(sorted(support), dtype=np.int64)
        fit_values = X[support_members]
        support_variance = float(np.sum(np.var(fit_values, axis=0, ddof=0)))
        small = support_members.size < q + 2 or support_variance <= epsilon
        if small or spherical:
            sigma = max(float(np.median(radius[members])), np.sqrt(epsilon))
            residual_variance = max(sigma * sigma, epsilon)
            isotropic_variance = residual_variance
            basis = np.empty((dimension, 0), dtype=np.float64)
            covariance = np.empty((0, 0), dtype=np.float64)
            precision = np.empty((0, 0), dtype=np.float64)
            directional_reliability = 0.0
            oas_shrinkage = 1.0
            axis_ratio = 1.0
            geometry_kind = "isotropic-small" if small else "isotropic-forced"
            raw = np.linalg.norm(values - center, axis=1) / np.sqrt(residual_variance)
            zeta = max(float(np.quantile(raw, boundary_coverage)), np.sqrt(epsilon))
            weights = np.ones(members.size, dtype=np.float64) if small else _weights(raw, zeta)
        else:
            qb = min(dimension if full_space else q, dimension, support_members.size - 2)
            pca = PCA(n_components=qb, svd_solver="full", random_state=0)
            projected = pca.fit_transform(fit_values)
            basis = np.asarray(pca.components_.T, dtype=np.float64)
            oas = OAS(store_precision=True, assume_centered=False).fit(projected)
            covariance = np.asarray(oas.covariance_, dtype=np.float64)
            precision = np.asarray(oas.precision_, dtype=np.float64)
            total_variance = float(np.sum(np.var(fit_values, axis=0, ddof=0)))
            retained = float(np.sum(np.var(projected, axis=0, ddof=0)))
            floor = _positive_floor(fit_values, epsilon)
            isotropic_variance = max(total_variance / max(dimension, 1), floor)
            discarded_dims = dimension - qb
            if discarded_dims > 0:
                raw_residual = max((total_variance - retained) / discarded_dims, floor)
            else:
                raw_residual = isotropic_variance
            oas_shrinkage = float(np.clip(oas.shrinkage_, 0.0, 1.0))
            residual_variance = max(
                (1.0 - oas_shrinkage) * raw_residual
                + oas_shrinkage * isotropic_variance,
                floor,
            )
            support = float(
                np.clip(
                    (support_members.size - qb - 1)
                    / max(support_members.size + qb + 1, 1),
                    0.0,
                    1.0,
                )
            )
            directional_reliability = (
                np.sqrt((1.0 - oas_shrinkage) * support)
                if reliability_blend
                else 1.0
            )
            eigenvalues = np.linalg.eigvalsh(covariance)
            axis_ratio = float(
                np.sqrt(
                    max(float(eigenvalues.max()), floor)
                    / max(float(eigenvalues.min()), floor)
                )
            )
            geometry_kind = (
                "topology-supported-adaptive"
                if originally_small and reliability_blend
                else "topology-supported-intrinsic"
                if originally_small
                else "adaptive-reliability-blend"
                if reliability_blend
                else "full-space"
                if full_space
                else "intrinsic-unblended"
            )
            proto = Ellipsoid(
                component=component,
                members=members,
                center=center,
                basis=basis,
                covariance=covariance,
                precision=precision,
                residual_variance=residual_variance,
                isotropic_variance=isotropic_variance,
                directional_reliability=directional_reliability,
                oas_shrinkage=oas_shrinkage,
                axis_ratio=axis_ratio,
                rank_deficient=support_members.size - 1 < dimension,
                geometry_kind=geometry_kind,
                zeta=1.0,
                boundary_weights=np.ones(members.size),
                small_component=False,
            )
            raw = proto.distance(values)
            zeta = max(float(np.quantile(raw, boundary_coverage)), np.sqrt(epsilon))
            weights = _weights(raw, zeta)
        point_weights[members] = weights
        for array in (members, center, basis, covariance, precision, weights):
            array.setflags(write=False)
        ellipsoids.append(
            Ellipsoid(
                component=component,
                members=members,
                center=center,
                basis=basis,
                covariance=covariance,
                precision=precision,
                residual_variance=float(residual_variance),
                isotropic_variance=float(isotropic_variance),
                directional_reliability=float(directional_reliability),
                oas_shrinkage=float(oas_shrinkage),
                axis_ratio=float(axis_ratio),
                rank_deficient=bool(support_members.size - 1 < dimension),
                geometry_kind=geometry_kind,
                zeta=float(zeta),
                boundary_weights=weights,
                small_component=small,
            )
        )
    point_weights.setflags(write=False)
    return ellipsoids, point_weights


def _weights(distance: NDArray[np.float64], zeta: float) -> NDArray[np.float64]:
    beta = np.asarray(distance, dtype=np.float64) / zeta
    return np.clip(2.0 * beta / (1.0 + beta * beta), 0.0, 1.0)


def ellipsoid_compatibility(left: Ellipsoid, right: Ellipsoid, epsilon: float) -> float:
    left_view = float(left.distance(right.center[None, :])[0]) / max(left.zeta, epsilon)
    right_view = float(right.distance(left.center[None, :])[0]) / max(right.zeta, epsilon)
    return float(np.exp(-0.5 * (left_view + right_view)))


def cross_envelope_compatibility(
    left: Ellipsoid,
    right: Ellipsoid,
    X: NDArray[np.float64],
    contact_pairs: list[tuple[int, int, float]],
    contact_weights: NDArray[np.float64],
    epsilon: float,
) -> float:
    """Measure anisotropic compatibility at observed cross-component contacts.

    A contact is compatible when each endpoint lies near or inside the other component's
    operational ellipsoid.  Only the excess outside the 95% envelope is penalized, so local
    centers do not need to overlap along a curved manifold.
    """

    if not contact_pairs:
        return 0.0
    left_points = X[np.asarray([i for i, _, _ in contact_pairs], dtype=np.int64)]
    right_points = X[np.asarray([j for _, j, _ in contact_pairs], dtype=np.int64)]
    left_in_right = right.distance(left_points) / max(right.zeta, epsilon)
    right_in_left = left.distance(right_points) / max(left.zeta, epsilon)
    excess = np.maximum(left_in_right - 1.0, 0.0) + np.maximum(
        right_in_left - 1.0, 0.0
    )
    scores = np.exp(-0.5 * excess)
    total = float(np.sum(contact_weights))
    if total <= epsilon:
        return float(np.mean(scores))
    return float(np.clip(np.sum(contact_weights * scores) / total, 0.0, 1.0))

# ============================================================================
# Original module: aedpcv2/graph.py
# ============================================================================

"""Boundary–Geometry–Flow micro-component evidence graph."""


from dataclasses import dataclass, replace
from math import erfc, sqrt

import numpy as np
from numpy.typing import NDArray



@dataclass(frozen=True, slots=True)
class MicroEdge:
    left: int
    right: int
    weight: float
    boundary: float
    geometry: float
    continuity: float
    contacts: int
    saddle_density: float = -np.inf
    saddle_error: float = np.inf
    saddle_z: float = np.inf
    saddle_continuity: float = 1.0
    flow_contacts: int = 0
    flow_identifiable: bool = False
    flow_enabled: bool = False
    flow_bidirectional: bool = False
    flow_strength: float = 0.0
    flow_balance: float = 0.0
    flow_forward: float = 0.0
    flow_reverse: float = 0.0


def build_micro_graph(
    X: NDArray[np.float64],
    labels: NDArray[np.int64],
    reciprocal: list[set[int]],
    distance_lookup: list[dict[int, float]],
    radius: NDArray[np.float64],
    log_density: NDArray[np.float64],
    log_density_error: NDArray[np.float64],
    prominence: NDArray[np.float64],
    boundary_weights: NDArray[np.float64],
    ellipsoids: list[Ellipsoid],
    epsilon: float,
    *,
    ascent_ranks: list[dict[int, int]] | None = None,
    evidence: str = "BGF",
    geometry_mode: str = "cross",
) -> list[MicroEdge]:
    n_components = int(labels.max()) + 1
    degrees = np.asarray([len(neighbors) for neighbors in reciprocal], dtype=np.float64)
    volumes = np.bincount(
        labels, weights=boundary_weights * degrees, minlength=n_components
    ).astype(np.float64)
    contacts: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for i, neighbors in enumerate(reciprocal):
        for j in sorted(neighbors):
            if i >= j or labels[i] == labels[j]:
                continue
            component_i = int(labels[i])
            component_j = int(labels[j])
            if component_i < component_j:
                left, right = component_i, component_j
                left_endpoint, right_endpoint = i, j
            else:
                left, right = component_j, component_i
                left_endpoint, right_endpoint = j, i
            distance = distance_lookup[i].get(j, distance_lookup[j].get(i))
            if distance is None:
                raise RuntimeError("BGF edge has no computed neighbor distance")
            delta = distance / np.sqrt(max(radius[i] * radius[j], epsilon))
            similarity = float(np.exp(-(delta**2)))
            # cross_envelope_compatibility assumes that the first endpoint belongs to
            # ``left`` and the second belongs to ``right``.  Sample-index order carries no
            # component semantics, so orient the endpoints with the sorted component key.
            contacts.setdefault((left, right), []).append(
                (left_endpoint, right_endpoint, similarity)
            )

    peak_density = np.full(n_components, -np.inf, dtype=np.float64)
    peak_error = np.full(n_components, np.inf, dtype=np.float64)
    for component in range(n_components):
        members = np.flatnonzero(labels == component)
        peak = int(members[np.argmax(log_density[members])])
        peak_density[component] = float(log_density[peak])
        peak_error[component] = float(max(log_density_error[peak], epsilon))
    result: list[MicroEdge] = []
    for (left, right), pairs in sorted(contacts.items()):
        if any(labels[i] != left or labels[j] != right for i, j, _ in pairs):
            raise RuntimeError("cross-component contact endpoints are not component-oriented")
        pair_weights = np.asarray(
            [boundary_weights[i] * boundary_weights[j] * similarity for i, j, similarity in pairs],
            dtype=np.float64,
        )
        score_sum = float(pair_weights.sum())
        denominator = float(volumes[left] + volumes[right])
        boundary = float(np.clip(2.0 * score_sum / max(denominator, epsilon), 0.0, 1.0))
        if geometry_mode == "cross":
            geometry = cross_envelope_compatibility(
                ellipsoids[left],
                ellipsoids[right],
                X,
                pairs,
                pair_weights,
                epsilon,
            )
        elif geometry_mode == "center":
            geometry = ellipsoid_compatibility(
                ellipsoids[left], ellipsoids[right], epsilon
            )
        else:
            raise ValueError(f"unknown ellipsoid contact mode: {geometry_mode}")
        saddle_candidates = np.asarray(
            [min(log_density[i], log_density[j]) for i, j, _ in pairs], dtype=np.float64
        )
        saddle_index = int(np.argmax(saddle_candidates))
        saddle_density = float(saddle_candidates[saddle_index])
        saddle_left, saddle_right, _ = pairs[saddle_index]
        saddle_error = float(
            np.sqrt(
                log_density_error[saddle_left] ** 2
                + log_density_error[saddle_right] ** 2
            )
            / 2.0
        )
        z_left = (peak_density[left] - saddle_density) / np.sqrt(
            peak_error[left] ** 2 + saddle_error**2 + epsilon**2
        )
        z_right = (peak_density[right] - saddle_density) / np.sqrt(
            peak_error[right] ** 2 + saddle_error**2 + epsilon**2
        )
        saddle_z = max(min(float(z_left), float(z_right)), 0.0)
        saddle_continuity = float(
            np.clip(erfc(saddle_z / sqrt(2.0)), 0.0, 1.0)
        )
        flow_forward_numerator = 0.0
        flow_reverse_numerator = 0.0
        flow_denominator = 0.0
        flow_contacts = 0
        if ascent_ranks is not None:
            for (i, j, _), contact_weight in zip(pairs, pair_weights):
                rank = ascent_ranks[i].get(j)
                if rank is None:
                    rank = ascent_ranks[j].get(i)
                if rank is None:
                    continue
                source_component = int(labels[i]) if j in ascent_ranks[i] else int(labels[j])
                if source_component == left:
                    flow_forward_numerator += float(contact_weight) / float(rank)
                else:
                    flow_reverse_numerator += float(contact_weight) / float(rank)
                flow_denominator += float(contact_weight)
                flow_contacts += 1
        flow_identifiable = flow_denominator > epsilon
        if flow_identifiable:
            flow_forward = flow_forward_numerator / flow_denominator
            flow_reverse = flow_reverse_numerator / flow_denominator
            flow_strength = flow_forward + flow_reverse
            flow_balance = (
                2.0 * min(flow_forward, flow_reverse) / flow_strength
                if flow_strength > epsilon
                else 0.0
            )
            # A cross-component candidate cannot have rank one: such an edge would be the actual
            # parent and both endpoints would share a root.  Rank two is therefore the attainable
            # maximum counterfactual ascent, so the exchange score is normalized by that bound.
            continuity = float(
                np.clip(2.0 * flow_strength * flow_balance, 0.0, 1.0)
            )
        else:
            # Missing routing evidence is neutral, not negative: an SNN-only evidence edge did not
            # participate in parent choice and therefore cannot falsify ascent exchange.
            flow_forward = 0.0
            flow_reverse = 0.0
            flow_strength = 0.0
            flow_balance = 0.0
            continuity = 1.0
        flow_bidirectional = bool(flow_forward > 0 and flow_reverse > 0)
        pieces = {
            "B": float(np.nan_to_num(boundary, nan=0.0, posinf=0.0, neginf=0.0)),
            "G": float(np.nan_to_num(geometry, nan=0.0, posinf=0.0, neginf=0.0)),
            "F": float(np.nan_to_num(continuity, nan=0.0, posinf=0.0, neginf=0.0)),
            "S": float(
                np.nan_to_num(
                    saddle_continuity, nan=0.0, posinf=0.0, neginf=0.0
                )
            ),
        }
        selected = [
            pieces[token]
            for token in evidence
            if token in pieces and not (token == "F" and not flow_bidirectional)
        ]
        if not selected:
            raise ValueError("evidence must contain at least one of B, G, F")
        weight = float(np.prod(selected) ** (1.0 / len(selected)))
        if not np.isfinite(weight):
            raise RuntimeError("BGF construction produced a non-finite edge weight")
        result.append(
            MicroEdge(
                left=left,
                right=right,
                weight=float(np.clip(weight, 0.0, 1.0)),
                boundary=boundary,
                geometry=geometry,
                continuity=continuity,
                contacts=len(pairs),
                saddle_density=saddle_density,
                saddle_error=saddle_error,
                saddle_z=saddle_z,
                saddle_continuity=saddle_continuity,
                flow_contacts=flow_contacts,
                flow_identifiable=flow_identifiable,
                flow_bidirectional=flow_bidirectional,
                flow_strength=flow_strength,
                flow_balance=flow_balance,
                flow_forward=flow_forward,
                flow_reverse=flow_reverse,
            )
        )
    flow_enabled = bool(
        "F" in evidence
        and result
        and 2 * sum(edge.flow_identifiable for edge in result) > len(result)
    )
    if "F" in evidence:
        revised: list[MicroEdge] = []
        for edge in result:
            pieces = {
                "B": edge.boundary,
                "G": edge.geometry,
                "F": edge.continuity,
                "S": edge.saddle_continuity,
            }
            selected = [
                pieces[token]
                for token in evidence
                if token in pieces
                and not (
                    token == "F"
                    and (not flow_enabled or not edge.flow_bidirectional)
                )
            ]
            weight = float(np.prod(selected) ** (1.0 / len(selected)))
            revised.append(
                replace(
                    edge,
                    weight=float(np.clip(weight, 0.0, 1.0)),
                    flow_enabled=flow_enabled,
                )
            )
        result = revised
    return result

# ============================================================================
# Original module: aedpcv2/neighbors.py
# ============================================================================

"""Deterministic nearest-neighbor backends and reciprocal adaptive graphs."""


from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors



@dataclass(frozen=True, slots=True)
class NeighborResult:
    distances: NDArray[np.float64]
    indices: NDArray[np.int64]
    backend: str


@dataclass(frozen=True, slots=True)
class CalibratedNeighborhoods:
    """Two feature-only neighborhood views derived from ABIDE and topology."""

    routing_k: NDArray[np.int64]
    evidence_k: NDArray[np.int64]
    routing_candidate: list[set[int]]
    routing_reciprocal: list[set[int]]
    evidence_candidate: list[set[int]]
    evidence_reciprocal: list[set[int]]
    connectivity_floor: int
    natural_saturation_scale: int
    reciprocal_percolation_scale: int
    routing_closure_edges: int
    evidence_closure_edges: int
    routing_edges: int
    evidence_edges: int
    hubness_max_occurrence: int
    hubness_occurrence_skewness: float
    pointwise_status: str = "not-applicable"


def _canonicalize(
    distances: NDArray[np.float64], indices: NDArray[np.int64], maxk: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    n = indices.shape[0]
    canonical_d = np.empty((n, maxk + 1), dtype=np.float64)
    canonical_i = np.empty((n, maxk + 1), dtype=np.int64)
    for row in range(n):
        candidates = [(float(d), int(j)) for d, j in zip(distances[row], indices[row]) if j != row]
        candidates.sort(key=lambda pair: (pair[0], pair[1]))
        if len(candidates) < maxk:
            raise RuntimeError("nearest-neighbor backend returned too few non-self neighbors")
        canonical_d[row, 0] = 0.0
        canonical_i[row, 0] = row
        chosen = candidates[:maxk]
        canonical_d[row, 1:] = [pair[0] for pair in chosen]
        canonical_i[row, 1:] = [pair[1] for pair in chosen]
    return canonical_d, canonical_i


def compute_neighbors(X: NDArray[np.float64], config: AEDPCv2Config) -> NeighborResult:
    n, dimension = X.shape
    maxk = min(config.max_neighbors, n - 1)
    backend = config.neighbor_backend
    if backend == "auto":
        backend = "exact" if n <= config.exact_max_n else "hnsw"

    if backend == "exact":
        estimator = NearestNeighbors(
            n_neighbors=maxk + 1, algorithm="brute", metric="euclidean", n_jobs=1
        )
        estimator.fit(X)
        distances, indices = estimator.kneighbors(X, return_distance=True)
    elif backend == "hnsw":
        try:
            import hnswlib
        except ImportError as exc:  # pragma: no cover - dependency failure
            raise RuntimeError("hnsw backend requested but hnswlib is not installed") from exc
        index = hnswlib.Index(space="l2", dim=dimension)
        index.init_index(
            max_elements=n,
            ef_construction=config.hnsw_ef_construction,
            M=config.hnsw_m,
            random_seed=config.random_seed,
        )
        index.set_num_threads(1)
        insertion = np.arange(n, dtype=np.int64)
        index.add_items(X.astype(np.float32), insertion)
        index.set_ef(max(config.hnsw_ef_search, maxk + 1))
        indices, squared = index.knn_query(X.astype(np.float32), k=maxk + 1, num_threads=1)
        distances = np.sqrt(np.maximum(squared, 0.0), dtype=np.float64)
    else:  # pragma: no cover - validated by Literal in normal use
        raise ValueError(f"unknown neighbor backend: {backend}")

    distances, indices = _canonicalize(
        np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64), maxk
    )
    return NeighborResult(distances=distances, indices=indices, backend=backend)


def adaptive_neighbor_sets(
    indices: NDArray[np.int64], kstar: NDArray[np.int64]
) -> tuple[list[set[int]], list[set[int]]]:
    n, width = indices.shape
    candidate: list[set[int]] = []
    for i in range(n):
        k = int(np.clip(kstar[i], 1, width - 1))
        candidate.append(set(map(int, indices[i, 1 : k + 1])))
    reciprocal: list[set[int]] = []
    for i, neighbors in enumerate(candidate):
        reciprocal.append({j for j in neighbors if i in candidate[j]})
    return candidate, reciprocal




def reverse_neighbor_saturation_scale(indices: NDArray[np.int64]) -> tuple[int, int]:
    """Return a label-free topology scale from reverse-neighbor coverage.

    The lower bound is the standard logarithmic kNN connectivity scale.  Above it, the search
    stops at the first three-step plateau in the number of observations with zero reverse
    neighbors.  This is a deterministic Natural-Neighbor-style saturation rule, not a fitted
    clustering threshold.
    """

    n, width = indices.shape
    maximum = width - 1
    connectivity_floor = int(np.clip(np.ceil(np.log2(max(n, 2))), 2, maximum))
    incoming = np.zeros(n, dtype=np.int64)
    previous_zeros = n
    unchanged_steps = 0
    plateau = maximum
    for k in range(1, maximum + 1):
        np.add.at(incoming, indices[:, k], 1)
        zeros = int(np.count_nonzero(incoming == 0))
        if zeros == previous_zeros:
            unchanged_steps += 1
        else:
            unchanged_steps = 0
        previous_zeros = zeros
        if unchanged_steps >= 2:
            plateau = k
            break
    return connectivity_floor, max(connectivity_floor, plateau)


def _largest_reciprocal_component(reciprocal: list[set[int]]) -> int:
    unseen = set(range(len(reciprocal)))
    largest = 0
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            new = reciprocal[node] & unseen
            unseen.difference_update(new)
            stack.extend(sorted(new, reverse=True))
        largest = max(largest, size)
    return largest


def reciprocal_percolation_scale(
    indices: NDArray[np.int64], lower: int, upper: int
) -> int:
    """Choose the last k before the largest reciprocal giant-component jump."""

    if upper <= lower:
        return lower
    largest_sizes: list[int] = []
    for k in range(lower, upper + 1):
        fixed = np.full(indices.shape[0], k, dtype=np.int64)
        _, reciprocal = adaptive_neighbor_sets(indices, fixed)
        largest_sizes.append(_largest_reciprocal_component(reciprocal))
    jumps = np.diff(np.asarray(largest_sizes, dtype=np.int64))
    return lower + int(np.argmax(jumps))




def calibrated_neighbor_sets(
    distances: NDArray[np.float64],
    indices: NDArray[np.int64],
    shared_neighbor_evidence: bool = True,
) -> CalibratedNeighborhoods:
    """Construct the unified routing and BGF-evidence neighborhoods for AEDPCv2-r7."""

    lower, upper = reverse_neighbor_saturation_scale(indices)
    shell_caps = np.full(indices.shape[0], upper, dtype=np.int64)
    if upper > lower:
        machine = np.finfo(np.float64).eps
        for i in range(indices.shape[0]):
            for rank in range(lower, upper):
                if distances[i, rank] <= machine and distances[i, rank + 1] > machine:
                    shell_caps[i] = rank
                    break
    saturation_candidate, saturation_mutual = adaptive_neighbor_sets(indices, shell_caps)
    incoming = np.zeros(indices.shape[0], dtype=np.int64)
    for row in saturation_candidate:
        if row:
            np.add.at(incoming, np.fromiter(row, dtype=np.int64), 1)
    mean_occurrence = float(np.mean(incoming))
    std_occurrence = float(np.std(incoming))
    if std_occurrence > 0:
        occurrence_skewness = float(
            np.mean(((incoming - mean_occurrence) / std_occurrence) ** 3)
        )
    else:
        occurrence_skewness = 0.0

    percolation = reciprocal_percolation_scale(indices, lower, upper)
    if occurrence_skewness > 1.0:
        routing_k = shell_caps.copy()
        evidence_k = shell_caps.copy()
        routing_base_candidate = saturation_candidate
        routing_mutual = saturation_mutual
        base_candidate = saturation_candidate
        base_mutual = saturation_mutual
    else:
        routing_k = np.full(indices.shape[0], lower, dtype=np.int64)
        evidence_k = routing_k.copy()
        routing_base_candidate, routing_mutual = adaptive_neighbor_sets(indices, routing_k)
        base_candidate = routing_base_candidate
        base_mutual = routing_mutual

    pair_scores: dict[tuple[int, int], float] = {}
    incident: list[list[float]] = [[] for _ in range(indices.shape[0])]
    pairs = {
        (min(i, j), max(i, j))
        for i, row in enumerate(base_candidate)
        for j in row
        if i != j
    }
    for left, right in sorted(pairs):
        shared = len(base_candidate[left] & base_candidate[right])
        union = len(base_candidate[left] | base_candidate[right])
        snn = shared / union if union else 0.0
        hub_scale = np.sqrt(
            (1.0 + incoming[left] / max(float(evidence_k[left]), 1.0))
            * (1.0 + incoming[right] / max(float(evidence_k[right]), 1.0))
        )
        score = float(snn / hub_scale)
        pair_scores[(left, right)] = score
        if score > 0:
            incident[left].append(score)
            incident[right].append(score)

    thresholds = np.asarray(
        [np.median(values) if values else np.inf for values in incident], dtype=np.float64
    )
    routing_reciprocal = [set(row) for row in routing_mutual]
    evidence_reciprocal = [set(row) for row in base_mutual]
    if shared_neighbor_evidence:
        for (left, right), score in pair_scores.items():
            if score > 0 and score >= min(thresholds[left], thresholds[right]):
                evidence_reciprocal[left].add(right)
                evidence_reciprocal[right].add(left)
    routing_candidate = [set(row) for row in routing_reciprocal]
    evidence_candidate = [set(row) for row in evidence_reciprocal]
    routing_added = 0
    evidence_added = 0
    return CalibratedNeighborhoods(
        routing_k=routing_k,
        evidence_k=evidence_k,
        routing_candidate=routing_candidate,
        routing_reciprocal=routing_reciprocal,
        evidence_candidate=evidence_candidate,
        evidence_reciprocal=evidence_reciprocal,
        connectivity_floor=lower,
        natural_saturation_scale=upper,
        reciprocal_percolation_scale=percolation,
        routing_closure_edges=routing_added,
        evidence_closure_edges=evidence_added,
        routing_edges=sum(map(len, routing_reciprocal)) // 2,
        evidence_edges=sum(map(len, evidence_reciprocal)) // 2,
        hubness_max_occurrence=int(np.max(incoming)),
        hubness_occurrence_skewness=occurrence_skewness,
    )


def pointwise_calibrated_neighbor_sets(
    distances: NDArray[np.float64],
    indices: NDArray[np.int64],
    density_kstar: NDArray[np.int64],
    intrinsic_dimension: float,
    shared_neighbor_evidence: bool = True,
) -> CalibratedNeighborhoods:
    """Preserve ABIDE pointwise scales inside topology-derived support bounds.

    Routing first uses ``clip(k_i*, ceil(log2 n), k_Ti)``.  Persistent hubness is measured on the
    saturation graph, independently of the variable pointwise degrees.  The legacy calibrated
    graph is returned byte-for-byte when hubness persists, intrinsic dimension exceeds the
    connectivity floor, or a strict majority of points lack an interior adaptive scale.  Thus
    local scale adaptation is used exactly when it has both statistical and topological support.
    """

    lower, upper = reverse_neighbor_saturation_scale(indices)
    shell_caps = np.full(indices.shape[0], upper, dtype=np.int64)
    if upper > lower:
        machine = np.finfo(np.float64).eps
        for i in range(indices.shape[0]):
            for rank in range(lower, upper):
                if distances[i, rank] <= machine and distances[i, rank + 1] > machine:
                    shell_caps[i] = rank
                    break
    pointwise = np.asarray(density_kstar, dtype=np.int64)
    if pointwise.shape != (indices.shape[0],):
        raise ValueError("density_kstar must contain one scale per observation")
    saturation_candidate, _ = adaptive_neighbor_sets(indices, shell_caps)
    detector_incoming = np.zeros(indices.shape[0], dtype=np.int64)
    for row in saturation_candidate:
        if row:
            np.add.at(detector_incoming, np.fromiter(row, dtype=np.int64), 1)
    mean_occurrence = float(np.mean(detector_incoming))
    std_occurrence = float(np.std(detector_incoming))
    occurrence_skewness = (
        float(
            np.mean(
                ((detector_incoming - mean_occurrence) / std_occurrence) ** 3
            )
        )
        if std_occurrence > 0
        else 0.0
    )
    routing_k = np.minimum(np.maximum(pointwise, lower), shell_caps)
    if occurrence_skewness > 1.0:
        return replace(
            calibrated_neighbor_sets(
                distances, indices, shared_neighbor_evidence
            ),
            pointwise_status="hubness-saturation",
        )
    elif intrinsic_dimension >= lower:
        return replace(
            calibrated_neighbor_sets(
                distances, indices, shared_neighbor_evidence
            ),
            pointwise_status="intrinsic-dimension-floor",
        )
    elif upper - lower < 2 or 2 * int(np.count_nonzero(routing_k > lower)) <= indices.shape[0]:
        return replace(
            calibrated_neighbor_sets(
                distances, indices, shared_neighbor_evidence
            ),
            pointwise_status="insufficient-pointwise-support",
        )
    routing_base_candidate, routing_mutual = adaptive_neighbor_sets(indices, routing_k)
    evidence_k = routing_k.copy()
    evidence_base_candidate, evidence_mutual = adaptive_neighbor_sets(indices, evidence_k)
    evidence_incoming = np.zeros(indices.shape[0], dtype=np.int64)
    for row in evidence_base_candidate:
        if row:
            np.add.at(evidence_incoming, np.fromiter(row, dtype=np.int64), 1)

    pair_scores: dict[tuple[int, int], float] = {}
    incident: list[list[float]] = [[] for _ in range(indices.shape[0])]
    pairs = {
        (min(i, j), max(i, j))
        for i, row in enumerate(evidence_base_candidate)
        for j in row
        if i != j
    }
    for left, right in sorted(pairs):
        shared = len(evidence_base_candidate[left] & evidence_base_candidate[right])
        union = len(evidence_base_candidate[left] | evidence_base_candidate[right])
        snn = shared / union if union else 0.0
        hub_scale = np.sqrt(
            (1.0 + evidence_incoming[left] / max(float(evidence_k[left]), 1.0))
            * (1.0 + evidence_incoming[right] / max(float(evidence_k[right]), 1.0))
        )
        score = float(snn / hub_scale)
        pair_scores[(left, right)] = score
        if score > 0:
            incident[left].append(score)
            incident[right].append(score)
    thresholds = np.asarray(
        [np.median(values) if values else np.inf for values in incident],
        dtype=np.float64,
    )
    evidence_reciprocal = [set(row) for row in evidence_mutual]
    if shared_neighbor_evidence:
        for (left, right), score in pair_scores.items():
            if score > 0 and score >= min(thresholds[left], thresholds[right]):
                evidence_reciprocal[left].add(right)
                evidence_reciprocal[right].add(left)
    return CalibratedNeighborhoods(
        routing_k=routing_k,
        evidence_k=evidence_k,
        routing_candidate=[set(row) for row in routing_base_candidate],
        routing_reciprocal=[set(row) for row in routing_mutual],
        evidence_candidate=[set(row) for row in evidence_base_candidate],
        evidence_reciprocal=evidence_reciprocal,
        connectivity_floor=lower,
        natural_saturation_scale=upper,
        reciprocal_percolation_scale=reciprocal_percolation_scale(
            indices, lower, upper
        ),
        routing_closure_edges=0,
        evidence_closure_edges=0,
        routing_edges=sum(map(len, routing_mutual)) // 2,
        evidence_edges=sum(map(len, evidence_reciprocal)) // 2,
        hubness_max_occurrence=int(np.max(detector_incoming)),
        hubness_occurrence_skewness=occurrence_skewness,
        pointwise_status="pointwise",
    )

# ============================================================================
# Original module: aedpcv2/speciation.py
# ============================================================================

"""Cross-fitted one-ellipsoid speciation test for guarding Markov refinement."""


import hashlib
import numpy as np
import diptest
from numpy.typing import NDArray
from scipy.stats import beta as beta_distribution
from scipy.stats import binomtest, kstest
from sklearn.covariance import OAS
from sklearn.decomposition import PCA


def optimal_two_segment_index(values: NDArray[np.float64], epsilon: float = 1e-12) -> float:
    """Minimum one-dimensional two-group within/total sum-of-squares ratio."""

    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size < 4:
        return 1.0
    cumulative = np.cumsum(ordered)
    cumulative_sq = np.cumsum(ordered**2)
    left_n = np.arange(1, ordered.size, dtype=np.float64)
    right_n = np.arange(ordered.size - 1, 0, -1, dtype=np.float64)
    left = cumulative_sq[:-1] - cumulative[:-1] ** 2 / left_n
    right_sum = cumulative[-1] - cumulative[:-1]
    right_sq = cumulative_sq[-1] - cumulative_sq[:-1]
    right = right_sq - right_sum**2 / right_n
    total = float(np.sum((ordered - np.mean(ordered)) ** 2))
    return float(np.min(left + right) / max(total, epsilon))


def _single_crossfit_ellipsoidal_speciation(
    X: NDArray[np.float64],
    simulations: int,
    seed: int,
    epsilon: float = 1e-12,
    alpha: float = 0.05,
    axis_alpha: float = 0.01,
) -> tuple[float, dict[str, object]]:
    """Test one Gaussian ellipsoid against any stable two-state projection.

    Principal axes are learned on one half and evaluated on the held-out half.  The procedure is
    repeated in the opposite direction.  Each held-out split index is calibrated by standard
    Gaussian Monte Carlo. Directional and fold p-values are Bonferroni combined, retaining validity
    under their dependence. The test never supplies a cluster count or labels to AEDPC; it only
    decides whether Markov refinement is admissible.
    """

    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 8:
        return 1.0, {
            "mode": "crossfit-one-ellipsoid-speciation",
            "status": "insufficient-observations",
            "p_value": 1.0,
        }
    if simulations <= 0:
        return 1.0, {
            "mode": "crossfit-one-ellipsoid-speciation",
            "status": "no-Monte-Carlo-budget",
            "p_value": 1.0,
        }
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(values.shape[0])
    folds = ((permutation[::2], permutation[1::2]), (permutation[1::2], permutation[::2]))
    fold_records: list[dict[str, object]] = []
    p_values: list[float] = []
    dip_fold_p_values: list[float] = []
    angular_fold_p_values: list[float] = []
    angular_records: list[dict[str, float | int]] = []
    for fold_index, (train, test) in enumerate(folds):
        n_directions = min(10, values.shape[1], train.size - 1)
        projector = PCA(
            n_components=n_directions,
            svd_solver="randomized",
            random_state=seed + fold_index,
        )
        projector.fit(values[train])
        projected = projector.transform(values[test])
        null = np.asarray(
            [
                optimal_two_segment_index(rng.normal(size=test.size), epsilon)
                for _ in range(simulations)
            ],
            dtype=np.float64,
        )
        direction_records: list[dict[str, float]] = []
        directional_p: list[float] = []
        directional_dip_p: list[float] = []
        for direction in range(n_directions):
            observed = optimal_two_segment_index(projected[:, direction], epsilon)
            direction_p = float(
                (1 + np.count_nonzero(null <= observed)) / (simulations + 1)
            )
            directional_p.append(direction_p)
            _, dip_p = diptest.diptest(projected[:, direction], boot_pval=False)
            dip_p = float(dip_p)
            directional_dip_p.append(dip_p)
            direction_records.append(
                {
                    "direction": direction,
                    "observed_index": observed,
                    "p_value": direction_p,
                    "dip_p_value": dip_p,
                    "explained_variance_ratio": float(
                        projector.explained_variance_ratio_[direction]
                    ),
                }
            )
        p_value = float(min(1.0, n_directions * min(directional_p)))
        dip_fold_p = float(min(1.0, n_directions * min(directional_dip_p)))
        p_values.append(p_value)
        dip_fold_p_values.append(dip_fold_p)
        fold_records.append(
            {
                "fold": fold_index,
                "train_size": int(train.size),
                "test_size": int(test.size),
                "p_value": p_value,
                "dip_p_value": dip_fold_p,
                "null_median": float(np.median(null)),
                "null_q05": float(np.quantile(null, 0.05)),
                "directions": direction_records,
            }
        )
        if values.shape[1] <= 1:
            angular_fold_p_values.append(1.0)
            angular_records.append(
                {"fold": fold_index, "p_value": 1.0, "pairs": int(test.size // 2)}
            )
        else:
            covariance = OAS(store_precision=False).fit(values[train])
            eigenvalues, eigenvectors = np.linalg.eigh(covariance.covariance_)
            inverse_root = eigenvectors @ np.diag(
                1.0 / np.sqrt(np.maximum(eigenvalues, epsilon))
            ) @ eigenvectors.T
            whitened = (values[test] - covariance.location_) @ inverse_root
            whitened /= np.maximum(
                np.linalg.norm(whitened, axis=1, keepdims=True), epsilon
            )
            paired_size = 2 * (whitened.shape[0] // 2)
            cosine = np.sum(
                whitened[:paired_size:2] * whitened[1:paired_size:2], axis=1
            )
            squared = cosine**2
            beta_null = beta_distribution(0.5, (values.shape[1] - 1) / 2.0)
            radial_p = float(kstest(squared, beta_null.cdf).pvalue)
            sign_p = float(
                binomtest(int(np.count_nonzero(cosine > 0)), cosine.size, 0.5).pvalue
            )
            angular_p = float(min(1.0, 2.0 * min(radial_p, sign_p)))
            angular_fold_p_values.append(angular_p)
            angular_records.append(
                {
                    "fold": fold_index,
                    "p_value": angular_p,
                    "cosine_squared_p_value": radial_p,
                    "sign_symmetry_p_value": sign_p,
                    "pairs": int(cosine.size),
                }
            )
    gaussian_p = float(min(1.0, 2.0 * min(p_values)))
    dip_p = float(min(1.0, 2.0 * min(dip_fold_p_values)))
    angular_p = float(min(1.0, 2.0 * min(angular_fold_p_values)))
    combined = float(max(gaussian_p, dip_p))
    direction_count = len(fold_records[0]["directions"])
    directional_threshold = alpha / max(2 * direction_count, 1)
    direction_p_matrix = np.asarray(
        [
            [record["p_value"] for record in fold["directions"]]
            for fold in fold_records
        ],
        dtype=np.float64,
    )
    minimum_direction_p = np.min(
        direction_p_matrix,
        axis=0,
    )
    maximum_direction_p = np.max(
        direction_p_matrix,
        axis=0,
    )
    replicated_leading_axes = 0
    for direction in range(direction_count):
        if np.all(direction_p_matrix[:, direction] <= axis_alpha):
            replicated_leading_axes += 1
        else:
            break
    significant_axes = int(
        np.count_nonzero(minimum_direction_p <= directional_threshold)
    )
    gaussian_minimum_states = significant_axes + 1 if gaussian_p <= alpha else 1
    minimum_states = gaussian_minimum_states if combined <= alpha else 1
    return combined, {
        "mode": "crossfit-one-ellipsoid-speciation",
        "status": "tested",
        "p_value": combined,
        "gaussian_cluster_index_p_value": gaussian_p,
        "dip_unimodality_p_value": dip_p,
        "elliptical_angular_symmetry_p_value": angular_p,
        "elliptical_angular_symmetry_folds": angular_records,
        "gaussian_ellipsoid_rejected": bool(gaussian_p <= alpha),
        "unimodality_rejected": bool(dip_p <= alpha),
        "elliptical_angular_symmetry_rejected": bool(angular_p <= alpha),
        "folds": fold_records,
        "simulations_per_fold": simulations,
        "combination": "intersection-union-max-of-Gaussian-CI-and-dip-p-values",
        "alpha": alpha,
        "directional_threshold": directional_threshold,
        "axis_alpha": axis_alpha,
        "minimum_direction_p": minimum_direction_p.tolist(),
        "maximum_direction_p": maximum_direction_p.tolist(),
        "significant_axes": significant_axes,
        "gaussian_state_lower_bound": gaussian_minimum_states,
        "replicated_leading_axes": replicated_leading_axes,
        "minimum_states": minimum_states,
    }


def _canonical_feature_digest(X: NDArray[np.float64]) -> bytes:
    """Hash an order-invariant canonical representation of the observed feature array."""

    values = np.asarray(X, dtype=np.float64)
    order = np.lexsort(
        tuple(values[:, column] for column in range(values.shape[1] - 1, -1, -1))
    )
    return hashlib.sha256(np.ascontiguousarray(values[order]).tobytes()).digest()


def feature_determined_crossfit_seeds(
    X: NDArray[np.float64], count: int = 5
) -> tuple[int, ...]:
    """Derive reproducible cross-fit seeds from features, independent of user RNG state."""

    base = _canonical_feature_digest(X)
    return tuple(
        int.from_bytes(
            hashlib.sha256(base + index.to_bytes(4, "little")).digest()[:4],
            "little",
        )
        & 0x7FFFFFFF
        for index in range(count)
    )


def cauchy_combine_pvalues(p_values: list[float]) -> float:
    """Cauchy-combine dependent cross-fit p-values without selecting a favorable fold."""

    values = np.clip(np.asarray(p_values, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    statistic = float(np.mean(np.tan(np.pi * (0.5 - values))))
    return float(np.clip(0.5 - np.arctan(statistic) / np.pi, 0.0, 1.0))


def crossfit_ellipsoidal_speciation(
    X: NDArray[np.float64],
    simulations: int,
    seed: int,
    epsilon: float = 1e-12,
    alpha: float = 0.05,
    axis_alpha: float = 0.01,
) -> tuple[float, dict[str, object]]:
    """Feature-determined robust evidence across five cross-fit realizations.

    The public random seed continues to govern neighbor approximation, graph nulls, and optimizer
    starts. Speciation alone is a statistic of the observed feature array: five canonical feature
    seeds produce cross-fitted Gaussian-CI, dip, and angular p-values, which are combined within
    evidence type by the dependence-robust Cauchy method. The Gaussian state lower bound uses the
    median across cross-fits and therefore cannot be raised by a single favorable split.
    """

    del seed
    seeds = feature_determined_crossfit_seeds(X)
    results = [
        _single_crossfit_ellipsoidal_speciation(
            X, simulations, selected, epsilon, alpha, axis_alpha
        )
        for selected in seeds
    ]
    records = [result[1] for result in results]
    gaussian_p = cauchy_combine_pvalues(
        [float(record["gaussian_cluster_index_p_value"]) for record in records]
    )
    dip_p = cauchy_combine_pvalues(
        [float(record["dip_unimodality_p_value"]) for record in records]
    )
    angular_p = cauchy_combine_pvalues(
        [float(record["elliptical_angular_symmetry_p_value"]) for record in records]
    )
    gaussian_minimum_states = int(
        np.median([int(record.get("gaussian_state_lower_bound", 1)) for record in records])
    )
    replicated_leading_axes = int(
        np.median([int(record.get("replicated_leading_axes", 0)) for record in records])
    )
    combined = float(max(gaussian_p, dip_p))
    diagnostics = dict(records[0])
    diagnostics.update(
        {
            "mode": "feature-determined-five-crossfit-Cauchy-speciation",
            "status": "tested",
            "p_value": combined,
            "gaussian_cluster_index_p_value": gaussian_p,
            "dip_unimodality_p_value": dip_p,
            "elliptical_angular_symmetry_p_value": angular_p,
            "gaussian_ellipsoid_rejected": bool(gaussian_p <= alpha),
            "unimodality_rejected": bool(dip_p <= alpha),
            "elliptical_angular_symmetry_rejected": bool(angular_p <= alpha),
            "gaussian_state_lower_bound": (
                gaussian_minimum_states if gaussian_p <= alpha else 1
            ),
            "minimum_states": (
                gaussian_minimum_states if combined <= alpha else 1
            ),
            "replicated_leading_axes": replicated_leading_axes,
            "feature_determined_crossfit_seeds": list(seeds),
            "crossfit_evidence_p_values": [
                {
                    "gaussian": record["gaussian_cluster_index_p_value"],
                    "dip": record["dip_unimodality_p_value"],
                    "angular": record["elliptical_angular_symmetry_p_value"],
                }
                for record in records
            ],
            "combination": "coordinatewise-Cauchy-over-five-feature-determined-crossfits",
            "folds": [],
        }
    )
    return combined, diagnostics

# ============================================================================
# Original module: aedpcv2/fusion.py
# ============================================================================

"""Analytic stationary set flow and degree-preserving max-statistic null fusion."""


from dataclasses import dataclass
import hashlib
from statistics import NormalDist

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as beta_distribution



@dataclass(frozen=True, slots=True)
class FusionStep:
    connected_component: int
    iteration: int
    left: tuple[int, ...]
    right: tuple[int, ...]
    observed: float
    p_value: float
    merged: bool
    null_max_median: float
    null_max_q95: float
    successful_swaps: int
    requested_swaps: int
    degenerate_nulls: int
    multigraph_fallbacks: int


@dataclass(frozen=True, slots=True)
class MarkovCorrectionStep:
    """Single global BGF Markov-stability correction decision."""

    preliminary_clusters: int
    candidate_clusters: int
    preliminary_stability: float
    candidate_stability: float
    observed_improvement: float
    p_value: float
    corrected: bool
    null_improvement_median: float
    null_improvement_q95: float
    degenerate_nulls: int
    multigraph_fallbacks: int




def stationary_set_flow(
    adjacency: NDArray[np.float64], left: set[int], right: set[int], epsilon: float = 1e-12
) -> float:
    left_idx = np.fromiter(sorted(left), dtype=np.int64)
    right_idx = np.fromiter(sorted(right), dtype=np.int64)
    cut = float(adjacency[np.ix_(left_idx, right_idx)].sum())
    degrees = adjacency.sum(axis=1)
    volume_left = float(degrees[left_idx].sum())
    volume_right = float(degrees[right_idx].sum())
    if cut <= 0 or volume_left <= 0 or volume_right <= 0:
        return 0.0
    return float(cut / (2.0 * np.sqrt(max(volume_left * volume_right, epsilon))))






def _best_pair_edges(
    n_nodes: int,
    edge_pairs: NDArray[np.int64],
    weights: NDArray[np.float64],
    groups: list[set[int]],
) -> tuple[int, int, float] | None:
    """Edge-linear equivalent of ``_best_pair`` for sparse micro-graphs."""
    membership = np.full(n_nodes, -1, dtype=np.int64)
    for group_index, group in enumerate(groups):
        membership[list(group)] = group_index
    degree = np.zeros(n_nodes, dtype=np.float64)
    if edge_pairs.size:
        np.add.at(degree, edge_pairs[:, 0], weights)
        np.add.at(degree, edge_pairs[:, 1], weights)
    volumes = np.asarray([degree[list(group)].sum() for group in groups], dtype=np.float64)
    cuts: dict[tuple[int, int], float] = {}
    for (left, right), weight in zip(edge_pairs, weights):
        group_left = int(membership[int(left)])
        group_right = int(membership[int(right)])
        if group_left < 0 or group_right < 0 or group_left == group_right:
            continue
        key = tuple(sorted((group_left, group_right)))
        cuts[key] = cuts.get(key, 0.0) + float(weight)
    best: tuple[int, int, float] | None = None
    for (left, right), cut in cuts.items():
        denominator = 2.0 * np.sqrt(max(volumes[left] * volumes[right], 1e-12))
        score = float(cut / denominator)
        if score > 0 and (best is None or (score, -left, -right) > (best[2], -best[0], -best[1])):
            best = (left, right, score)
    return best


def _rewired_edge_list(
    n_nodes: int,
    edge_pairs: NDArray[np.int64],
    weights: NDArray[np.float64],
    rng: np.random.Generator,
    swaps_per_edge: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64], int, int, bool, bool]:
    n_edges = int(edge_pairs.shape[0])
    requested = swaps_per_edge * n_edges
    if n_edges < 2 or requested == 0:
        return edge_pairs.copy(), weights.copy(), 0, requested, True, False
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(map(tuple, edge_pairs.tolist()))
    before = {tuple(sorted(edge)) for edge in graph.edges()}
    fallback = False
    try:
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        nx.double_edge_swap(
            graph,
            nswap=requested,
            max_tries=max(requested * 30, 100),
            seed=seed,
        )
        null_pairs = np.asarray(sorted(graph.edges()), dtype=np.int64)
        successful = requested
    except (nx.NetworkXAlgorithmError, nx.NetworkXError):
        degree_sequence_values = [degree for _, degree in graph.degree()]
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        multigraph = nx.configuration_model(degree_sequence_values, seed=seed)
        null_pairs = np.asarray(list(multigraph.edges()), dtype=np.int64)
        successful = 0
        fallback = True
    null_weights = weights[rng.permutation(weights.size)]
    if null_pairs.shape[0] != null_weights.size:
        null_weights = np.resize(null_weights, null_pairs.shape[0])
    after = {tuple(sorted(map(int, edge))) for edge in null_pairs.tolist()}
    degenerate = before == after
    return null_pairs, null_weights, successful, requested, degenerate, fallback






def density_topography_components(
    initial_labels: NDArray[np.int64],
    edges: list[MicroEdge],
    peak_density: NDArray[np.float64],
    peak_error: NDArray[np.float64],
    alpha: float,
    epsilon: float = 1e-12,
) -> tuple[NDArray[np.int64], dict[str, object]]:
    """Merge reciprocal macro-components whose lower peak is not distinct from the saddle.

    This is a DPA-inspired hierarchical operation on density-flow components.  It starts from
    the reciprocal topology partition and can only merge groups; no target K is supplied.
    """

    labels = relabel_consecutive(np.asarray(initial_labels, dtype=np.int64))
    n_groups = int(labels.max()) + 1
    parent = np.arange(n_groups, dtype=np.int64)
    group_peak_density = np.full(n_groups, -np.inf, dtype=np.float64)
    group_peak_error = np.full(n_groups, np.inf, dtype=np.float64)
    for group in range(n_groups):
        nodes = np.flatnonzero(labels == group)
        peak_node = int(nodes[np.argmax(peak_density[nodes])])
        group_peak_density[group] = peak_density[peak_node]
        group_peak_error[group] = max(float(peak_error[peak_node]), epsilon)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    critical = NormalDist().inv_cdf(1.0 - alpha)
    accepted: list[dict[str, object]] = []
    rejected = 0
    ordered = sorted(
        (edge for edge in edges if np.isfinite(edge.saddle_density)),
        key=lambda edge: (-edge.saddle_density, edge.saddle_z, edge.left, edge.right),
    )
    for edge in ordered:
        left = find(int(labels[edge.left]))
        right = find(int(labels[edge.right]))
        if left == right:
            continue
        if group_peak_density[left] <= group_peak_density[right]:
            lower, higher = left, right
        else:
            lower, higher = right, left
        error = np.sqrt(
            group_peak_error[lower] ** 2 + edge.saddle_error**2 + epsilon**2
        )
        z_score = (group_peak_density[lower] - edge.saddle_density) / error
        if z_score <= critical:
            parent[lower] = higher
            if group_peak_density[lower] > group_peak_density[higher]:
                group_peak_density[higher] = group_peak_density[lower]
                group_peak_error[higher] = group_peak_error[lower]
            accepted.append(
                {
                    "left": int(edge.left),
                    "right": int(edge.right),
                    "saddle_density": float(edge.saddle_density),
                    "z_score": float(z_score),
                }
            )
        else:
            rejected += 1

    merged = np.asarray([find(int(label)) for label in labels], dtype=np.int64)
    merged = relabel_consecutive(merged)
    return merged, {
        "mode": "peak-saddle-density-topography",
        "critical_z": float(critical),
        "initial_clusters": n_groups,
        "selected_clusters": int(np.unique(merged).size),
        "accepted_merges": len(accepted),
        "rejected_saddles": rejected,
        "merge_records": accepted,
    }


def _labels_to_groups(labels: NDArray[np.int64]) -> list[set[int]]:
    labels = relabel_consecutive(np.asarray(labels, dtype=np.int64))
    return [
        set(np.flatnonzero(labels == group).tolist())
        for group in range(int(labels.max()) + 1)
    ]


def mass_weighted_partition_similarity(
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    masses: NDArray[np.float64] | None = None,
    epsilon: float = 1e-12,
) -> float:
    """Symmetric information retention between two micro-component partitions.

    The contingency table is weighted by the number of original observations represented by each
    density-flow micro-component.  The result is the arithmetic normalized mutual information,
    bounded in ``[0, 1]`` and invariant to community-label permutations.  It is used only to trace
    a partition lineage over Markov time; it never compares against reference labels.
    """

    left_labels = relabel_consecutive(np.asarray(left, dtype=np.int64))
    right_labels = relabel_consecutive(np.asarray(right, dtype=np.int64))
    if left_labels.shape != right_labels.shape or left_labels.ndim != 1:
        raise ValueError("partitions must be one-dimensional and have equal length")
    if masses is None:
        weight = np.ones(left_labels.size, dtype=np.float64)
    else:
        weight = np.asarray(masses, dtype=np.float64)
        if weight.shape != left_labels.shape:
            raise ValueError("partition masses must match the number of nodes")
        if np.any(~np.isfinite(weight)) or np.any(weight < 0):
            raise ValueError("partition masses must be finite and nonnegative")
    total = float(weight.sum())
    if total <= epsilon:
        return 1.0
    contingency = np.zeros(
        (int(left_labels.max()) + 1, int(right_labels.max()) + 1),
        dtype=np.float64,
    )
    np.add.at(contingency, (left_labels, right_labels), weight / total)
    left_mass = contingency.sum(axis=1)
    right_mass = contingency.sum(axis=0)
    positive_left = left_mass > 0
    positive_right = right_mass > 0
    entropy_left = float(-np.sum(left_mass[positive_left] * np.log(left_mass[positive_left])))
    entropy_right = float(
        -np.sum(right_mass[positive_right] * np.log(right_mass[positive_right]))
    )
    if entropy_left <= epsilon and entropy_right <= epsilon:
        return 1.0
    expected = left_mass[:, None] * right_mass[None, :]
    present = contingency > 0
    mutual_information = float(
        np.sum(contingency[present] * np.log(contingency[present] / expected[present]))
    )
    similarity = 2.0 * mutual_information / max(entropy_left + entropy_right, epsilon)
    return float(np.clip(similarity, 0.0, 1.0))


def _partition_signature(labels: NDArray[np.int64]) -> str:
    values = np.asarray(labels, dtype=np.int64)
    canonical = np.empty_like(values)
    mapping: dict[int, int] = {}
    for index, value in enumerate(values.tolist()):
        if value not in mapping:
            mapping[value] = len(mapping)
        canonical[index] = mapping[value]
    return hashlib.sha256(canonical.astype("<i8", copy=False).tobytes()).hexdigest()


def _select_fragmentation_guarded_lineage_index(
    partitions: list[NDArray[np.int64]],
    stabilities: list[float],
    connected_components_count: int,
    masses: NDArray[np.float64] | None = None,
    epsilon: float = 1e-12,
) -> tuple[int, str, list[float], int, list[dict[str, float | int]]]:
    """Resolve partition lineages without confusing equal counts with equal memberships.

    Exact repetition certifies persistence.  Membership drift is crossed only while the candidate
    count is at least twice the graph's disconnected support floor, reusing AEDPC's existing
    two-fragments-per-state definition.  Near that floor, further diffusion can only reshuffle
    nearly irreducible supports, so the earlier partition is retained.  No target K or similarity
    threshold is supplied.
    """

    if not partitions:
        return 0, "no-partitions", [], 0, []
    similarities = [
        mass_weighted_partition_similarity(left, right, masses, epsilon)
        for left, right in zip(partitions[:-1], partitions[1:])
    ]
    counts = [int(np.unique(partition).size) for partition in partitions]
    same_count_drift = 0
    interval_records: list[dict[str, float | int]] = []
    for index, similarity in enumerate(similarities):
        if counts[index] != counts[index + 1]:
            continue
        if similarity >= 1.0 - epsilon:
            if counts[index] == connected_components_count and index >= 1:
                return (
                    index - 1,
                    "lineage-repeat-before-disconnected-limit",
                    similarities,
                    same_count_drift,
                    interval_records,
                )
            return (
                index,
                "lineage-repeat-selected",
                similarities,
                same_count_drift,
                interval_records,
            )
        same_count_drift += 1
        fragmentation_remains = bool(
            counts[index] >= 2 * connected_components_count
        )
        interval_records.append(
            {
                "left_index": index,
                "right_index": index + 1,
                "clusters": counts[index],
                "disconnected_support_floor": connected_components_count,
                "lineage_similarity": similarity,
                "fragmentation_remains": int(fragmentation_remains),
            }
        )
        if fragmentation_remains:
            continue
        return index, "lineage-conflict-near-support-floor", similarities, same_count_drift, interval_records
    first_positive = next(
        (index for index, value in enumerate(stabilities) if value > epsilon),
        0,
    )
    return first_positive, "first-positive-fallback", similarities, same_count_drift, interval_records


def _select_count_plateau_index(
    partitions: list[NDArray[np.int64]],
    stabilities: list[float],
    connected_components_count: int,
    epsilon: float = 1e-12,
) -> tuple[int, str]:
    """Frozen r25 count-only selector retained solely for a controlled ablation."""

    counts = [int(np.unique(partition).size) for partition in partitions]
    for index in range(len(counts) - 1):
        if counts[index] != counts[index + 1]:
            continue
        if counts[index] == connected_components_count and index >= 1:
            return index - 1, "count-plateau-before-disconnected-limit"
        return index, "count-plateau-selected"
    first_positive = next(
        (index for index, value in enumerate(stabilities) if value > epsilon),
        0,
    )
    return first_positive, "first-positive-fallback"


def _weighted_graph(
    n_nodes: int, edge_pairs: NDArray[np.int64], weights: NDArray[np.float64]
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    for (left, right), weight in zip(edge_pairs, weights):
        left_i, right_i = int(left), int(right)
        value = float(weight)
        if not np.isfinite(value) or value <= 0:
            continue
        previous = float(graph.get_edge_data(left_i, right_i, {}).get("weight", 0.0))
        graph.add_edge(left_i, right_i, weight=previous + value)
    return graph


def _best_markov_stability_partition(
    graph: nx.Graph,
    seed: int,
    resolution: float = 1.0,
    starts: int = 8,
    epsilon: float = 1e-12,
) -> tuple[NDArray[np.int64], float]:
    n_nodes = graph.number_of_nodes()
    if graph.number_of_edges() == 0 or graph.size(weight="weight") <= 0:
        return np.arange(n_nodes, dtype=np.int64), 0.0
    best_labels: NDArray[np.int64] | None = None
    best_score = -np.inf
    best_signature: tuple[tuple[int, ...], ...] | None = None
    for offset in range(starts):
        groups = [
            set(map(int, group))
            for group in nx.community.louvain_communities(
                graph,
                weight="weight",
                resolution=resolution,
                seed=int(seed + offset),
            )
        ]
        score = float(
            nx.community.modularity(graph, groups, weight="weight", resolution=resolution)
        )
        signature = tuple(sorted(tuple(sorted(group)) for group in groups))
        if score > best_score + epsilon or (
            abs(score - best_score) <= epsilon
            and (best_signature is None or signature < best_signature)
        ):
            labels = np.full(n_nodes, -1, dtype=np.int64)
            for group_index, group in enumerate(sorted(groups, key=lambda value: min(value))):
                labels[list(group)] = group_index
            best_labels = relabel_consecutive(labels)
            best_score = score
            best_signature = signature
    if best_labels is None:
        raise RuntimeError("Markov-stability optimizer produced no partition")
    return best_labels, best_score


def persistent_markov_time(
    n_nodes: int,
    edges: list[MicroEdge],
    base_resolution: float,
    seed: int,
    minimum_clusters: int = 1,
    node_masses: NDArray[np.float64] | None = None,
    lineage_persistence: bool = True,
) -> tuple[float, dict[str, object]]:
    """Select a mass-weighted density-flow partition lineage over Markov time.

    Times are doubled on a finite grid.  Exact partition repetition certifies a plateau.  Equal
    cluster counts with different memberships continue only while at least two candidate fragments
    remain per disconnected graph support.  Near that support floor, the earlier partition is
    retained.  The disconnected limit and the speciation lower bound remain guarded.
    """

    pairs = np.asarray(
        [(edge.left, edge.right) for edge in edges if edge.weight > 0], dtype=np.int64
    )
    if pairs.size == 0:
        return 1.0, {"mode": "persistent-Markov-time", "status": "no-positive-edges"}
    weights = np.asarray([edge.weight for edge in edges if edge.weight > 0], dtype=np.float64)
    graph = _weighted_graph(n_nodes, pairs, weights)
    connected = nx.number_connected_components(graph)
    maximum_time = 2 ** int(np.ceil(np.log2(max(n_nodes, 2))))
    records: list[dict[str, object]] = []
    times: list[float] = []
    partitions: list[NDArray[np.int64]] = []
    stabilities: list[float] = []
    time_value = 1.0
    while time_value <= maximum_time:
        labels, stability = _best_markov_stability_partition(
            graph,
            seed,
            resolution=base_resolution / time_value,
        )
        count = int(np.unique(labels).size)
        times.append(time_value)
        partitions.append(labels)
        stabilities.append(float(stability))
        records.append(
            {
                "time": time_value,
                "clusters": count,
                "stability": float(stability),
                "partition_sha256": _partition_signature(labels),
            }
        )
        if (
            len(records) == 1
            and minimum_clusters > 1
            and count <= 2 * minimum_clusters
        ):
            return 1.0, {
                "mode": "persistent-Markov-time",
                "status": "not-fragmented-at-one-step",
                "connected_components": connected,
                "selected_time": 1.0,
                "minimum_clusters": minimum_clusters,
                "one_step_clusters": count,
                "fragmentation_certified": False,
                "speciation_floor_applied": False,
                "records": records,
            }
        if len(partitions) >= 2:
            previous_count = int(np.unique(partitions[-2]).size)
            if previous_count == count:
                similarity = mass_weighted_partition_similarity(
                    partitions[-2], partitions[-1], node_masses
                )
                if (
                    not lineage_persistence
                    or similarity >= 1.0 - 1e-12
                    or count < 2 * connected
                ):
                    break
        time_value *= 2.0
    _, _, similarities, same_count_drift, interval_records = (
        _select_fragmentation_guarded_lineage_index(
            partitions,
            stabilities,
            connected,
            node_masses,
        )
    )
    if lineage_persistence:
        selected_index, status, _, _, _ = _select_fragmentation_guarded_lineage_index(
            partitions,
            stabilities,
            connected,
            node_masses,
        )
    else:
        selected_index, status = _select_count_plateau_index(
            partitions, stabilities, connected
        )
    for index, similarity in enumerate(similarities):
        records[index]["lineage_similarity_to_next"] = float(similarity)
    selected = times[selected_index]
    one_step_clusters = int(records[0]["clusters"]) if records else 1
    fragmentation_certified = bool(
        minimum_clusters <= 1 or one_step_clusters > 2 * minimum_clusters
    )
    if not fragmentation_certified:
        selected_index = 0
        selected = times[selected_index]
    selected_count = next(
        (int(record["clusters"]) for record in records if record["time"] == selected),
        1,
    )
    floor_applied = False
    if selected_count < minimum_clusters:
        eligible_indices = [
            index
            for index, record in enumerate(records)
            if int(record["clusters"]) >= minimum_clusters
        ]
        if eligible_indices:
            selected_index = max(eligible_indices)
            selected = times[selected_index]
            floor_applied = True
    return selected, {
        "mode": "persistent-Markov-time",
        "status": status,
        "connected_components": connected,
        "selected_time": selected,
        "selected_partition_sha256": records[selected_index]["partition_sha256"],
        "selected_lineage_similarity": (
            float(similarities[selected_index])
            if selected_index < len(similarities)
            else 1.0
        ),
        "same_count_partition_drift_events": same_count_drift,
        "lineage_conflict_intervals": interval_records,
        "lineage_persistence_enabled": lineage_persistence,
        "minimum_clusters": minimum_clusters,
        "one_step_clusters": one_step_clusters,
        "fragmentation_certified": fragmentation_certified,
        "speciation_floor_applied": floor_applied,
        "records": records,
    }


def persistent_point_manifold_partition(
    routing_neighbors: list[set[int]],
    distance_lookup: list[dict[int, float]],
    radius: NDArray[np.float64],
    micro_labels: NDArray[np.int64],
    seed: int,
    lineage_persistence: bool = True,
    epsilon: float = 1e-12,
) -> tuple[NDArray[np.int64], dict[str, object]]:
    """Find a persistent low-dimensional point-graph partition and map it to micro-components."""

    n = len(routing_neighbors)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i, row in enumerate(routing_neighbors):
        for j in sorted(row):
            if i >= j:
                continue
            distance = distance_lookup[i].get(j, distance_lookup[j].get(i))
            if distance is None:
                continue
            weight = float(
                np.exp(-(distance**2) / max(radius[i] * radius[j], epsilon))
            )
            graph.add_edge(i, j, weight=weight)
    if graph.number_of_edges() == 0:
        return np.arange(int(micro_labels.max()) + 1, dtype=np.int64), {
            "mode": "persistent-point-manifold",
            "status": "no-positive-edges",
        }
    connected = nx.number_connected_components(graph)
    maximum_time = 2 ** int(np.ceil(np.log2(max(n, 2))))
    times: list[float] = []
    partitions: list[NDArray[np.int64]] = []
    stabilities: list[float] = []
    records: list[dict[str, object]] = []
    time_value = 1.0
    while time_value <= maximum_time:
        labels, stability = _best_markov_stability_partition(
            graph, seed, resolution=1.0 / time_value
        )
        count = int(np.unique(labels).size)
        times.append(time_value)
        partitions.append(labels)
        stabilities.append(float(stability))
        records.append(
            {
                "time": time_value,
                "clusters": count,
                "stability": float(stability),
                "partition_sha256": _partition_signature(labels),
            }
        )
        if len(partitions) >= 2:
            previous_count = int(np.unique(partitions[-2]).size)
            if previous_count == count:
                similarity = mass_weighted_partition_similarity(
                    partitions[-2], partitions[-1], None, epsilon
                )
                if (
                    not lineage_persistence
                    or similarity >= 1.0 - epsilon
                    or count < 2 * connected
                ):
                    break
        time_value *= 2.0
    _, _, similarities, same_count_drift, interval_records = (
        _select_fragmentation_guarded_lineage_index(
            partitions,
            stabilities,
            connected,
            np.ones(n, dtype=np.float64),
            epsilon,
        )
    )
    if lineage_persistence:
        selected_index, status, _, _, _ = _select_fragmentation_guarded_lineage_index(
            partitions,
            stabilities,
            connected,
            np.ones(n, dtype=np.float64),
            epsilon,
        )
    else:
        selected_index, status = _select_count_plateau_index(
            partitions, stabilities, connected, epsilon
        )
    for index, similarity in enumerate(similarities):
        records[index]["lineage_similarity_to_next"] = float(similarity)
    point_labels = partitions[selected_index]
    n_micro = int(micro_labels.max()) + 1
    macro_by_micro = np.full(n_micro, -1, dtype=np.int64)
    split_microcomponents = 0
    for micro in range(n_micro):
        values, counts_local = np.unique(
            point_labels[micro_labels == micro], return_counts=True
        )
        if values.size > 1:
            split_microcomponents += 1
        maximum = int(np.max(counts_local))
        macro_by_micro[micro] = int(np.min(values[counts_local == maximum]))
    macro_by_micro = relabel_consecutive(macro_by_micro)
    return macro_by_micro, {
        "mode": "persistent-point-manifold",
        "status": status,
        "connected_components": connected,
        "selected_time": times[selected_index],
        "selected_partition_sha256": records[selected_index]["partition_sha256"],
        "selected_lineage_similarity": (
            float(similarities[selected_index])
            if selected_index < len(similarities)
            else 1.0
        ),
        "same_count_partition_drift_events": same_count_drift,
        "lineage_conflict_intervals": interval_records,
        "lineage_persistence_enabled": lineage_persistence,
        "selected_point_clusters": int(np.unique(point_labels).size),
        "selected_micro_clusters": int(np.unique(macro_by_micro).size),
        "split_microcomponents": split_microcomponents,
        "records": records,
    }


def markov_correct_partition(
    preliminary_labels: NDArray[np.int64],
    n_nodes: int,
    edges: list[MicroEdge],
    alpha: float,
    permutations: int,
    swaps_per_edge: int,
    seed: int,
    resolution: float = 1.0,
    hubness_consolidation: bool = False,
    minimum_clusters: int = 1,
    sequential_null: bool = True,
    epsilon: float = 1e-12,
) -> tuple[NDArray[np.int64], list[MarkovCorrectionStep], dict[str, object]]:
    """Apply one statistically calibrated Markov-stability correction on the BGF graph.

    The reciprocal density-flow topology supplies the preliminary macro partition.  Louvain is
    only an optimizer for the one-step Markov-stability objective on the BGF graph.  Replacement
    occurs when its stability improvement exceeds a degree-preserving max-null at ``alpha``;
    neither labels nor a requested number of clusters enter the decision.
    """

    preliminary = relabel_consecutive(np.asarray(preliminary_labels, dtype=np.int64))
    pairs = np.asarray(
        [(edge.left, edge.right) for edge in edges if edge.weight > 0], dtype=np.int64
    )
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    weights = np.asarray([edge.weight for edge in edges if edge.weight > 0], dtype=np.float64)
    graph = _weighted_graph(n_nodes, pairs, weights)
    if graph.number_of_edges() == 0 or graph.size(weight="weight") <= 0:
        return preliminary, [], {
            "mode": "null-calibrated-BGF-Markov-correction",
            "status": "no-positive-BGF-edges",
            "corrected": False,
        }

    preliminary_groups = _labels_to_groups(preliminary)
    preliminary_score = float(
        nx.community.modularity(
            graph, preliminary_groups, weight="weight", resolution=resolution
        )
    )
    candidate, candidate_score = _best_markov_stability_partition(
        graph, seed, resolution=resolution, epsilon=epsilon
    )
    observed = max(candidate_score - preliminary_score, 0.0)
    negative_preliminary_stability = preliminary_score < -epsilon
    hubness_consolidated = bool(
        hubness_consolidation
        and observed > epsilon
        and np.unique(candidate).size < np.unique(preliminary).size
    )
    requires_null = bool(
        observed > epsilon
        and not negative_preliminary_stability
        and not hubness_consolidated
    )
    rng = np.random.default_rng(seed)
    null_improvements: list[float] = []
    degenerates = 0
    fallbacks = 0
    successful_swaps = 0
    requested_swaps = 0
    null_decision: str | None = None
    confidence_lower = 0.0
    confidence_upper = 1.0
    minimum_trials = max(int(np.ceil(1.0 / alpha)) - 1, 1)
    checkpoints: list[int] = []
    checkpoint = minimum_trials
    while checkpoint < permutations:
        checkpoints.append(checkpoint)
        checkpoint = 2 * checkpoint + 1
    if permutations > 0:
        checkpoints.append(permutations)
    checkpoint_set = set(checkpoints)
    risk_per_checkpoint = alpha / max(len(checkpoints), 1)
    if requires_null:
        for permutation in range(permutations):
            (
                null_pairs,
                null_weights,
                successful,
                requested,
                degenerate,
                fallback,
            ) = _rewired_edge_list(n_nodes, pairs, weights, rng, swaps_per_edge)
            null_graph = _weighted_graph(n_nodes, null_pairs, null_weights)
            if null_graph.number_of_edges() == 0 or null_graph.size(weight="weight") <= 0:
                null_improvements.append(0.0)
            else:
                null_preliminary = float(
                    nx.community.modularity(
                        null_graph,
                        preliminary_groups,
                        weight="weight",
                        resolution=resolution,
                    )
                )
                _, null_candidate = _best_markov_stability_partition(
                    null_graph,
                    seed + 1009 * (permutation + 1),
                    resolution=resolution,
                    epsilon=epsilon,
                )
                null_improvements.append(max(null_candidate - null_preliminary, 0.0))
            degenerates += int(degenerate)
            fallbacks += int(fallback)
            successful_swaps += successful
            requested_swaps += requested
            trials = permutation + 1
            if sequential_null and trials in checkpoint_set:
                exceedances_now = int(
                    np.count_nonzero(np.asarray(null_improvements) >= observed)
                )
                confidence_lower = (
                    0.0
                    if exceedances_now == 0
                    else float(
                        beta_distribution.ppf(
                            risk_per_checkpoint / 2.0,
                            exceedances_now,
                            trials - exceedances_now + 1,
                        )
                    )
                )
                confidence_upper = (
                    1.0
                    if exceedances_now == trials
                    else float(
                        beta_distribution.ppf(
                            1.0 - risk_per_checkpoint / 2.0,
                            exceedances_now + 1,
                            trials - exceedances_now,
                        )
                    )
                )
                if confidence_upper < alpha:
                    null_decision = "significant"
                    break
                if confidence_lower > alpha:
                    null_decision = "nonsignificant"
                    break
    null_array = np.asarray(null_improvements, dtype=np.float64)
    exceedances = int(np.count_nonzero(null_array >= observed)) if null_array.size else 0
    p_value = (
        (1 + exceedances) / (null_array.size + 1)
        if requires_null
        else 0.0
        if observed > epsilon
        else 1.0
    )
    corrected = bool(
        observed > epsilon
        and (
            negative_preliminary_stability
            or hubness_consolidated
            or null_decision == "significant"
            or (null_decision is None and p_value <= alpha)
        )
    )
    selected = candidate if corrected else preliminary
    forced_nontrivial = bool(
        np.unique(selected).size < minimum_clusters
        and np.unique(candidate).size >= minimum_clusters
    )
    if forced_nontrivial:
        selected = candidate
        corrected = True
    step = MarkovCorrectionStep(
        preliminary_clusters=int(np.unique(preliminary).size),
        candidate_clusters=int(np.unique(candidate).size),
        preliminary_stability=preliminary_score,
        candidate_stability=candidate_score,
        observed_improvement=observed,
        p_value=float(p_value),
        corrected=corrected,
        null_improvement_median=float(np.median(null_array)) if null_array.size else 0.0,
        null_improvement_q95=float(np.quantile(null_array, 0.95)) if null_array.size else 0.0,
        degenerate_nulls=degenerates,
        multigraph_fallbacks=fallbacks,
    )
    return relabel_consecutive(selected), [step], {
        "mode": "null-calibrated-BGF-Markov-correction",
        "status": (
            "speciation-forced-nontrivial-candidate"
            if forced_nontrivial
            else "negative-stability-corrected"
            if corrected and negative_preliminary_stability
            else "hubness-consolidated"
            if corrected and hubness_consolidated
            else "null-significant-corrected"
            if corrected
            else "preliminary-retained"
        ),
        "corrected": corrected,
        "minimum_clusters": minimum_clusters,
        "forced_nontrivial": forced_nontrivial,
        "negative_preliminary_stability": negative_preliminary_stability,
        "hubness_consolidation_enabled": hubness_consolidation,
        "hubness_consolidated": hubness_consolidated,
        "preliminary_clusters": step.preliminary_clusters,
        "candidate_clusters": step.candidate_clusters,
        "selected_clusters": int(np.unique(selected).size),
        "preliminary_stability": preliminary_score,
        "candidate_stability": candidate_score,
        "observed_improvement": observed,
        "p_value": float(p_value),
        "null_required": requires_null,
        "null_decision": null_decision,
        "null_trials_performed": len(null_improvements),
        "null_trials_maximum": permutations,
        "null_confidence_lower": confidence_lower,
        "null_confidence_upper": confidence_upper,
        "resampling_risk_bound": alpha,
        "sequential_null": sequential_null,
        "null_skipped_reason": (
            "negative-preliminary-stability"
            if negative_preliminary_stability
            else "hubness-consolidation"
            if hubness_consolidated
            else "no-positive-improvement"
            if observed <= epsilon
            else None
        ),
        "resolution": float(resolution),
        "null_permutations": permutations,
        "successful_swaps": successful_swaps,
        "requested_swaps": requested_swaps,
        "degenerate_nulls": degenerates,
        "multigraph_fallbacks": fallbacks,
    }


def evidence_gap_components(
    n_nodes: int,
    edges: list[MicroEdge],
    micro_sizes: NDArray[np.int64],
    directional_reliability: NDArray[np.float64],
    total_points: int,
    near_global_coverage: float,
    epsilon: float = 1e-12,
) -> tuple[NDArray[np.int64], dict[str, object]]:
    """Fuse BGF components, resolving a near-global component by its strongest MST gap.

    The r2 rule retained every basin whenever one positive-edge component covered at least
    ``near_global_coverage`` of the observations, so BGF magnitudes had no operational role.
    Here a maximum spanning tree retains the strongest evidence paths.  The largest adjacent
    gap in its sorted edge weights defines a parameter-free cut; if no positive gap exists,
    the original basins are conservatively retained.
    """

    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_weighted_edges_from(
        (edge.left, edge.right, edge.weight) for edge in edges if edge.weight > 0
    )
    groups: list[set[int]] = []
    gap_records: list[dict[str, object]] = []
    accepted_components = 0
    unresolved_components = 0
    graph_components = list(nx.connected_components(graph))
    for component_index, nodes_raw in enumerate(graph_components):
        nodes = set(map(int, nodes_raw))
        mass = float(np.sum(micro_sizes[list(nodes)])) / max(total_points, 1)
        if len(nodes) <= 1:
            groups.append(nodes)
            continue
        if mass < near_global_coverage:
            groups.append(nodes)
            accepted_components += 1
            continue

        component_reliability = np.asarray(
            directional_reliability[list(nodes)], dtype=np.float64
        )
        median_reliability = float(np.median(component_reliability))
        reliable_nodes = int(np.count_nonzero(component_reliability > 0))
        if 2 * reliable_nodes <= len(nodes):
            groups.extend({node} for node in sorted(nodes))
            unresolved_components += 1
            gap_records.append(
                {
                    "component": component_index,
                    "mass": mass,
                    "status": "directional-majority-unreliable",
                    "n_nodes": len(nodes),
                    "median_directional_reliability": median_reliability,
                    "reliable_nodes": reliable_nodes,
                }
            )
            continue

        tree = nx.maximum_spanning_tree(graph.subgraph(nodes), weight="weight")
        tree_edges = sorted(
            (
                (min(int(left), int(right)), max(int(left), int(right)), float(data["weight"]))
                for left, right, data in tree.edges(data=True)
            ),
            key=lambda item: (item[2], item[0], item[1]),
        )
        weights = np.asarray([item[2] for item in tree_edges], dtype=np.float64)
        if weights.size < 2:
            groups.extend({node} for node in sorted(nodes))
            unresolved_components += 1
            gap_records.append(
                {
                    "component": component_index,
                    "mass": mass,
                    "status": "insufficient-tree-edges",
                    "n_nodes": len(nodes),
                }
            )
            continue
        gaps = np.diff(weights)
        gap_index = int(np.argmax(gaps))
        maximum_gap = float(gaps[gap_index])
        if maximum_gap <= epsilon:
            groups.extend({node} for node in sorted(nodes))
            unresolved_components += 1
            gap_records.append(
                {
                    "component": component_index,
                    "mass": mass,
                    "status": "no-positive-evidence-gap",
                    "n_nodes": len(nodes),
                    "maximum_gap": maximum_gap,
                }
            )
            continue

        cut_count = gap_index + 1
        tree.remove_edges_from((left, right) for left, right, _ in tree_edges[:cut_count])
        resolved = [set(map(int, group)) for group in nx.connected_components(tree)]
        groups.extend(resolved)
        gap_records.append(
            {
                "component": component_index,
                "mass": mass,
                "status": "evidence-gap-cut",
                "n_nodes": len(nodes),
                "median_directional_reliability": median_reliability,
                "maximum_gap": maximum_gap,
                "lower_weight": float(weights[gap_index]),
                "upper_weight": float(weights[gap_index + 1]),
                "cut_edges": cut_count,
                "result_groups": len(resolved),
            }
        )

    labels = np.full(n_nodes, -1, dtype=np.int64)
    for group_index, group in enumerate(sorted(groups, key=lambda value: min(value))):
        labels[list(group)] = group_index
    if np.any(labels < 0):
        raise RuntimeError("evidence-gap fusion did not assign every micro-component")
    return relabel_consecutive(labels), {
        "mode": "reliability-aware-BGF-evidence-gap",
        "graph_connected_components": len(graph_components),
        "accepted_subglobal_components": accepted_components,
        "unresolved_near_global_components": unresolved_components,
        "gap_records": gap_records,
    }


def automatic_fusion_regime(
    intrinsic_dimension: float,
    low_id_cutoff: float,
    low_id_partition_weak: bool | None,
) -> tuple[str, dict[str, object]]:
    """Select a label-free fusion regime from reciprocal-partition stability.

    Higher-dimensional data always use the null-calibrated correction.  A
    low-dimensional reciprocal partition receives the Markov-stability rescue only
    when its own weighted stability is unresolved.  Thus the automatic rule never
    preserves fragmentation merely because many microcomponents were discovered.

    The rule is fixed before evaluation and uses neither labels nor a requested K.
    """

    if not np.isfinite(intrinsic_dimension) or intrinsic_dimension <= 0:
        raise ValueError("intrinsic_dimension must be finite and positive")
    if low_id_cutoff <= 0:
        raise ValueError("low_id_cutoff must be positive")
    if intrinsic_dimension <= low_id_cutoff and low_id_partition_weak is None:
        raise ValueError("low-ID automatic fusion requires a partition-stability assessment")
    if intrinsic_dimension <= low_id_cutoff and low_id_partition_weak:
        selected = "stability"
        reason = "low-ID-weak-reciprocal-partition"
    elif intrinsic_dimension <= low_id_cutoff:
        selected = "correction"
        reason = "low-ID-resolved-reciprocal-partition"
    else:
        selected = "correction"
        reason = "high-ID-null-calibrated-correction"
    return selected, {
        "mode": "label-free-automatic-fusion-regime",
        "selected": selected,
        "reason": reason,
        "intrinsic_dimension": float(intrinsic_dimension),
        "low_id_cutoff": float(low_id_cutoff),
        "low_id_partition_weak": low_id_partition_weak,
    }


def partition_stability_assessment(
    preliminary_labels: NDArray[np.int64],
    n_nodes: int,
    edges: list[MicroEdge],
    intrinsic_dimension: float,
    epsilon: float = 1e-12,
) -> dict[str, object]:
    """Assess whether a preliminary BGF partition has resolvable positive stability."""

    preliminary = relabel_consecutive(np.asarray(preliminary_labels, dtype=np.int64))
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_weighted_edges_from(
        (edge.left, edge.right, float(edge.weight))
        for edge in edges
        if np.isfinite(edge.weight) and edge.weight > 0
    )
    q = int(max(1, np.rint(intrinsic_dimension)))
    minimum_graph_nodes = 2 * (q + 2)
    finite_resolution_floor = 2.0 / max(n_nodes, 1)
    if graph.number_of_edges() == 0:
        return {
            "status": "no-positive-finite-BGF-edges",
            "weak": False,
            "preliminary_modularity": None,
            "finite_resolution_floor": finite_resolution_floor,
            "minimum_graph_nodes": minimum_graph_nodes,
        }
    groups = [
        set(np.flatnonzero(preliminary == group).tolist())
        for group in range(int(preliminary.max()) + 1)
    ]
    preliminary_modularity = float(
        nx.community.modularity(graph, groups, weight="weight", resolution=1.0)
    )
    weak = preliminary_modularity < -epsilon or (
        preliminary_modularity <= finite_resolution_floor + epsilon
        and n_nodes >= minimum_graph_nodes
    )
    return {
        "status": "weak-preliminary" if weak else "preliminary-stability-resolved",
        "weak": bool(weak),
        "preliminary_modularity": preliminary_modularity,
        "finite_resolution_floor": finite_resolution_floor,
        "minimum_graph_nodes": minimum_graph_nodes,
    }


def markov_stability_guard(
    preliminary_labels: NDArray[np.int64],
    n_nodes: int,
    edges: list[MicroEdge],
    micro_sizes: NDArray[np.int64],
    intrinsic_dimension: float,
    seed: int,
    epsilon: float = 1e-12,
) -> tuple[NDArray[np.int64], dict[str, object]]:
    """Replace a weak preliminary partition by a BGF Markov-stability partition.

    Weighted modularity on an undirected graph is the one-step Markov-stability objective.
    The preliminary partition is retained whenever it has resolvable positive stability.
    Otherwise deterministic multi-start Louvain optimizes the same BGF graph at a resolution
    determined from intrinsic dimension and the dominant density-flow basin mass.  No labels
    or target cluster count enter the decision.
    """
    preliminary_labels = relabel_consecutive(preliminary_labels)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_weighted_edges_from(
        (edge.left, edge.right, float(edge.weight))
        for edge in edges
        if np.isfinite(edge.weight) and edge.weight > 0
    )
    if graph.number_of_edges() == 0:
        return preliminary_labels, {
            "mode": "markov-stability-guard",
            "status": "no-positive-finite-BGF-edges",
            "preliminary_modularity": None,
            "replaced": False,
        }

    preliminary_groups = [
        set(np.flatnonzero(preliminary_labels == group).tolist())
        for group in range(int(preliminary_labels.max()) + 1)
    ]
    preliminary_modularity = float(
        nx.community.modularity(graph, preliminary_groups, weight="weight", resolution=1.0)
    )
    q = int(max(1, np.rint(intrinsic_dimension)))
    minimum_graph_nodes = 2 * (q + 2)
    finite_resolution_floor = 2.0 / max(n_nodes, 1)
    weak = preliminary_modularity < -epsilon or (
        preliminary_modularity <= finite_resolution_floor + epsilon
        and n_nodes >= minimum_graph_nodes
    )
    if not weak:
        return preliminary_labels, {
            "mode": "markov-stability-guard",
            "status": "preliminary-stability-resolved",
            "preliminary_modularity": preliminary_modularity,
            "finite_resolution_floor": finite_resolution_floor,
            "minimum_graph_nodes": minimum_graph_nodes,
            "replaced": False,
        }

    dominant_basin_mass = float(np.max(micro_sizes)) / max(float(np.sum(micro_sizes)), 1.0)
    resolution = float(np.clip(intrinsic_dimension * dominant_basin_mass, 0.5, 3.0))
    best_groups: list[set[int]] | None = None
    best_modularity = -np.inf
    best_signature: tuple[tuple[int, ...], ...] | None = None
    starts = 8
    for offset in range(starts):
        groups = [
            set(map(int, group))
            for group in nx.community.louvain_communities(
                graph,
                weight="weight",
                resolution=resolution,
                seed=int(seed + offset),
            )
        ]
        modularity = float(
            nx.community.modularity(graph, groups, weight="weight", resolution=resolution)
        )
        signature = tuple(sorted(tuple(sorted(group)) for group in groups))
        if modularity > best_modularity + epsilon or (
            abs(modularity - best_modularity) <= epsilon
            and (best_signature is None or signature < best_signature)
        ):
            best_groups = groups
            best_modularity = modularity
            best_signature = signature
    if best_groups is None:
        raise RuntimeError("Markov-stability optimization produced no partition")
    labels = np.full(n_nodes, -1, dtype=np.int64)
    for group_index, group in enumerate(sorted(best_groups, key=lambda value: min(value))):
        labels[list(group)] = group_index
    if np.any(labels < 0):
        raise RuntimeError("Markov-stability partition did not assign every micro-component")
    return relabel_consecutive(labels), {
        "mode": "markov-stability-guard",
        "status": "weak-preliminary-replaced",
        "preliminary_modularity": preliminary_modularity,
        "finite_resolution_floor": finite_resolution_floor,
        "minimum_graph_nodes": minimum_graph_nodes,
        "dominant_basin_mass": dominant_basin_mass,
        "resolution": resolution,
        "starts": starts,
        "selected_modularity": best_modularity,
        "preliminary_clusters": int(np.unique(preliminary_labels).size),
        "selected_clusters": int(np.unique(labels).size),
        "replaced": True,
    }


def fuse_components(
    n_nodes: int,
    edges: list[MicroEdge],
    alpha: float,
    permutations: int,
    swaps_per_edge: int,
    seed: int,
) -> tuple[NDArray[np.int64], list[FusionStep], dict[str, object]]:
    all_pairs = np.asarray([(edge.left, edge.right) for edge in edges], dtype=np.int64)
    if all_pairs.size == 0:
        all_pairs = np.empty((0, 2), dtype=np.int64)
    all_weights = np.asarray([edge.weight for edge in edges], dtype=np.float64)
    edge_pairs = [(edge.left, edge.right) for edge in edges if edge.weight > 0]
    graph_components = connected_components(range(n_nodes), edge_pairs)
    rng = np.random.default_rng(seed)
    all_groups: list[set[int]] = []
    history: list[FusionStep] = []
    null_summaries: list[dict[str, object]] = []

    for component_index, nodes in enumerate(graph_components):
        if all_pairs.size:
            mask = np.isin(all_pairs[:, 0], list(nodes)) & np.isin(all_pairs[:, 1], list(nodes))
            component_pairs = all_pairs[mask]
            component_weights = all_weights[mask]
        else:
            component_pairs = np.empty((0, 2), dtype=np.int64)
            component_weights = np.empty(0, dtype=np.float64)
        groups = [{node} for node in sorted(nodes)]
        iteration = 0
        while len(groups) > 1:
            best = _best_pair_edges(n_nodes, component_pairs, component_weights, groups)
            if best is None:
                break
            left_index, right_index, observed = best
            null_maxima: list[float] = []
            swaps = 0
            requested = 0
            degenerates = 0
            fallbacks = 0
            for _ in range(permutations):
                (
                    null_pairs,
                    null_weights,
                    success,
                    request,
                    degenerate,
                    fallback,
                ) = _rewired_edge_list(
                    n_nodes, component_pairs, component_weights, rng, swaps_per_edge
                )
                null_best = _best_pair_edges(n_nodes, null_pairs, null_weights, groups)
                null_maxima.append(0.0 if null_best is None else null_best[2])
                swaps += success
                requested += request
                degenerates += int(degenerate)
                fallbacks += int(fallback)
            null_array = np.asarray(null_maxima, dtype=np.float64)
            exceed = int(np.count_nonzero(null_array >= observed))
            p_value = (1 + exceed) / (permutations + 1)
            merge = p_value <= alpha
            left_group = tuple(sorted(groups[left_index]))
            right_group = tuple(sorted(groups[right_index]))
            history.append(
                FusionStep(
                    connected_component=component_index,
                    iteration=iteration,
                    left=left_group,
                    right=right_group,
                    observed=float(observed),
                    p_value=float(p_value),
                    merged=merge,
                    null_max_median=float(np.median(null_array)) if null_array.size else 0.0,
                    null_max_q95=float(np.quantile(null_array, 0.95)) if null_array.size else 0.0,
                    successful_swaps=swaps,
                    requested_swaps=requested,
                    degenerate_nulls=degenerates,
                    multigraph_fallbacks=fallbacks,
                )
            )
            null_summaries.append(
                {
                    "component": component_index,
                    "iteration": iteration,
                    "minimum": float(null_array.min()) if null_array.size else 0.0,
                    "median": float(np.median(null_array)) if null_array.size else 0.0,
                    "q95": float(np.quantile(null_array, 0.95)) if null_array.size else 0.0,
                    "maximum": float(null_array.max()) if null_array.size else 0.0,
                    "p_value": float(p_value),
                }
            )
            if not merge:
                break
            merged = groups[left_index] | groups[right_index]
            groups = [group for idx, group in enumerate(groups) if idx not in {left_index, right_index}]
            groups.append(merged)
            groups.sort(key=lambda group: min(group))
            iteration += 1
        all_groups.extend(groups)

    micro_labels = np.full(n_nodes, -1, dtype=np.int64)
    for group_index, group in enumerate(sorted(all_groups, key=lambda value: min(value))):
        micro_labels[list(group)] = group_index
    return relabel_consecutive(micro_labels), history, {
        "graph_connected_components": len(graph_components),
        "null_summaries": null_summaries,
    }

# ============================================================================
# Original module: aedpcv2/recurrent.py
# ============================================================================

"""Recurrent Ellipsoidal Absorption for macro-state legitimacy.

This module is part of AEDPCv2's own density-flow -> ellipsoid -> Markov line. It does not
estimate a target cluster count and does not call an external clustering algorithm.
"""


import hashlib

import numpy as np
from numpy.typing import NDArray



def certified_floor_is_geometrically_underidentified(
    macro_by_micro: NDArray[np.int64],
    micro_labels: NDArray[np.int64],
    recurrence_diagnostics: dict[str, object],
    minimum_cluster_points: int,
) -> tuple[bool, dict[str, object]]:
    """Detect a certified state floor supported only by a tiny promoted state.

    The recurrence null may be unable to identify as many states as an independent
    ellipsoidal-speciation test certifies.  Promotion enforces that lower bound, but a promoted
    state smaller than the covariance-identifiability support ``q + 2`` is not a geometrically
    admissible macro-state.  The condition is fully label-free and introduces no new threshold.
    """

    macro = relabel_consecutive(np.asarray(macro_by_micro, dtype=np.int64))
    micro = np.asarray(micro_labels, dtype=np.int64)
    if macro.shape != (int(micro.max()) + 1,):
        raise ValueError("macro mapping and micro labels must agree")
    if minimum_cluster_points < 1:
        raise ValueError("minimum_cluster_points must be positive")
    null = recurrence_diagnostics.get("recurrence_null", {})
    floor_applied = bool(
        isinstance(null, dict) and null.get("minimum_state_floor_applied", False)
    )
    point_labels = macro[micro]
    sizes = np.bincount(point_labels, minlength=int(point_labels.max()) + 1)
    minimum = int(np.min(sizes))
    underidentified = bool(floor_applied and minimum < minimum_cluster_points)
    return underidentified, {
        "floor_applied": floor_applied,
        "minimum_cluster_points": minimum,
        "geometric_support_floor": int(minimum_cluster_points),
        "initial_clusters": int(sizes.size),
        "underidentified": underidentified,
    }


def coarsen_to_certified_state_floor(
    macro_by_micro: NDArray[np.int64],
    micro_sizes: NDArray[np.int64],
    edges: list[MicroEdge],
    target_states: int,
    epsilon: float = 1e-12,
) -> tuple[NDArray[np.int64], list[dict[str, float | int]]]:
    """Coarsen an exceptional fallback by maximum stationary set flow.

    This operation is used only after the primary recurrence correction has been rejected as
    geometrically underidentified.  It returns the coarsest connected fallback consistent with an
    independently certified state lower bound.  The analytic score is the lazy-walk stationary
    set flow ``W(U,V)/(2*sqrt(vol(U)*vol(V)))``; no requested K or external label is used.
    """

    labels = relabel_consecutive(np.asarray(macro_by_micro, dtype=np.int64))
    sizes = np.asarray(micro_sizes, dtype=np.int64)
    if labels.shape != sizes.shape:
        raise ValueError("macro mapping and micro sizes must agree")
    if target_states < 1:
        raise ValueError("target_states must be positive")
    target = min(int(target_states), int(np.unique(labels).size))
    degree = np.zeros(labels.size, dtype=np.float64)
    for edge in edges:
        if np.isfinite(edge.weight) and edge.weight > 0:
            degree[edge.left] += edge.weight
            degree[edge.right] += edge.weight
    history: list[dict[str, float | int]] = []
    while np.unique(labels).size > target:
        states = np.unique(labels)
        volume = {
            int(state): float(np.sum(degree[labels == state])) for state in states
        }
        candidates: list[tuple[float, float, int, int]] = []
        for left_position, left in enumerate(states[:-1]):
            for right in states[left_position + 1 :]:
                cross = float(
                    sum(
                        edge.weight
                        for edge in edges
                        if edge.weight > 0
                        and (
                            (labels[edge.left] == left and labels[edge.right] == right)
                            or (labels[edge.left] == right and labels[edge.right] == left)
                        )
                    )
                )
                if cross <= epsilon:
                    continue
                denominator = 2.0 * np.sqrt(
                    max(volume[int(left)] * volume[int(right)], epsilon)
                )
                score = cross / denominator
                candidates.append((float(score), cross, int(left), int(right)))
        if not candidates:
            break
        score, cross, left, right = max(
            candidates,
            key=lambda item: (item[0], item[1], -item[2], -item[3]),
        )
        labels[labels == right] = left
        labels = relabel_consecutive(labels)
        history.append(
            {
                "left": left,
                "right": right,
                "stationary_set_flow": score,
                "cross_weight": cross,
                "remaining_states": int(np.unique(labels).size),
            }
        )
    return labels, history




def null_calibrated_recurrent_states(
    macro_by_micro: NDArray[np.int64],
    micro_sizes: NDArray[np.int64],
    edges: list[MicroEdge],
    alpha: float,
    permutations: int,
    seed: int,
    epsilon: float = 1e-12,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    dict[str, object],
]:
    """Test conditional state retention under a graph-determined label-permutation null."""

    del seed
    macro = relabel_consecutive(np.asarray(macro_by_micro, dtype=np.int64))
    sizes = np.asarray(micro_sizes, dtype=np.float64)
    n_micro = macro.size
    n_macro = int(macro.max()) + 1
    diagonal = (sizes - 1.0) / sizes
    valid_edges = [
        edge for edge in edges if np.isfinite(edge.weight) and edge.weight > 0
    ]
    left = np.asarray([edge.left for edge in valid_edges], dtype=np.int64)
    right = np.asarray([edge.right for edge in valid_edges], dtype=np.int64)
    weight = np.asarray([edge.weight for edge in valid_edges], dtype=np.float64)
    degree = diagonal.copy()
    if weight.size:
        np.add.at(degree, left, weight)
        np.add.at(degree, right, weight)
    total = float(np.sum(degree))

    def statistics(
        labels: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        volume = np.bincount(labels, weights=degree, minlength=n_macro)
        internal = np.bincount(labels, weights=diagonal, minlength=n_macro)
        if weight.size:
            same = labels[left] == labels[right]
            np.add.at(internal, labels[left[same]], 2.0 * weight[same])
        retention = np.divide(
            internal,
            volume,
            out=np.zeros_like(internal),
            where=volume > epsilon,
        )
        contribution = (
            internal / total - (volume / total) ** 2
            if total > epsilon
            else np.zeros(n_macro, dtype=np.float64)
        )
        closed = (volume > epsilon) & (np.maximum(volume - internal, 0.0) <= epsilon)
        return retention, contribution, closed

    observed_retention, contribution, closed = statistics(macro)
    if n_macro == 1:
        p_values = np.zeros(1, dtype=np.float64)
        return (
            np.asarray([0], dtype=np.int64),
            np.empty(0, dtype=np.int64),
            contribution,
            p_values,
            {
                "mode": "feature-determined-state-retention-permutation-Holm-null",
                "status": "single-state",
                "requested_permutations": permutations,
                "effective_permutations": 0,
                "observed_state_retention": observed_retention.tolist(),
            },
        )
    if permutations <= 0:
        raise ValueError("positive recurrence-null permutations are required")
    effective_permutations = max(int(permutations), 9_999)
    graph_bytes = b"".join(
        [
            np.ascontiguousarray(macro).tobytes(),
            np.ascontiguousarray(sizes).tobytes(),
            np.ascontiguousarray(left).tobytes(),
            np.ascontiguousarray(right).tobytes(),
            np.ascontiguousarray(weight).tobytes(),
        ]
    )
    graph_seed = int.from_bytes(hashlib.sha256(graph_bytes).digest()[:8], "little")
    rng = np.random.default_rng(graph_seed)
    null_retention = np.empty((effective_permutations, n_macro), dtype=np.float64)
    for iteration in range(effective_permutations):
        null_retention[iteration], _, _ = statistics(rng.permutation(macro))
    raw_p = np.asarray(
        [
            (1.0 + np.count_nonzero(null_retention[:, state] >= value - epsilon))
            / (effective_permutations + 1.0)
            for state, value in enumerate(observed_retention)
        ],
        dtype=np.float64,
    )
    order = np.argsort(raw_p, kind="stable")
    adjusted_sorted = np.maximum.accumulate(
        (n_macro - np.arange(n_macro)) * raw_p[order]
    )
    p_values = np.empty(n_macro, dtype=np.float64)
    p_values[order] = np.minimum(adjusted_sorted, 1.0)
    recurrent = np.flatnonzero((p_values <= alpha) | closed)
    recurrent_set = set(map(int, recurrent))
    transient = np.asarray(
        [state for state in range(n_macro) if state not in recurrent_set], dtype=np.int64
    )
    return recurrent, transient, contribution, p_values, {
        "mode": "feature-determined-state-retention-permutation-Holm-null",
        "status": "tested",
        "alpha": alpha,
        "requested_permutations": int(permutations),
        "effective_permutations": effective_permutations,
        "graph_seed": graph_seed,
        "observed_state_retention": observed_retention.tolist(),
        "raw_p_values": raw_p.tolist(),
        "holm_p_values": p_values.tolist(),
        "null_state_q50": np.quantile(null_retention, 0.50, axis=0).tolist(),
        "null_state_q95": np.quantile(null_retention, 0.95, axis=0).tolist(),
        "closed_states": np.flatnonzero(closed).astype(int).tolist(),
    }


def recurrent_ellipsoidal_absorption(
    X: NDArray[np.float64],
    micro_labels: NDArray[np.int64],
    macro_by_micro: NDArray[np.int64],
    ellipsoids: list[Ellipsoid],
    edges: list[MicroEdge],
    epsilon: float = 1e-12,
    alpha: float = 0.05,
    permutations: int = 499,
    seed: int = 20260824,
    minimum_states: int = 1,
) -> tuple[NDArray[np.int64], dict[str, object]]:
    """Absorb non-recurrent macro-states by intrinsic ellipsoidal entry energy.

    ``E(U,C)`` is the median, over observations in transient state ``U``, of their minimum
    operational Mahalanobis depth relative to an ellipsoid in recurrent core ``C``. The median is
    a fixed robust functional. Ties prefer the larger recurrent observation mass and then the
    smaller deterministic state index.
    """

    values = np.asarray(X, dtype=np.float64)
    micro = np.asarray(micro_labels, dtype=np.int64)
    macro = relabel_consecutive(np.asarray(macro_by_micro, dtype=np.int64))
    n_micro = int(micro.max()) + 1
    if macro.shape != (n_micro,) or len(ellipsoids) != n_micro:
        raise ValueError("micro labels, macro mapping, and ellipsoids must agree")
    micro_sizes = np.bincount(micro, minlength=n_micro)
    recurrent, transient, contribution, recurrence_p, null_diagnostics = (
        null_calibrated_recurrent_states(
            macro,
            micro_sizes,
            edges,
            alpha,
            permutations,
            seed,
            epsilon,
        )
    )
    state_floor = min(max(int(minimum_states), 1), int(macro.max()) + 1)
    macro_mass = np.bincount(
        macro, weights=micro_sizes, minlength=int(macro.max()) + 1
    )
    promoted: list[int] = []
    calibration_admissible = recurrent.size >= state_floor
    if recurrent.size < state_floor:
        closed = set(map(int, null_diagnostics.get("closed_states", [])))
        recurrent = np.asarray(
            [
                state
                for state, value in enumerate(contribution)
                if value > epsilon or state in closed
            ],
            dtype=np.int64,
        )
        if recurrent.size < state_floor:
            observed_retention = np.asarray(
                null_diagnostics.get(
                    "observed_state_retention", np.zeros(macro_mass.size)
                ),
                dtype=np.float64,
            )
            recurrent_set = set(map(int, recurrent))
            candidates = [
                state for state in range(macro_mass.size) if state not in recurrent_set
            ]
            candidates.sort(
                key=lambda state: (
                    float(recurrence_p[state]),
                    -float(contribution[state]),
                    -float(observed_retention[state]),
                    -float(macro_mass[state]),
                    state,
                )
            )
            promoted = candidates[: state_floor - recurrent.size]
            recurrent = np.asarray(
                sorted([*map(int, recurrent), *promoted]), dtype=np.int64
            )
        recurrent_set = set(map(int, recurrent))
        transient = np.asarray(
            [state for state in range(macro_mass.size) if state not in recurrent_set],
            dtype=np.int64,
        )
    null_diagnostics["certified_minimum_states"] = state_floor
    null_diagnostics["calibration_admissible"] = calibration_admissible
    null_diagnostics["underidentified_action"] = (
        "not-applicable"
        if calibration_admissible
        else "retain-uncalibrated-positive-recurrence-and-promote-to-certified-floor"
        if promoted
        else "retain-uncalibrated-positive-recurrence"
    )
    null_diagnostics["minimum_state_floor_applied"] = bool(promoted)
    null_diagnostics["floor_promoted_states"] = promoted
    if not recurrent.size:
        recurrent = np.asarray([int(np.argmax(macro_mass))], dtype=np.int64)
        transient = np.asarray(
            [state for state in range(macro_mass.size) if state != recurrent[0]],
            dtype=np.int64,
        )
    if not transient.size:
        return macro, {
            "mode": "recurrent-ellipsoidal-absorption",
            "status": "all-states-recurrent",
            "initial_states": int(np.unique(macro).size),
            "selected_states": int(np.unique(macro).size),
            "recurrent_states": int(recurrent.size),
            "transient_states": 0,
            "parent_flow_retention": ((micro_sizes - 1.0) / micro_sizes).tolist(),
            "stability_contribution": contribution.tolist(),
            "recurrence_p_values": recurrence_p.tolist(),
            "recurrence_null": null_diagnostics,
            "absorption_records": [],
        }

    selected = macro.copy()
    records: list[dict[str, object]] = []
    for state in transient:
        source_micro = np.flatnonzero(macro == state)
        source_points = np.flatnonzero(np.isin(micro, source_micro))
        energies = np.full(recurrent.size, np.inf, dtype=np.float64)
        for position, core in enumerate(recurrent):
            target_micro = np.flatnonzero(macro == core)
            point_energy = np.full(source_points.size, np.inf, dtype=np.float64)
            for target in target_micro:
                ellipsoid = ellipsoids[int(target)]
                normalized = ellipsoid.distance(values[source_points]) / max(
                    float(ellipsoid.zeta), epsilon
                )
                point_energy = np.minimum(point_energy, normalized)
            energies[position] = float(np.median(point_energy))
        minimum = float(energies.min())
        ties = np.flatnonzero(np.isclose(energies, minimum, atol=1e-12))
        if ties.size > 1:
            masses = macro_mass[recurrent[ties]]
            target_position = int(ties[np.argmax(masses)])
        else:
            target_position = int(ties[0])
        target = int(recurrent[target_position])
        selected[macro == state] = target
        records.append(
            {
                "transient_state": int(state),
                "absorbing_core": target,
                "entry_energy": minimum,
                "observations": int(source_points.size),
            }
        )
    selected = relabel_consecutive(selected)
    return selected, {
        "mode": "recurrent-ellipsoidal-absorption",
        "status": "transient-states-absorbed",
        "initial_states": int(np.unique(macro).size),
        "selected_states": int(np.unique(selected).size),
        "recurrent_states": int(recurrent.size),
        "transient_states": int(transient.size),
        "parent_flow_retention": ((micro_sizes - 1.0) / micro_sizes).tolist(),
        "stability_contribution": contribution.tolist(),
        "recurrence_p_values": recurrence_p.tolist(),
        "recurrence_null": null_diagnostics,
        "absorption_records": records,
    }

# ============================================================================
# Original module: aedpcv2/model.py
# ============================================================================

"""Scikit-learn-style AEDPCv2 estimator."""


import platform
import time
from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClusterMixin



class AEDPCv2(ClusterMixin, BaseEstimator):
    """Transductive AEDPCv2 clustering estimator.

    Parameters
    ----------
    alpha_n:
        ABIDE neighborhood-scale significance level.
    alpha_m:
        Max-statistic Markov fusion significance level.
    config:
        Optional frozen research configuration. If supplied, its statistical controls are
        replaced by the explicit ``alpha_n`` and ``alpha_m`` arguments.
    """

    def __init__(
        self,
        alpha_n: float = 0.01,
        alpha_m: float = 0.05,
        config: AEDPCv2Config | None = None,
    ) -> None:
        self.alpha_n = alpha_n
        self.alpha_m = alpha_m
        self.config = config

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "AEDPCv2":
        del y
        started = time.perf_counter()
        configuration = self.config or AEDPCv2Config(alpha_n=self.alpha_n, alpha_m=self.alpha_m)
        if configuration.alpha_n != self.alpha_n or configuration.alpha_m != self.alpha_m:
            configuration = replace(configuration, alpha_n=self.alpha_n, alpha_m=self.alpha_m)
        standardized, preprocessing = mean_impute_zscore(X)
        neighbors = compute_neighbors(standardized, configuration)
        density = run_abide_pak(neighbors.distances, neighbors.indices, configuration)
        density_kstar = density.kstar
        kstar = density_kstar.copy()
        calibration_diagnostics: dict[str, object] = {"mode": "legacy"}
        if configuration.routing_scale in {"calibrated", "pointwise"} and configuration.adaptive_neighborhood:
            calibrated = (
                pointwise_calibrated_neighbor_sets(
                    neighbors.distances,
                    neighbors.indices,
                    density_kstar,
                    density.intrinsic_dimension,
                    configuration.shared_neighbor_evidence,
                )
                if configuration.routing_scale == "pointwise"
                else calibrated_neighbor_sets(
                    neighbors.distances,
                    neighbors.indices,
                    configuration.shared_neighbor_evidence,
                )
            )
            kstar = calibrated.routing_k
            candidate = calibrated.routing_candidate
            reciprocal = calibrated.routing_reciprocal
            evidence_neighbors = calibrated.evidence_reciprocal
            evidence_radius = neighbors.distances[
                np.arange(standardized.shape[0]), calibrated.evidence_k
            ]
            calibration_diagnostics = {
                "mode": (
                    "pointwise-ABIDE-topology-bounded-reciprocal"
                    if configuration.routing_scale == "pointwise"
                    else "connectivity-calibrated-adaptive-reciprocal"
                ),
                "connectivity_floor": calibrated.connectivity_floor,
                "natural_saturation_scale": calibrated.natural_saturation_scale,
                "reciprocal_percolation_scale": calibrated.reciprocal_percolation_scale,
                "routing_k_min": int(np.min(calibrated.routing_k)),
                "routing_k_median": float(np.median(calibrated.routing_k)),
                "routing_k_max": int(np.max(calibrated.routing_k)),
                "evidence_k_min": int(np.min(calibrated.evidence_k)),
                "evidence_k_median": float(np.median(calibrated.evidence_k)),
                "evidence_k_max": int(np.max(calibrated.evidence_k)),
                "routing_closure_edges": calibrated.routing_closure_edges,
                "evidence_closure_edges": calibrated.evidence_closure_edges,
                "routing_edges": calibrated.routing_edges,
                "evidence_edges": calibrated.evidence_edges,
                "hubness_max_occurrence": calibrated.hubness_max_occurrence,
                "hubness_occurrence_skewness": calibrated.hubness_occurrence_skewness,
                "pointwise_status": calibrated.pointwise_status,
            }
        elif not configuration.adaptive_neighborhood:
            fixed = min(int(np.ceil(np.sqrt(standardized.shape[0]))), neighbors.indices.shape[1] - 1)
            kstar = np.full(standardized.shape[0], fixed, dtype=np.int64)
        elif configuration.routing_scale == "stabilized":
            if density.intrinsic_dimension <= configuration.low_id_cutoff:
                # A logarithmic mutual graph preserves separated curved manifolds;
                # ABIDE can otherwise select a scale large enough to bridge them.
                fixed = min(
                    int(np.ceil(np.log2(standardized.shape[0]))),
                    neighbors.indices.shape[1] - 1,
                )
                kstar = np.full(standardized.shape[0], max(fixed, 2), dtype=np.int64)
            else:
                floor = min(
                    int(np.ceil(np.sqrt(standardized.shape[0]))),
                    neighbors.indices.shape[1] - 1,
                )
                kstar = np.maximum(kstar, floor)
        if calibration_diagnostics["mode"] == "legacy":
            candidate, reciprocal = adaptive_neighbor_sets(neighbors.indices, kstar)
            evidence_neighbors = reciprocal
            evidence_radius = neighbors.distances[np.arange(standardized.shape[0]), kstar]
        routing_neighbors = reciprocal if configuration.reciprocity_mode == "mutual" else candidate
        log_density = density.log_density
        log_density_error = density.log_density_error
        if configuration.prominence_mode == "raw":
            prominence = np.asarray(log_density, dtype=np.float64).copy()
        elif configuration.prominence_mode == "lower_bound":
            prominence = lower_confidence_prominence(
                log_density, log_density_error, configuration.alpha_n
            )
        else:
            prominence = confidence_prominence(
                log_density, log_density_error, candidate, configuration.epsilon
            )
        prominence, prominence_repairs = repair_nonfinite_prominence(prominence, candidate)
        lookup = distance_maps(neighbors.distances, neighbors.indices)
        routing_radius = neighbors.distances[np.arange(standardized.shape[0]), kstar]
        routing_radius = np.maximum(routing_radius, configuration.epsilon)
        if "F" in configuration.evidence:
            parent, ascent_ranks = route_parents_with_ranks(
                prominence,
                routing_neighbors,
                lookup,
                routing_radius,
                configuration.epsilon,
                configuration.parent_mode,
            )
        else:
            parent = route_parents(
                prominence,
                routing_neighbors,
                lookup,
                routing_radius,
                configuration.epsilon,
                configuration.parent_mode,
            )
            ascent_ranks = None
        validate_parent_forest(parent, prominence)
        micro_labels = labels_from_parent(parent)
        ellipsoids, boundary_weights = fit_ellipsoids(
            standardized,
            micro_labels,
            density.intrinsic_dimension,
            routing_radius,
            configuration.boundary_coverage,
            configuration.epsilon,
            spherical=configuration.geometry_mode == "spherical",
            full_space=configuration.geometry_mode == "full",
            reliability_blend=configuration.geometry_mode == "adaptive",
            support_neighbors=(
                routing_neighbors if configuration.topology_supported_ellipsoid else None
            ),
        )
        geometry_calibration: dict[str, object] = {
            "requested": configuration.geometry_mode,
            "selected": configuration.geometry_mode,
        }
        if configuration.geometry_mode == "adaptive":
            reliable_fraction = float(
                np.mean([item.directional_reliability > 0 for item in ellipsoids])
            )
            reliability_threshold = 1.0 / np.sqrt(max(len(ellipsoids), 1))
            if reliable_fraction > reliability_threshold:
                ellipsoids, boundary_weights = fit_ellipsoids(
                    standardized,
                    micro_labels,
                    density.intrinsic_dimension,
                    routing_radius,
                    configuration.boundary_coverage,
                    configuration.epsilon,
                    spherical=False,
                    full_space=False,
                    reliability_blend=False,
                    support_neighbors=(
                        routing_neighbors
                        if configuration.topology_supported_ellipsoid
                        else None
                    ),
                )
                geometry_calibration["selected"] = "intrinsic-supported"
            else:
                geometry_calibration["selected"] = "shrinkage-blended"
            geometry_calibration["reliable_fraction"] = reliable_fraction
            geometry_calibration["reliability_threshold"] = reliability_threshold
        micro_graph = build_micro_graph(
            standardized,
            micro_labels,
            evidence_neighbors,
            lookup,
            np.maximum(evidence_radius, configuration.epsilon),
            log_density,
            log_density_error,
            prominence,
            boundary_weights,
            ellipsoids,
            configuration.epsilon,
            ascent_ranks=ascent_ranks,
            evidence=configuration.evidence,
            geometry_mode=configuration.ellipsoid_contact_mode,
        )
        n_micro = int(micro_labels.max()) + 1
        effective_fusion_mode = configuration.fusion_mode
        automatic_fusion_diagnostics: dict[str, object] | None = None
        if configuration.fusion_mode == "automatic":
            low_id_assessment: dict[str, object] | None = None
            if density.intrinsic_dimension <= configuration.low_id_cutoff:
                automatic_point_groups = connected_components(
                    range(standardized.shape[0]),
                    [
                        tuple(sorted((i, j)))
                        for i, row in enumerate(routing_neighbors)
                        for j in row
                        if i != j
                    ],
                )
                automatic_point_group = np.empty(standardized.shape[0], dtype=np.int64)
                for group_id, group in enumerate(automatic_point_groups):
                    automatic_point_group[list(group)] = group_id
                automatic_preliminary = np.full(n_micro, -1, dtype=np.int64)
                for micro in range(n_micro):
                    memberships = np.unique(automatic_point_group[micro_labels == micro])
                    if memberships.size != 1:
                        raise RuntimeError("a density-flow basin crossed a routing component")
                    automatic_preliminary[micro] = memberships[0]
                automatic_preliminary = relabel_consecutive(automatic_preliminary)
                low_id_assessment = partition_stability_assessment(
                    automatic_preliminary,
                    n_micro,
                    micro_graph,
                    density.intrinsic_dimension,
                    configuration.epsilon,
                )
            effective_fusion_mode, automatic_fusion_diagnostics = automatic_fusion_regime(
                density.intrinsic_dimension,
                configuration.low_id_cutoff,
                None if low_id_assessment is None else bool(low_id_assessment["weak"]),
            )
            automatic_fusion_diagnostics["low_id_partition_assessment"] = low_id_assessment
        if effective_fusion_mode == "correction":
            point_groups = connected_components(
                range(standardized.shape[0]),
                [
                    tuple(sorted((i, j)))
                    for i, row in enumerate(routing_neighbors)
                    for j in row
                    if i != j
                ],
            )
            point_group = np.empty(standardized.shape[0], dtype=np.int64)
            for group_id, group in enumerate(point_groups):
                point_group[list(group)] = group_id
            preliminary = np.full(n_micro, -1, dtype=np.int64)
            for micro in range(n_micro):
                memberships = np.unique(point_group[micro_labels == micro])
                if memberships.size != 1:
                    raise RuntimeError("a density-flow basin crossed a routing component")
                preliminary[micro] = memberships[0]
            preliminary = relabel_consecutive(preliminary)
            micro_peak_density = np.full(n_micro, -np.inf, dtype=np.float64)
            micro_peak_error = np.full(n_micro, np.inf, dtype=np.float64)
            for micro in range(n_micro):
                members = np.flatnonzero(micro_labels == micro)
                peak = int(members[np.argmax(log_density[members])])
                micro_peak_density[micro] = log_density[peak]
                micro_peak_error[micro] = max(
                    float(log_density_error[peak]), configuration.epsilon
                )
            hubness_skewness = float(
                calibration_diagnostics.get("hubness_occurrence_skewness", 0.0)
            )
            speciation_p: float | None = None
            speciation_diagnostics: dict[str, object] = {
                "mode": "crossfit-one-ellipsoid-speciation",
                "status": "not-required",
            }
            speciation_rejected = False
            speciation_minimum_states = 1
            multi_axial_speciation = False
            if (
                configuration.crossfit_ellipsoidal_speciation
                and hubness_skewness > 1.0
            ):
                speciation_p, speciation_diagnostics = crossfit_ellipsoidal_speciation(
                    standardized,
                    configuration.null_permutations,
                    configuration.random_seed,
                    configuration.epsilon,
                    configuration.alpha_m,
                    configuration.alpha_n,
                )
                speciation_rejected = bool(speciation_p <= configuration.alpha_m)
                speciation_minimum_states = int(
                    speciation_diagnostics.get("minimum_states", 1)
                )
                multi_axial_speciation = speciation_minimum_states >= 4
                speciation_diagnostics.update(
                    {
                        "alpha": configuration.alpha_m,
                        "trigger": "hubness-conditioned-density-topography",
                        "rejected_one_ellipsoid": speciation_rejected,
                    }
                )
            if configuration.density_topography and (
                hubness_skewness <= 1.0
                or (
                    configuration.speciation_conditioned_topography
                    and multi_axial_speciation
                )
            ):
                preliminary, topography_diagnostics = density_topography_components(
                    preliminary,
                    micro_graph,
                    micro_peak_density,
                    micro_peak_error,
                    configuration.alpha_n,
                    configuration.epsilon,
                )
                topography_diagnostics["status"] = (
                    "evaluated-under-hubness-with-speciation"
                    if hubness_skewness > 1.0
                    else "evaluated"
                )
            else:
                topography_diagnostics = {
                    "mode": "peak-saddle-density-topography",
                    "status": (
                        "skipped-under-hubness"
                        if hubness_skewness > 1.0
                        else "disabled-ablation"
                    ),
                    "initial_clusters": int(np.unique(preliminary).size),
                    "selected_clusters": int(np.unique(preliminary).size),
                    "accepted_merges": 0,
                }
            point_manifold_diagnostics: dict[str, object] = {
                "mode": "persistent-point-manifold",
                "status": "not-triggered",
            }
            if (
                configuration.persistent_point_manifold
                and density.intrinsic_dimension <= 2.0
                and topography_diagnostics.get("selected_clusters") == 1
            ):
                preliminary, point_manifold_diagnostics = (
                    persistent_point_manifold_partition(
                        routing_neighbors,
                        lookup,
                        routing_radius,
                        micro_labels,
                        configuration.random_seed,
                        configuration.partition_lineage_persistence,
                        configuration.epsilon,
                    )
                )
            micro_sizes = np.bincount(micro_labels, minlength=n_micro)
            dominant_basin_mass = float(np.max(micro_sizes)) / standardized.shape[0]
            finite_component_mass = 1.0 / n_micro
            base_markov_resolution = max(
                1.0,
                density.intrinsic_dimension
                * (dominant_basin_mass + finite_component_mass),
            )
            positive_edge_count = sum(edge.weight > 0 for edge in micro_graph)
            average_micro_degree = 2.0 * positive_edge_count / max(n_micro, 1)
            if (
                configuration.markov_time_mode == "adaptive"
                and hubness_skewness > 1.0
                and average_micro_degree > 0
            ):
                mixing_ratio = max(
                    1.0, np.log2(max(n_micro, 2)) / average_micro_degree
                )
                markov_time = float(2 ** int(np.ceil(np.log2(mixing_ratio))))
            else:
                markov_time = 1.0
            fragmentation_threshold = np.sqrt(
                standardized.shape[0] * max(density.intrinsic_dimension, 1.0)
            )
            persistent_time_diagnostics: dict[str, object] = {
                "mode": "persistent-Markov-time",
                "status": "not-triggered",
            }
            if (
                configuration.markov_time_mode == "adaptive"
                and configuration.persistent_topography_time
                and density.intrinsic_dimension > 2.0
                and (
                    (
                        topography_diagnostics.get("selected_clusters") == 1
                        and n_micro > fragmentation_threshold
                    )
                    or multi_axial_speciation
                )
            ):
                markov_time, persistent_time_diagnostics = persistent_markov_time(
                    n_micro,
                    micro_graph,
                    base_markov_resolution,
                    configuration.random_seed,
                    speciation_minimum_states,
                    micro_sizes.astype(np.float64, copy=False),
                    configuration.partition_lineage_persistence,
                )
            markov_resolution = base_markov_resolution / markov_time
            macro_by_micro, fusion_history, fusion_diagnostics = markov_correct_partition(
                preliminary,
                n_micro,
                micro_graph,
                configuration.alpha_m,
                configuration.null_permutations,
                configuration.null_swaps_per_edge,
                configuration.random_seed,
                markov_resolution,
                hubness_skewness > 1.0,
                speciation_minimum_states,
                configuration.sequential_null,
                configuration.epsilon,
            )
            fusion_diagnostics = {
                **fusion_diagnostics,
                "routing_connected_components": len(point_groups),
                "dominant_basin_mass": dominant_basin_mass,
                "finite_component_mass": finite_component_mass,
                "base_markov_resolution": base_markov_resolution,
                "markov_time": markov_time,
                "average_micro_degree": average_micro_degree,
                "fragmentation_threshold": fragmentation_threshold,
                "persistent_time": persistent_time_diagnostics,
                "density_topography": topography_diagnostics,
                "point_manifold": point_manifold_diagnostics,
            }
            candidate_stability = float(
                fusion_diagnostics.get("candidate_stability", np.inf)
            )
            if (
                configuration.crossfit_ellipsoidal_speciation
                and speciation_p is None
                and candidate_stability <= configuration.epsilon
            ):
                speciation_p, speciation_diagnostics = crossfit_ellipsoidal_speciation(
                    standardized,
                    configuration.null_permutations,
                    configuration.random_seed,
                    configuration.epsilon,
                    configuration.alpha_m,
                    configuration.alpha_n,
                )
                speciation_diagnostics["alpha"] = configuration.alpha_m
                speciation_diagnostics["trigger"] = (
                    "nonpositive-candidate-stability"
                )
                speciation_rejected = bool(speciation_p <= configuration.alpha_m)
                speciation_minimum_states = int(
                    speciation_diagnostics.get("minimum_states", 1)
                )
                speciation_diagnostics["rejected_one_ellipsoid"] = speciation_rejected
            if speciation_p is not None:
                gaussian_rejected = bool(
                    speciation_diagnostics.get("gaussian_ellipsoid_rejected", False)
                )
                dip_rejected = bool(
                    speciation_diagnostics.get("unimodality_rejected", False)
                )
                angular_rejected = bool(
                    speciation_diagnostics.get(
                        "elliptical_angular_symmetry_rejected", False
                    )
                )
                positive_graph_stability = bool(
                    candidate_stability > configuration.epsilon
                )
                speciation_rejected = bool(
                    gaussian_rejected
                    and (
                        dip_rejected
                        or (positive_graph_stability and angular_rejected)
                    )
                )
                speciation_minimum_states = (
                    int(
                        speciation_diagnostics.get(
                            "gaussian_state_lower_bound", 1
                        )
                    )
                    if speciation_rejected
                    else 1
                )
                speciation_diagnostics.update(
                    {
                        "positive_graph_stability": positive_graph_stability,
                        "final_multistate_admissible": speciation_rejected,
                        "final_state_lower_bound": speciation_minimum_states,
                        "decision_rule": (
                            "G_and_(D_or_(positive_stability_and_A))"
                        ),
                        "rejected_one_ellipsoid": speciation_rejected,
                    }
                )
            if speciation_p is not None and not speciation_rejected:
                macro_by_micro = np.zeros(n_micro, dtype=np.int64)
                fusion_diagnostics["status"] = "one-ellipsoid-null-retained"
                fusion_diagnostics["selected_clusters"] = 1
            elif (
                speciation_rejected
                and configuration.speciation_conditioned_topography
                and candidate_stability <= configuration.epsilon
                and np.unique(preliminary).size >= speciation_minimum_states
            ):
                macro_by_micro = preliminary.copy()
                fusion_diagnostics["status"] = (
                    "speciation-certified-density-topography-retained"
                )
                fusion_diagnostics["selected_clusters"] = int(
                    np.unique(preliminary).size
                )
            fusion_diagnostics["ellipsoidal_speciation"] = speciation_diagnostics
        elif effective_fusion_mode in {"stability", "hybrid"} and density.intrinsic_dimension <= configuration.low_id_cutoff:
            point_groups = connected_components(
                range(standardized.shape[0]),
                [
                    tuple(sorted((i, j)))
                    for i, row in enumerate(routing_neighbors)
                    for j in row
                    if i != j
                ],
            )
            point_group = np.empty(standardized.shape[0], dtype=np.int64)
            for group_id, group in enumerate(point_groups):
                point_group[list(group)] = group_id
            macro_by_micro = np.full(n_micro, -1, dtype=np.int64)
            for micro in range(n_micro):
                memberships = np.unique(point_group[micro_labels == micro])
                if memberships.size != 1:
                    raise RuntimeError("a parent basin crossed a reciprocal graph component")
                macro_by_micro[micro] = memberships[0]
            macro_by_micro = relabel_consecutive(macro_by_micro)
            fusion_history = []
            fusion_diagnostics = {
                "mode": "stable-reciprocal-components",
                "graph_connected_components": len(point_groups),
            }
        elif effective_fusion_mode in {"stability", "hybrid"}:
            micro_sizes = np.bincount(micro_labels, minlength=n_micro)
            macro_by_micro, fusion_diagnostics = evidence_gap_components(
                n_micro,
                micro_graph,
                micro_sizes,
                np.asarray(
                    [item.directional_reliability for item in ellipsoids],
                    dtype=np.float64,
                ),
                standardized.shape[0],
                configuration.boundary_coverage,
                configuration.epsilon,
            )
            fusion_history = []
        elif effective_fusion_mode == "none":
            macro_by_micro = np.arange(n_micro, dtype=np.int64)
            fusion_history = []
            fusion_diagnostics = {"mode": "none", "graph_connected_components": n_micro}
        elif effective_fusion_mode == "greedy":
            groups = connected_components(
                range(n_micro), [(edge.left, edge.right) for edge in micro_graph if edge.weight > 0]
            )
            macro_by_micro = np.full(n_micro, -1, dtype=np.int64)
            for group_id, group in enumerate(groups):
                macro_by_micro[list(group)] = group_id
            fusion_history = []
            fusion_diagnostics = {"mode": "greedy-connected-components", "graph_connected_components": len(groups)}
        else:
            macro_by_micro, fusion_history, fusion_diagnostics = fuse_components(
                n_micro,
                micro_graph,
                configuration.alpha_m,
                configuration.null_permutations,
                configuration.null_swaps_per_edge,
                configuration.random_seed,
            )
        if effective_fusion_mode == "stability":
            preliminary_diagnostics = fusion_diagnostics
            micro_sizes = np.bincount(micro_labels, minlength=n_micro)
            macro_by_micro, stability_diagnostics = markov_stability_guard(
                macro_by_micro,
                n_micro,
                micro_graph,
                micro_sizes,
                density.intrinsic_dimension,
                configuration.random_seed,
                configuration.epsilon,
            )
            fusion_diagnostics = {
                "mode": "preliminary-plus-markov-stability-guard",
                "preliminary": preliminary_diagnostics,
                "stability": stability_diagnostics,
            }
        micro_sizes = np.bincount(micro_labels, minlength=n_micro)
        recurrence_minimum_states = int(
            fusion_diagnostics.get("ellipsoidal_speciation", {}).get(
                "final_state_lower_bound", 1
            )
        )
        macro_by_micro, recurrence_diagnostics = recurrent_ellipsoidal_absorption(
            standardized,
            micro_labels,
            macro_by_micro,
            ellipsoids,
            micro_graph,
            epsilon=configuration.epsilon,
            alpha=configuration.alpha_m,
            permutations=configuration.null_permutations,
            seed=configuration.random_seed,
            minimum_states=recurrence_minimum_states,
        )
        q = int(
            np.clip(
                np.rint(density.intrinsic_dimension),
                1,
                standardized.shape[1],
            )
        )
        underidentified, collapse_guard_diagnostics = (
            certified_floor_is_geometrically_underidentified(
                macro_by_micro,
                micro_labels,
                recurrence_diagnostics,
                q + 2,
            )
        )
        collapse_guard_diagnostics["status"] = "not-triggered"
        if (
            configuration.certified_collapse_guard
            and effective_fusion_mode == "correction"
            and underidentified
        ):
            stability_preliminary, stability_preliminary_diagnostics = (
                evidence_gap_components(
                    n_micro,
                    micro_graph,
                    micro_sizes,
                    np.asarray(
                        [item.directional_reliability for item in ellipsoids],
                        dtype=np.float64,
                    ),
                    standardized.shape[0],
                    configuration.boundary_coverage,
                    configuration.epsilon,
                )
            )
            stability_preliminary, stability_guard_diagnostics = markov_stability_guard(
                stability_preliminary,
                n_micro,
                micro_graph,
                micro_sizes,
                density.intrinsic_dimension,
                configuration.random_seed,
                configuration.epsilon,
            )
            stability_macro, stability_recurrence = recurrent_ellipsoidal_absorption(
                standardized,
                micro_labels,
                stability_preliminary,
                ellipsoids,
                micro_graph,
                epsilon=configuration.epsilon,
                alpha=configuration.alpha_m,
                permutations=configuration.null_permutations,
                seed=configuration.random_seed,
                minimum_states=recurrence_minimum_states,
            )
            stability_macro, set_flow_history = coarsen_to_certified_state_floor(
                stability_macro,
                micro_sizes,
                micro_graph,
                recurrence_minimum_states,
                configuration.epsilon,
            )
            fallback_underidentified, fallback_support = (
                certified_floor_is_geometrically_underidentified(
                    stability_macro,
                    micro_labels,
                    stability_recurrence,
                    q + 2,
                )
            )
            fallback_clusters = int(np.unique(stability_macro).size)
            accepted = bool(
                not fallback_underidentified
                and fallback_clusters >= recurrence_minimum_states
                and int(fallback_support["minimum_cluster_points"]) >= q + 2
            )
            collapse_guard_diagnostics.update(
                {
                    "status": (
                        "stability-fallback-accepted"
                        if accepted
                        else "stability-fallback-rejected"
                    ),
                    "certified_minimum_states": recurrence_minimum_states,
                    "fallback_clusters": fallback_clusters,
                    "fallback_support": fallback_support,
                    "fallback_preliminary": stability_preliminary_diagnostics,
                    "fallback_stability": stability_guard_diagnostics,
                    "set_flow_coarsening": set_flow_history,
                    "primary_recurrence": recurrence_diagnostics,
                }
            )
            if accepted:
                macro_by_micro = stability_macro
                recurrence_diagnostics = stability_recurrence
        fusion_diagnostics = {
            **fusion_diagnostics,
            "recurrent_ellipsoidal_absorption": recurrence_diagnostics,
            "certified_collapse_guard": collapse_guard_diagnostics,
        }
        if automatic_fusion_diagnostics is not None:
            fusion_diagnostics = {
                **fusion_diagnostics,
                "automatic_fusion": automatic_fusion_diagnostics,
            }
        labels = relabel_consecutive(macro_by_micro[micro_labels])

        for array in (labels, prominence, parent, micro_labels):
            array.setflags(write=False)
        self._labels: NDArray[np.int64] = labels
        self._prominence: NDArray[np.float64] = prominence
        self._parent: NDArray[np.int64] = parent
        self._microcomponent_labels: NDArray[np.int64] = micro_labels
        self._ellipsoids: tuple[Ellipsoid, ...] = tuple(ellipsoids)
        self._micro_graph: tuple[MicroEdge, ...] = tuple(micro_graph)
        self._fusion_history: tuple[FusionStep | MarkovCorrectionStep, ...] = tuple(
            fusion_history
        )
        self._diagnostics: Mapping[str, object] = MappingProxyType({
            "config": configuration.to_dict(),
            "preprocessing": preprocessing,
            "neighbor_backend": neighbors.backend,
            "intrinsic_dimension": density.intrinsic_dimension,
            "intrinsic_dimension_path": density.intrinsic_dimension_path,
            "intrinsic_dimension_error_path": density.intrinsic_dimension_error_path,
            "kstar": kstar,
            "density_kstar": density_kstar,
            "kstar_path": density.kstar_path,
            "neighborhood_calibration": calibration_diagnostics,
            "density_radius": density.radius,
            "dadapy_version": density.dadapy_version,
            "pak_optimized": density.pak_optimized,
            "pak_equal_shells": density.pak_equal_shells,
            "prominence_nonfinite_repairs": prominence_repairs,
            "n_microcomponents": n_micro,
            "n_clusters": int(np.unique(labels).size),
            "geometry": {
                "calibration": geometry_calibration,
                "adaptive_components": int(
                    sum(item.geometry_kind == "adaptive-reliability-blend" for item in ellipsoids)
                ),
                "isotropic_components": int(
                    sum(item.directional_reliability == 0 for item in ellipsoids)
                ),
                "rank_deficient_components": int(sum(item.rank_deficient for item in ellipsoids)),
                "mean_directional_reliability": float(
                    np.mean([item.directional_reliability for item in ellipsoids])
                ),
                "median_directional_reliability": float(
                    np.median([item.directional_reliability for item in ellipsoids])
                ),
                "median_axis_ratio": float(np.median([item.axis_ratio for item in ellipsoids])),
                "median_oas_shrinkage": float(
                    np.median([item.oas_shrinkage for item in ellipsoids])
                ),
            },
            "fusion": fusion_diagnostics,
            "runtime_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "platform": platform.platform(),
        })
        self.n_features_in_ = standardized.shape[1]
        return self

    def fit_predict(self, X: ArrayLike, y: ArrayLike | None = None) -> NDArray[np.int64]:
        return self.fit(X, y).labels_.copy()

    @property
    def labels_(self) -> NDArray[np.int64]:
        return self._labels

    @property
    def prominence_(self) -> NDArray[np.float64]:
        return self._prominence

    @property
    def parent_(self) -> NDArray[np.int64]:
        return self._parent

    @property
    def microcomponent_labels_(self) -> NDArray[np.int64]:
        return self._microcomponent_labels

    @property
    def ellipsoids_(self) -> tuple[Ellipsoid, ...]:
        return self._ellipsoids

    @property
    def micro_graph_(self) -> tuple[MicroEdge, ...]:
        return self._micro_graph

    @property
    def fusion_history_(self) -> tuple[FusionStep | MarkovCorrectionStep, ...]:
        return self._fusion_history

    @property
    def diagnostics_(self) -> Mapping[str, object]:
        return self._diagnostics

# ============================================================================
# Original module: aedpcv2/automatic.py
# ============================================================================

"""Label-free structural certification for the AEDPCv2 development estimator."""


from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn import metrics
from sklearn.base import BaseEstimator, ClusterMixin



def _partition_quality(
    X: NDArray[np.float64], labels: NDArray[np.int64], seed: int
) -> dict[str, float]:
    """Compute deterministic, label-free compactness/separation diagnostics."""

    n = labels.size
    k = int(np.unique(labels).size)
    if k <= 1 or k >= n:
        return {"silhouette": -1.0, "calinski_harabasz": 0.0,
                "davies_bouldin": float("inf"), "clusters": float(k)}
    silhouette = float(
        metrics.silhouette_score(
            X,
            labels,
            sample_size=min(2000, n),
            random_state=seed,
        )
    )
    return {
        "silhouette": silhouette,
        "calinski_harabasz": float(metrics.calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(metrics.davies_bouldin_score(X, labels)),
        "clusters": float(k),
    }


def _nearest_parent_is_certified(
    topology: Mapping[str, float],
    nearest: Mapping[str, float],
    n_points: int,
    intrinsic_dimension: float,
) -> tuple[bool, dict[str, float | bool]]:
    """Require unanimous internal improvement beyond finite-sample resolution."""

    epsilon = np.finfo(np.float64).eps
    unanimous = bool(
        nearest["silhouette"] > topology["silhouette"]
        and nearest["calinski_harabasz"] > topology["calinski_harabasz"]
        and nearest["davies_bouldin"] < topology["davies_bouldin"]
        and np.isfinite(nearest["davies_bouldin"])
    )
    if unanimous:
        ratios = (
            (1.0 + nearest["silhouette"] + epsilon)
            / (1.0 + topology["silhouette"] + epsilon),
            (nearest["calinski_harabasz"] + epsilon)
            / (topology["calinski_harabasz"] + epsilon),
            (topology["davies_bouldin"] + epsilon)
            / (nearest["davies_bouldin"] + epsilon),
        )
        composite_gain = float(np.prod(ratios) ** (1.0 / 3.0) - 1.0)
    else:
        composite_gain = float("-inf")
    resolution = float(np.sqrt(2.0 * max(intrinsic_dimension, 1.0) / n_points))
    accepted = bool(unanimous and composite_gain > resolution)
    return accepted, {
        "unanimous_improvement": unanimous,
        "composite_gain": composite_gain,
        "finite_sample_resolution": resolution,
        "accepted": accepted,
    }


def _fine_partition_is_certified(
    quality: Mapping[str, float],
    low_similarity: float | None,
    high_similarity: float | None,
    n_points: int,
) -> tuple[bool, dict[str, float | bool | None]]:
    """Require compactness and persistence across data-adaptive alpha scales."""

    resolution = float(1.0 / np.sqrt(n_points))
    compact = bool(
        quality["clusters"] > 1
        and quality["silhouette"] > resolution
    )
    persistence = (
        None
        if low_similarity is None or high_similarity is None
        else float(min(low_similarity, high_similarity))
    )
    persistence_floor = float(1.0 - resolution)
    accepted = bool(
        compact and persistence is not None and persistence >= persistence_floor
    )
    return accepted, {
        "compact_above_resolution": compact,
        "silhouette": float(quality["silhouette"]),
        "finite_sample_resolution": resolution,
        "low_scale_similarity": low_similarity,
        "high_scale_similarity": high_similarity,
        "persistence": persistence,
        "persistence_floor": persistence_floor,
        "accepted": accepted,
    }


class AutomaticAEDPCv2(ClusterMixin, BaseEstimator):
    """AEDPCv2 with label-free parent-view and fusion-persistence certification.

    The estimator exposes only the two statistical controls.  Research ablation
    switches remain internal to the candidate fits and are never selected from y,
    a requested K, or an external clustering metric.
    """

    def __init__(
        self,
        alpha_n: float = 0.01,
        alpha_m: float = 0.05,
        config: AEDPCv2Config | None = None,
    ) -> None:
        self.alpha_n = alpha_n
        self.alpha_m = alpha_m
        self.config = config

    def _fit_core(
        self, X: ArrayLike, configuration: AEDPCv2Config, parent: str, fusion: str,
        alpha_n: float | None = None,
    ) -> AEDPCv2:
        selected_alpha = configuration.alpha_n if alpha_n is None else alpha_n
        candidate = replace(
            configuration,
            alpha_n=selected_alpha,
            parent_mode=parent,
            evidence="BG",
            fusion_mode=fusion,
        )
        return AEDPCv2(
            alpha_n=selected_alpha,
            alpha_m=candidate.alpha_m,
            config=candidate,
        ).fit(X)

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "AutomaticAEDPCv2":
        del y
        configuration = self.config or AEDPCv2Config(
            alpha_n=self.alpha_n, alpha_m=self.alpha_m
        )
        if configuration.alpha_n != self.alpha_n or configuration.alpha_m != self.alpha_m:
            configuration = replace(
                configuration, alpha_n=self.alpha_n, alpha_m=self.alpha_m
            )
        standardized, _ = mean_impute_zscore(X)
        topology = self._fit_core(X, configuration, "topology", "automatic")
        nearest = self._fit_core(X, configuration, "nearest", "automatic")
        topology_quality = _partition_quality(
            standardized, topology.labels_, configuration.random_seed
        )
        nearest_quality = _partition_quality(
            standardized, nearest.labels_, configuration.random_seed
        )
        routing_dimension = float(topology.diagnostics_["intrinsic_dimension"])
        nearest_accepted, parent_gate = _nearest_parent_is_certified(
            topology_quality,
            nearest_quality,
            standardized.shape[0],
            routing_dimension,
        )
        selected_parent = "nearest" if nearest_accepted else "topology"
        coarse = nearest if nearest_accepted else topology

        intrinsic_dimension = float(coarse.diagnostics_["intrinsic_dimension"])
        fine: AEDPCv2 | None = None
        fine_quality: dict[str, float] | None = None
        low_similarity: float | None = None
        high_similarity: float | None = None
        fine_accepted = False
        fine_gate: dict[str, float | bool | None] = {
            "evaluated": False,
            "accepted": False,
        }
        if intrinsic_dimension > configuration.low_id_cutoff:
            fine = self._fit_core(X, configuration, selected_parent, "hybrid")
            fine_quality = _partition_quality(
                standardized, fine.labels_, configuration.random_seed
            )
            resolution = float(1.0 / np.sqrt(standardized.shape[0]))
            compact = bool(
                fine_quality["clusters"] > 1
                and fine_quality["silhouette"] > resolution
            )
            if compact:
                scale = float(np.exp(1.0 / np.sqrt(intrinsic_dimension)))
                alpha_low = float(np.clip(configuration.alpha_n / scale, 0.0005, 0.5))
                alpha_high = float(np.clip(configuration.alpha_n * scale, 0.0005, 0.5))
                low = self._fit_core(
                    X, configuration, selected_parent, "hybrid", alpha_low
                )
                high = self._fit_core(
                    X, configuration, selected_parent, "hybrid", alpha_high
                )
                low_similarity = float(
                    metrics.normalized_mutual_info_score(low.labels_, fine.labels_)
                )
                high_similarity = float(
                    metrics.normalized_mutual_info_score(fine.labels_, high.labels_)
                )
            fine_accepted, fine_gate = _fine_partition_is_certified(
                fine_quality,
                low_similarity,
                high_similarity,
                standardized.shape[0],
            )
            fine_gate.update({
                "evaluated": True,
                "intrinsic_dimension": intrinsic_dimension,
                "alpha_scale": float(np.exp(1.0 / np.sqrt(intrinsic_dimension))),
            })

        selected = fine if fine_accepted and fine is not None else coarse
        selected_diagnostics = dict(selected.diagnostics_)
        selected_diagnostics["automatic_structure"] = {
            "mode": "label-free-parent-and-multiscale-fusion-certification",
            "selected_parent": selected_parent,
            "selected_fusion": "hybrid" if fine_accepted else "stability-gated-correction",
            "topology_quality": topology_quality,
            "nearest_quality": nearest_quality,
            "parent_gate": parent_gate,
            "fine_quality": fine_quality,
            "fine_gate": fine_gate,
        }
        self._selected_model = selected
        self._diagnostics = MappingProxyType(selected_diagnostics)
        self.n_features_in_ = selected.n_features_in_
        return self

    def fit_predict(self, X: ArrayLike, y: ArrayLike | None = None) -> NDArray[np.int64]:
        return self.fit(X, y).labels_.copy()

    @property
    def labels_(self) -> NDArray[np.int64]:
        return self._selected_model.labels_

    @property
    def prominence_(self) -> NDArray[np.float64]:
        return self._selected_model.prominence_

    @property
    def parent_(self) -> NDArray[np.int64]:
        return self._selected_model.parent_

    @property
    def microcomponent_labels_(self) -> NDArray[np.int64]:
        return self._selected_model.microcomponent_labels_

    @property
    def ellipsoids_(self) -> tuple[Ellipsoid, ...]:
        return self._selected_model.ellipsoids_

    @property
    def micro_graph_(self) -> tuple[MicroEdge, ...]:
        return self._selected_model.micro_graph_

    @property
    def fusion_history_(self) -> tuple[FusionStep | MarkovCorrectionStep, ...]:
        return self._selected_model.fusion_history_

    @property
    def diagnostics_(self) -> Mapping[str, object]:
        return self._diagnostics

# Main algorithm entry points.
AEDPC = AutomaticAEDPCv2
AEDPCCore = AEDPCv2
AEDPCConfig = AEDPCv2Config
__version__ = "0.16.0"
__all__ = [
    "AEDPC", "AEDPCCore", "AEDPCConfig", "AutomaticAEDPCv2",
    "AEDPCv2", "AEDPCv2Config",
]
