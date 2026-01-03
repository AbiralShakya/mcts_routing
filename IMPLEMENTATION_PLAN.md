# Diffusion-MCTS Router for nextpnr-xilinx

## Implementation Plan

### Overview

Replace traditional routing with diffusion + MCTS. The core insight: **optimize for routing success, not wirelength proxies**.

```
Algorithm:
def iterate(root):
    node = ucb_select(root)
    while node.t > 0:
        node = denoise_step(node)
        if critic(node) < threshold:
            return backprop(node, 0)  # pruned
    score = nextpnr_route(node.routing)
    backprop(node, score)
```

---

## Core Concepts

### Routing State

A **routing state** represents partial routing assignment:
- Some nets fully routed (PIPs committed)
- Some nets unresolved (latent distributions over PIPs)
- Congestion map tracking resource usage

```python
@dataclass
class RoutingState:
    net_latents: Dict[int, torch.Tensor]  # net_id -> [num_pips] logits over feasible PIPs
    timestep: int                          # Diffusion timestep (T=noise, 0=committed)
    routed_nets: set                       # Nets with committed routes
    congestion_map: Optional[torch.Tensor] # [H, W] resource usage
```

### Net Latents

Each net n has latent z_n in Delta^|E_n| - a distribution over its feasible PIPs/edges.
- At t=T: High entropy (uniform-ish)
- At t=0: Low entropy (committed to specific PIPs)

---

## Components

### 1. `src/diffusion/` - Routing Denoising Model

**Purpose**: Generate routing assignments by denoising from noise, conditioned on netlist and current congestion.

| File | Description |
|------|-------------|
| `model.py` | `RoutingDiffusion` - Main diffusion model with net/congestion encoders |
| `schedule.py` | `DDPMSchedule` - Noise schedule (beta_t) |
| `sampler.py` | `denoise_step`, `sample_routing`, `initialize_routing_state` |

**Key classes**:
- `RoutingState`: Partial routing (net latents, congestion, routed nets)
- `RoutingDiffusion`: NetEncoder + CongestionEncoder + RoutingDenoiser
- `NetEncoder`: Encodes net features (fanout, bounding box, criticality)
- `CongestionEncoder`: Encodes spatial congestion patterns
- `RoutingDenoiser`: Predicts noise to remove at each step

### 2. `src/critic/` - GNN Routability Predictor

**Purpose**: Predict routing success from partial routing state. Enables early pruning of bad MCTS paths.

| File | Description |
|------|-------------|
| `gnn.py` | `RoutingCritic` - GNN with congestion-aware message passing |
| `features.py` | `RoutingGraphBuilder` - Converts partial routing -> graph |
| `training.py` | `CriticTrainer` - Training loop with BCE loss on router outcomes |

**Key classes**:
- `RoutingCritic`: GNN predicting V(s) ~ E[final routing score | s]
- `CongestionAwareMP`: Message passing layer that propagates congestion info
- `RoutingGraph`: Graph representation with node features, edges, congestion, unrouted mask
- `RoutingGraphBuilder`: Builds graph from RoutingState + Netlist

### 3. `src/mcts/` - Tree Search

**Purpose**: MCTS components for routing search.

| File | Description |
|------|-------------|
| `node.py` | `RoutingNode` - MCTS tree node with routing state |
| `tree.py` | `MCTSTree` - Tree structure, best terminal selection |
| `ucb.py` | `ucb_select`, `ucb_score` - UCB selection |
| `backprop.py` | `backpropagate` - Value backpropagation |
| `search.py` | `MCTSRouter`, `iterate` - Main search loop |

**Key classes**:
- `RoutingNode`: MCTS node with (state, parent, children, visits, Q-value, pruned)
- `MCTSRouter`: Full routing search (UCB select -> denoise -> critic -> route -> backprop)
- `RouterConfig`: Configuration (num_timesteps, ucb_c, critic_threshold, etc.)

### 4. `src/bridge/` - nextpnr Router Interface

**Purpose**: Interface to nextpnr router for ground-truth evaluation.

| File | Description |
|------|-------------|
| `router.py` | `NextPNRRouter` - Python interface (subprocess or C++ bindings) |
| `placement_io.py` | Export/import utilities |
| `bindings.cpp` | pybind11 bindings to router2 (stub) |
| `CMakeLists.txt` | Build config for C++ bindings |

**Modes**:
1. **Subprocess** (works now): Calls `nextpnr-xilinx` binary, parses output
2. **C++ bindings** (faster): Direct linkage to router2.cc

**Key classes**:
- `NextPNRRouter`: Routes placement and returns RoutingResult
- `RoutingResult`: Success, wirelength, congestion, timing, slack, runtime
- `route_from_assignment()`: Evaluates diffusion-generated routing assignments

---

## Directory Structure

```
src/
├── diffusion/          # Routing denoising model
│   ├── model.py        # RoutingDiffusion, RoutingState, NetEncoder
│   ├── schedule.py     # DDPMSchedule
│   └── sampler.py      # denoise_step, sample_routing
├── critic/             # GNN predicting routability
│   ├── gnn.py          # RoutingCritic, CongestionAwareMP
│   ├── features.py     # RoutingGraphBuilder, RoutingGraph
│   └── training.py     # CriticTrainer
├── mcts/               # Tree, UCB selection, backprop
│   ├── node.py         # RoutingNode
│   ├── tree.py         # MCTSTree
│   ├── ucb.py          # ucb_select, ucb_score
│   ├── backprop.py     # backpropagate
│   └── search.py       # MCTSRouter, iterate
├── bridge/             # Interface to nextpnr router
│   ├── router.py       # NextPNRRouter, RoutingResult
│   ├── placement_io.py # export/import utilities
│   ├── bindings.cpp    # pybind11 (stub)
│   └── CMakeLists.txt  # Build config
├── core/routing/       # Grid, Netlist, Placement (reuse)
├── integration/nextpnr/ # Reader/writer (reuse)
└── legacy/             # Old code (reference only)
```

---

## Algorithm Details

### Core MCTS Loop

```python
def iterate(root, diffusion, critic, router, grid, netlist, config):
    # 1. UCB Selection - traverse tree to promising leaf
    node = ucb_select(root, config.ucb_c)

    # 2. Denoise with critic pruning
    while node.t > 0:
        # Denoise step - commit routing decisions
        new_state = diffusion.denoise_step(node.state, net_features, net_positions)
        child = RoutingNode(state=new_state, parent=node)
        node.children.append(child)

        # Critic evaluation - predict routing success
        graph = graph_builder.build_graph(new_state, netlist)
        critic_score = critic(graph).item()

        if critic_score < config.critic_threshold:
            # PRUNE: Critic predicts routing failure
            child.pruned = True
            backpropagate(child, 0.0)
            return 0.0

        node = child

    # 3. Terminal: Evaluate with REAL router
    routing = diffusion.decode_routing(node.state, {})
    result = router.route_from_assignment(routing, netlist, grid)
    reward = result.as_reward()

    # 4. Backpropagate
    backpropagate(node, reward)
    return reward
```

### UCB Selection

```
UCB(child) = Q(child) + c * sqrt(ln(N_parent) / N_child)
```

- Q: Mean reward from child subtree
- c: Exploration constant (default 1.41 = sqrt(2))
- N: Visit counts

### Reward Function

```python
def as_reward(self) -> float:
    if not self.success:
        return 0.0

    base_score = 0.5  # For successful route
    timing_score = 0.3 if self.timing_met else max(0, 0.3 * (1 + slack/10))
    congestion_score = 0.1 * max(0, 1 - self.congestion)
    wl_score = 0.1 * max(0, 1 - self.wirelength / 10000)

    return min(1.0, base_score + timing_score + congestion_score + wl_score)
```

---

## Implementation Phases

### Phase 1: Core Infrastructure [DONE]
- [x] Create `src/diffusion/` with RoutingDiffusion, RoutingState
- [x] Create `src/critic/` with RoutingCritic, CongestionAwareMP
- [x] Create `src/mcts/` with RoutingNode, MCTSRouter, iterate()
- [x] Create `src/bridge/` with NextPNRRouter, route_from_assignment()

### Phase 2: Training Pipeline
- [ ] Implement critic data generation (partial routings + router labels)
- [ ] Train critic on synthetic data
- [ ] Validate critic accuracy vs actual routing outcomes

### Phase 3: Diffusion Training
- [ ] Collect routing trajectories from existing flows
- [ ] Train routing diffusion model
- [ ] Validate denoising quality

### Phase 4: Integration
- [ ] End-to-end test on small designs
- [ ] Benchmark vs traditional router (QoR, runtime)
- [ ] Build C++ bindings for speed

### Phase 5: Optimization
- [ ] Tune critic threshold
- [ ] Tune UCB exploration constant
- [ ] Parallelize MCTS simulations

---

## Quick Start

```python
# High-level API
from src.mcts import MCTSRouter, RouterConfig
from src.diffusion import RoutingDiffusion
from src.critic import RoutingCritic
from src.bridge import NextPNRRouter
from src.integration.nextpnr.reader import NextPNRReader

# Load design
reader = NextPNRReader()
grid, netlist, _, _ = reader.read_all("design.json")

# Create components
diffusion = RoutingDiffusion(
    num_timesteps=1000,
    hidden_dim=256,
    num_layers=6,
    max_pips_per_net=1000
)
critic = RoutingCritic(hidden_dim=128, num_layers=4)
router = NextPNRRouter(nextpnr_path="nextpnr-xilinx")

# Create MCTS router
config = RouterConfig(
    num_timesteps=1000,
    max_iterations=1000,
    critic_threshold=0.3,
    ucb_c=1.41
)

mcts_router = MCTSRouter(
    diffusion=diffusion,
    critic=critic,
    router=router,
    grid=grid,
    netlist=netlist,
    config=config,
    device="cuda"
)

# Run search
routing_assignment = mcts_router.route(max_iterations=1000)

# Get statistics
stats = mcts_router.get_statistics()
print(f"Pruned: {stats['num_pruned']}, Routed: {stats['num_routed']}")
```

---

## Key Differences from Placement-Based Approach

| Aspect | Placement | Routing |
|--------|-----------|---------|
| State | Cell positions (x, y) | Net latents over PIPs |
| Output | BEL assignments | PIP assignments per net |
| Latent | z in R^{N x 2} | z_n in Delta^{\|E_n\|} per net |
| Encoder | Cell features | Net features + congestion |
| Congestion | Implicit in density | Explicit congestion map |
| Scorer | Router on placement | Router validates routing |

---

## Testing Strategy

1. **Unit tests** for each component:
   - `tests/unit/test_critic.py` - GNN forward pass, congestion-aware MP
   - `tests/unit/test_bridge.py` - Router subprocess, result parsing
   - `tests/unit/test_mcts.py` - MCTS iteration, UCB selection

2. **Integration tests**:
   - `tests/integration/test_router_e2e.py` - Full flow on toy design

3. **Benchmark**:
   - Compare vs standard router on benchmarks
   - Metrics: routability rate, wirelength, runtime, prune rate

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Critic too slow | Batch evaluation, reduce GNN layers |
| Router subprocess overhead | Build C++ bindings |
| Diffusion doesn't converge | Pre-train on router-generated solutions |
| MCTS explores too much | Tune UCB constant, add prior from diffusion |
| Net latent space too large | Cluster PIPs, use hierarchical routing |

---

## Success Criteria

1. **Routability**: >95% of designs that traditional router handles
2. **Quality**: Within 5% wirelength of baseline
3. **Speed**: <2x traditional runtime (router-bound anyway)
4. **Pruning**: >50% of rollouts pruned by critic (cost reduction)
