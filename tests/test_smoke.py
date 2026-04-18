"""
Smoke test — runs the full forward pass with synthetic data.
No TabPFN, no MuJoCo required. Uses the fallback TransformerBackbone.

Run from project root:
    python tabrl/tests/test_smoke.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import numpy as np
from tabrl.configs.base_config import TabRLConfig
from tabrl.models.tabrl_agent import TabRLAgent
from tabrl.data.replay_buffer import ReplayBuffer
from tabrl.data.d4rl_loader import ICLTransitionDataset


def test_agent_forward():
    print("\n=== Smoke Test: Agent Forward Pass ===")

    config             = TabRLConfig()
    config.device      = "cpu"
    config.context_len = 16
    config.n_candidates = 4
    config.freeze_backbone = True
    config.cql_alpha   = 1.0
    config.shallow_value = False

    # Hopper-like dims
    obs_dim = 11
    act_dim = 3
    B       = 4    # batch size
    L       = config.context_len
    K       = config.n_candidates

    agent = TabRLAgent(config, obs_dim=obs_dim, act_dim=act_dim)
    print(f"  Backbone type: {type(agent.backbone).__name__}")
    print(f"  Trainable params: {sum(p.numel() for p in agent.parameters() if p.requires_grad):,}")

    # Synthetic batch
    ctx_X  = torch.randn(B, L, obs_dim + act_dim)
    ctx_y  = torch.randn(B, L)
    q_obs  = torch.randn(B, obs_dim)
    q_act  = torch.randn(B, act_dim).clamp(-1, 1)

    # ── Pass 1: Proposal ──────────────────────────────────────────────────────
    candidates, mu, sigma, h1 = agent.propose(ctx_X, ctx_y, q_obs)
    print(f"\n  [Pass 1 - Proposal]")
    print(f"    candidates : {candidates.shape}  (expected {B, K, act_dim})")
    print(f"    mu         : {mu.shape}")
    print(f"    sigma      : {sigma.shape}")
    print(f"    h1         : {h1.shape}")
    assert candidates.shape == (B, K, act_dim), f"candidates shape wrong: {candidates.shape}"
    assert mu.shape == (B, act_dim)
    assert sigma.shape == (B, act_dim)

    # ── Pass 2: Evaluation ────────────────────────────────────────────────────
    Q = agent.evaluate(ctx_X, ctx_y, q_obs, candidates)
    print(f"\n  [Pass 2 - Evaluation]")
    print(f"    Q          : {Q.shape}  (expected {B, K})")
    assert Q.shape == (B, K), f"Q shape wrong: {Q.shape}"

    # ── Action Selection ──────────────────────────────────────────────────────
    action, info = agent.select_action(
        ctx_X[:1], ctx_y[:1], q_obs[:1], deterministic=True
    )
    print(f"\n  [Action Selection]")
    print(f"    action     : {action.shape}  Q_max={info['Q_max']:.3f}")
    assert action.shape == (act_dim,)

    print("\n  ✓ Forward pass OK")


def test_td_step():
    print("\n=== Smoke Test: TD Step ===")

    config             = TabRLConfig()
    config.device      = "cpu"
    config.context_len = 16
    config.n_candidates = 4
    config.freeze_backbone = True
    config.cql_alpha   = 1.0
    config.shallow_value = False

    obs_dim, act_dim = 11, 3
    B = 4

    agent = TabRLAgent(config, obs_dim=obs_dim, act_dim=act_dim)

    batch = {
        "context_X": torch.randn(B, config.context_len, obs_dim + act_dim),
        "context_y": torch.randn(B, config.context_len),
        "query_obs": torch.randn(B, obs_dim),
        "query_act": torch.randn(B, act_dim).clamp(-1, 1),
        "query_rew": torch.randn(B),
        "next_obs":  torch.randn(B, obs_dim),
        "terminal":  torch.zeros(B),
    }

    # TD step
    td_loss, td_info = agent.td_step(batch)
    td_loss.backward()

    print(f"  td_loss  : {td_loss.item():.4f}")
    print(f"  q_pred   : {td_info['value/q_pred']:.4f}")
    print(f"  cql      : {td_info.get('value/cql_penalty', 'N/A')}")
    print("  ✓ TD step + backward OK")


def test_bc_step():
    print("\n=== Smoke Test: BC Step ===")

    config             = TabRLConfig()
    config.device      = "cpu"
    config.context_len = 16
    config.n_candidates = 4
    config.freeze_backbone = True
    config.proposal_type = "gaussian"

    obs_dim, act_dim = 11, 3
    B = 4

    agent = TabRLAgent(config, obs_dim=obs_dim, act_dim=act_dim)

    batch = {
        "context_X": torch.randn(B, config.context_len, obs_dim + act_dim),
        "context_y": torch.randn(B, config.context_len),
        "query_obs": torch.randn(B, obs_dim),
        "query_act": torch.randn(B, act_dim).clamp(-0.99, 0.99),
        "query_rew": torch.randn(B),
        "next_obs":  torch.randn(B, obs_dim),
        "terminal":  torch.zeros(B),
    }

    bc_loss, bc_info = agent.bc_step(batch)
    bc_loss.backward()

    print(f"  bc_loss  : {bc_loss.item():.4f}")
    print(f"  sigma    : {bc_info.get('proposal/sigma_mean', 'N/A'):.4f}")
    print("  ✓ BC step + backward OK")


def test_replay_buffer():
    print("\n=== Smoke Test: Replay Buffer ===")

    obs_dim, act_dim = 11, 3
    buf = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, max_size=500, context_len=16)

    # Add transitions
    for i in range(200):
        buf.add(
            obs=np.random.randn(obs_dim).astype(np.float32),
            act=np.random.randn(act_dim).astype(np.float32),
            rew=float(np.random.randn()),
            next_obs=np.random.randn(obs_dim).astype(np.float32),
            done=False,
        )

    assert len(buf) == 200

    # Sample context
    ctx = buf.sample_context(query_idx=100)
    assert ctx["context_X"].shape == (16, obs_dim + act_dim)
    assert ctx["context_y"].shape == (16,)

    # Sample batch
    batch = buf.sample_batch(batch_size=8)
    assert batch["context_X"].shape == (8, 16, obs_dim + act_dim)
    assert batch["query_obs"].shape  == (8, obs_dim)

    print(f"  Buffer size: {len(buf)}")
    print(f"  Context shape: {ctx['context_X'].shape}")
    print(f"  Batch context_X: {batch['context_X'].shape}")
    print("  ✓ Replay buffer OK")


def test_dataset():
    print("\n=== Smoke Test: ICL Dataset ===")

    obs_dim, act_dim = 11, 3
    N = 500

    # Synthetic data
    data = {
        "observations":      np.random.randn(N, obs_dim).astype(np.float32),
        "actions":           np.random.randn(N, act_dim).astype(np.float32),
        "rewards":           np.random.randn(N).astype(np.float32),
        "next_observations": np.random.randn(N, obs_dim).astype(np.float32),
        "terminals":         np.zeros(N, dtype=bool),
    }

    config = TabRLConfig()
    config.context_len  = 16
    config.n_candidates = 4

    dataset = ICLTransitionDataset(data, context_len=16, n_candidates=4)
    assert len(dataset) == N - 16

    item = dataset[0]
    assert item["context_X"].shape == (16, obs_dim + act_dim)
    assert item["context_y"].shape == (16,)
    assert item["query_obs"].shape  == (obs_dim,)
    assert item["query_act"].shape  == (act_dim,)

    print(f"  Dataset length: {len(dataset)}")
    print(f"  context_X: {item['context_X'].shape}")
    print(f"  query_obs: {item['query_obs'].shape}")
    print("  ✓ ICL Dataset OK")


def test_mlp_proposal_ablation():
    print("\n=== Smoke Test: MLP Proposal Head (P1 ablation) ===")

    config             = TabRLConfig()
    config.device      = "cpu"
    config.context_len = 16
    config.n_candidates = 4
    config.freeze_backbone = True
    config.proposal_type = "mlp"
    config.diversity_loss_weight = 0.1

    obs_dim, act_dim = 11, 3
    B = 4

    agent = TabRLAgent(config, obs_dim=obs_dim, act_dim=act_dim)

    batch = {
        "context_X": torch.randn(B, config.context_len, obs_dim + act_dim),
        "context_y": torch.randn(B, config.context_len),
        "query_obs": torch.randn(B, obs_dim),
        "query_act": torch.randn(B, act_dim).clamp(-0.99, 0.99),
        "query_rew": torch.randn(B),
        "next_obs":  torch.randn(B, obs_dim),
        "terminal":  torch.zeros(B),
    }

    bc_loss, bc_info = agent.bc_step(batch)
    bc_loss.backward()
    print(f"  bc_loss  : {bc_loss.item():.4f}")
    print(f"  div_loss : {bc_info.get('proposal/div_loss', 'N/A'):.4f}")
    print("  ✓ MLP proposal ablation OK")


def test_shallow_value_ablation():
    print("\n=== Smoke Test: Shallow Value Head (Design X ablation) ===")

    config             = TabRLConfig()
    config.device      = "cpu"
    config.context_len = 16
    config.n_candidates = 4
    config.freeze_backbone = True
    config.shallow_value = True

    obs_dim, act_dim = 11, 3
    B = 4

    agent = TabRLAgent(config, obs_dim=obs_dim, act_dim=act_dim)

    ctx_X = torch.randn(B, config.context_len, obs_dim + act_dim)
    ctx_y = torch.randn(B, config.context_len)
    q_obs = torch.randn(B, obs_dim)

    candidates, mu, sigma, h1 = agent.propose(ctx_X, ctx_y, q_obs)
    Q = agent.evaluate_shallow(h1, candidates)

    print(f"  Q shape  : {Q.shape}  (expected {B, config.n_candidates})")
    assert Q.shape == (B, config.n_candidates)
    print("  ✓ Shallow value ablation OK")


if __name__ == "__main__":
    test_replay_buffer()
    test_dataset()
    test_agent_forward()
    test_td_step()
    test_bc_step()
    test_mlp_proposal_ablation()
    test_shallow_value_ablation()

    print("\n" + "="*50)
    print("  ALL SMOKE TESTS PASSED ✓")
    print("="*50)
    print("\nNext steps:")
    print("  1. pip install tabpfn minari gymnasium[mujoco]")
    print("  2. python tabrl/train.py --env hopper-medium-v2 --phase pretrain")
