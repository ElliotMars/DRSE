import torch

from utils.credit_assignment import partial_router_objective
from utils.expert_memory import ExpertMemoryManager, VersionedMemoryItem
from utils.online_routing import OnlineRoutingCorrection
from utils.progressive_feedback import (
    ProgressiveFeedbackManager,
    ProgressiveForecastRecord,
)
from utils.subspace_protection import RegressorSubspaceProtector


def _memory_item(
    sample_id: int,
    alignment: float,
    responsibility: float = 0.9,
) -> VersionedMemoryItem:
    sketch = torch.tensor([1.0, 0.0])
    return VersionedMemoryItem(
        sample_id=sample_id,
        origin=sample_id,
        expert_id=0,
        x=torch.zeros(1, 2, 1),
        x_mark=torch.zeros(1, 2, 7),
        target=torch.zeros(1, 2, 1),
        prediction_capability_sketch=sketch,
        normalized_sketch=sketch,
        sample_responsibility=responsibility,
        last_alignment=alignment,
        stable_credit=responsibility * alignment,
        recovery_credit=responsibility * (1.0 - alignment),
        timestamp=sample_id,
    ).to_cpu_storage(torch.float16)


def test_progressive_pipeline_end_to_end_smoke() -> None:
    torch.manual_seed(4)
    horizon, channels, experts = 2, 1, 2
    stream = torch.tensor([0.0, 0.5, 1.0, 0.0, -0.5, 0.25])
    manager = ProgressiveFeedbackManager(
        pred_len=horizon,
        c_out=channels,
        local_credit_temperature=0.5,
    )
    correction = OnlineRoutingCorrection(
        pred_len=horizon,
        c_out=channels,
        num_experts=experts,
        device=torch.device("cpu"),
        correction_lr=0.2,
        correction_decay=0.01,
        correction_grad_clip=10.0,
        correction_logit_clip=5.0,
    )
    router_logits = torch.nn.Parameter(torch.zeros(horizon, channels, experts))
    expert_bias = torch.nn.Parameter(torch.tensor([-0.5, 0.75]))
    router_optimizer = torch.optim.SGD([router_logits], lr=0.1)
    expert_optimizer = torch.optim.SGD([expert_bias], lr=0.05)
    memory = ExpertMemoryManager(
        num_experts=experts,
        stable_capacity=8,
        recovery_capacity=2,
        responsibility_threshold=0.0,
        alignment_threshold=0.5,
        duplicate_threshold=1.1,
        failure_penalty=0.5,
        max_recovery_attempts=2,
    )
    order: list[tuple[int, str, int]] = []
    matured_horizons: list[int] = []
    full_updates = 0
    updated_origins: list[int] = []
    z_before = correction.z.clone()
    router_before = router_logits.detach().clone()

    for origin, observation in enumerate(stream):
        order.append((origin, "release", -1))
        correction.begin_origin(origin)
        events, completed = manager.release(origin, observation.reshape(1))
        for event in events:
            matured_horizons.append(event.horizon_index)
            correction.update(
                event.horizon_index,
                event.record.expert_predictions[event.horizon_index],
                event.record.mixture_prediction[event.horizon_index],
                event.target,
            )
            router_optimizer.zero_grad()
            current_prior = torch.softmax(
                router_logits[event.horizon_index].unsqueeze(0), dim=-1
            )
            router_loss, _ = partial_router_objective(
                current_prior=current_prior,
                correction=correction.z[event.horizon_index].unsqueeze(0),
                expert_prediction=event.record.expert_predictions[
                    event.horizon_index
                ].unsqueeze(0),
                target=event.target.unsqueeze(0),
                local_responsibility=event.local_responsibility.unsqueeze(0),
                local_confidence=event.local_confidence.unsqueeze(0),
                local_credit_weight=0.1,
                entropy_weight=0.01,
            )
            router_loss.backward()
            router_optimizer.step()

        for record in completed:
            # Learn with the old protection state, then commit this record.
            order.append((origin, "expert_update", record.origin))
            expert_optimizer.zero_grad()
            current_expert_prediction = expert_bias.view(1, 1, experts).expand(
                horizon, channels, experts
            )
            expert_loss = (
                current_expert_prediction
                - record.matured_targets.unsqueeze(-1)
            ).pow(2).mean()
            expert_loss.backward()
            expert_optimizer.step()
            full_updates += 1
            updated_origins.append(record.origin)
            order.append((origin, "memory_commit", record.origin))
            owner = int(record.sample_responsibility.argmax().item())
            candidate = _memory_item(record.origin, alignment=0.9)
            candidate.expert_id = owner
            candidate.sample_responsibility = float(
                record.sample_responsibility[owner].item()
            )
            memory.add_candidate(candidate)

        order.append((origin, "predict", origin))
        dense_prior = torch.softmax(router_logits.detach(), dim=-1)
        effective = correction.effective_weights(dense_prior, top_k=experts)
        expert_prediction = expert_bias.detach().view(1, 1, experts).expand(
            horizon, channels, experts
        )
        mixture = (effective * expert_prediction).sum(dim=-1)
        record = ProgressiveForecastRecord(
            origin=origin,
            x=torch.zeros(1, 2, channels),
            x_mark=torch.zeros(1, 2, 7),
            expert_predictions=expert_prediction,
            router_prior=dense_prior,
            router_weights=effective,
            mixture_prediction=mixture,
            capability_sketch=torch.eye(experts),
        )
        manager.add_record(record)

    for origin in range(len(stream)):
        labels = [label for step, label, _ in order if step == origin]
        assert labels.index("release") < labels.index("predict")
    for origin, label, record_origin in order:
        if label == "memory_commit":
            update_index = order.index((origin, "expert_update", record_origin))
            assert update_index < order.index((origin, label, record_origin))
    assert {0, 1}.issubset(set(matured_horizons))
    assert full_updates >= 1
    assert full_updates == len(set(updated_origins))
    assert not torch.equal(correction.z, z_before)
    assert not torch.equal(router_logits.detach(), router_before)

    # A rejected demotion is evicted and cannot silently remain Stable.
    demotion_memory = ExpertMemoryManager(
        num_experts=1,
        stable_capacity=1,
        recovery_capacity=1,
        responsibility_threshold=0.0,
        alignment_threshold=0.7,
        duplicate_threshold=1.1,
        failure_penalty=0.5,
        max_recovery_attempts=2,
    )
    demotion_memory.add_candidate(_memory_item(100, alignment=0.9))
    demotion_memory.add_candidate(_memory_item(101, alignment=0.0))
    stats = demotion_memory.refresh(
        lambda expert_id, item: (
            (0.6, 1.0) if item.sample_id == 100 else (0.0, 1.0)
        ),
        timestamp=200,
        count_recovery_attempts=False,
    )
    assert stats["evicted"] == 1
    assert not demotion_memory.stable_buffers[0].contains(100)

    # Repeated head activations remain visible in the uncentered moment.
    protector = RegressorSubspaceProtector(
        num_experts=1,
        feature_dim=3,
        rank=0,
        max_rank=3,
        energy_threshold=0.99,
        min_samples=2,
    )
    direction = torch.tensor([1.0, -2.0, 0.5])
    assert protector.refresh_expert(
        0,
        direction.repeat(4, 1),
        torch.ones(4),
        sample_count=4,
        step=len(stream),
    )
    basis = protector.states[0].basis[:, 0]
    assert torch.abs(torch.dot(basis, direction / direction.norm())) > 0.999
    tensors = [
        correction.z,
        router_logits.detach(),
        expert_bias.detach(),
        protector.states[0].basis,
    ]
    assert all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
