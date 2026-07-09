from mlflow.entities.gateway_budget_policy import (
    BudgetAction,
    BudgetDuration,
    BudgetDurationUnit,
    BudgetTargetScope,
    BudgetUnit,
    GatewayBudgetPolicy,
)


def _make_policy(**overrides):
    kwargs = {
        "budget_policy_id": "bp-1",
        "budget_unit": BudgetUnit.USD,
        "budget_amount": 42.0,
        "duration": BudgetDuration(unit=BudgetDurationUnit.DAYS, value=1),
        "target_scope": BudgetTargetScope.GLOBAL,
        "budget_action": BudgetAction.ALERT,
        "created_at": 10,
        "last_updated_at": 20,
    }
    kwargs.update(overrides)
    return GatewayBudgetPolicy(**kwargs)


def test_budget_target_scope_user_proto_roundtrip():
    assert BudgetTargetScope.from_proto(BudgetTargetScope.USER.to_proto()) == BudgetTargetScope.USER


def test_user_policy_proto_roundtrip_preserves_principal():
    policy = _make_policy(target_scope=BudgetTargetScope.USER, principal="alice@example.com")
    restored = GatewayBudgetPolicy.from_proto(policy.to_proto())
    assert restored.target_scope == BudgetTargetScope.USER
    assert restored.principal == "alice@example.com"


def test_non_user_policy_principal_defaults_none():
    policy = _make_policy()
    assert policy.principal is None
    restored = GatewayBudgetPolicy.from_proto(policy.to_proto())
    assert restored.principal is None


def test_string_target_scope_coerced_to_enum():
    policy = _make_policy(target_scope="USER", principal="bob")
    assert policy.target_scope is BudgetTargetScope.USER
