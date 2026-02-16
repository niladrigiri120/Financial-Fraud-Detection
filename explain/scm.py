def counterfactual_risk(model, tx, feature_builder, field, new_value):
    """
    Compute counterfactual fraud risk by intervening on one causal variable.
    """
    tx_cf = tx.copy()
    tx_cf[field] = new_value

    X_cf = feature_builder(tx_cf)
    prob_cf = model.predict_proba(X_cf)[0, 1]

    return prob_cf

def causal_explanation(model, tx, feature_builder):
    X_orig = feature_builder(tx)
    base_prob = model.predict_proba(X_orig)[0, 1]

    impacts = []

    for factor in [0.8, 0.5, 0.2, 0.05]:
        new_amount = tx["amount"] * factor
        prob_cf = counterfactual_risk(
            model, tx, feature_builder, "amount", new_amount
        )

        impacts.append({
            "factor": factor,
            "counterfactual_amount": new_amount,
            "risk_after": prob_cf,
            "risk_change": base_prob - prob_cf
        })

    return {
        "original_risk": base_prob,
        "amount_interventions": impacts
    }
