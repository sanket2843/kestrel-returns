# Kestrel Home - returns risk (Variant A)

Scores each order before dispatch so Ops can **call** the riskiest ~30% of customers before shipping. Recommendation: call, do not hold (see MEMO.md).

## Run it (clean machine, no API key, no internet needed)
```
pip install -r requirements.txt
python app.py            # open http://localhost:8000
```
- Screen: paste one order as JSON, press Check -> decision + reasons.
- Endpoint: `POST /predict` with one order as JSON -> `{score, decision, why[], note}`. Bad input returns HTTP 400 with a plain message.
- `model.joblib` is committed, so `app.py` starts without the data. No AI API is used anywhere, so nothing to fail without a key; cost per prediction is Rs 0.

## Rebuild model and predictions (needs the client data; NOT in this repo, policy s10)
Put train.csv, test_unlabelled.csv, customers.csv, products.csv in `data/` then `python train.py`.
It prints validation numbers, writes `model.joblib` and `predictions.csv` (order_id, score; higher = more likely returned).

## Decisions (made because nobody could be asked)
1. **Leakage removed:** last_service_event_type and pickup_scheduled_at are written after a return (REVERSE_PICKUP = 100% returned in train). Not used. INSTALL_BOOKED exists only in test, so not used either.
2. **Partner-feed duplicates (651 train rows):** kept the crm copy. Not checked whether copies disagree.
3. **October 2025 order values were ~100x too high** (new gateway): where value/(list price x qty) > 10, divided by 100. Unconfirmed with Tanmay.
4. Pincode 000000 = "no address" flag. Customer fields (shield, state, signup date) can be sent in the JSON; if missing the model treats them as unknown.
5. Validation = train Apr-2025..Mar-2026, test Apr-Jun 2026 (2,126 orders, 245 returns). **AUC 0.75 (95% interval 0.72-0.79), accuracy 88.9% vs 88.5% for "nothing returns".** 95% accuracy is not reachable; judge by ranking and rupees.
6. Costs used: return Rs 1,150 (policy, not Rs 600), call Rs 45, call prevents 35% of returns (spring pilot, unverified).

## Known problems
Model validated on one time split only; final model retrained on all 15 months, so its score is not separately measured. Score is a ranking, not a calibrated probability. Reasons come from historical return rates per segment, not from the model's internals. Weak on UPI orders, ceiling fans, mixer-grinders. Flags Shield members about twice as often as others - Ritu must decide the Shield rule before calling starts. Not checked: whether customer_prior_* and shield_member are as of the order date; delivery_note text unused; no tuning; no baseline-rule comparison.

## AI use
Built in a Claude chat session (Claude Sonnet 5.5, code sandbox). Cost: [fill in]. Discarded: leaked columns, the 95% target, the hold plan, Rs 600 cost.
