# Prompts For Gold Standard Validation

## Case 01: Emergency Supply Routing

You are planning a 24-hour emergency supply dispatch for a fictional island city after a typhoon. There are three districts:

- North Pier: 1,200 people, road access open, needs drinking water and medical kits.
- East Hill: 800 people, road access blocked, helicopter landing possible, has enough water for 12 hours but no medical kits.
- South Flats: 1,500 people, road access open but flood depth is 30 cm, needs water, food, and blankets.

Available assets:

- 2 trucks, each can carry 600 kg per trip, max 3 trips in 24 hours.
- 1 helicopter, can carry 180 kg per trip, max 4 trips in 24 hours.
- Supplies: water packs 1 kg each, food packs 0.8 kg each, blankets 1.2 kg each, medical kits 2 kg each.

Minimum target:

- Water: 2 packs per person for any district that needs water.
- Food: 1 pack per person for South Flats only.
- Blankets: 1 blanket per 3 people for South Flats only, round up.
- Medical kits: 1 kit per 20 people for North Pier and East Hill, round up.

Please produce a dispatch plan with quantities, vehicle assignment, remaining capacity, and explain any unmet demand or prioritization.

## Case 02: Subscription Cohort SQL

Given these tables:

```sql
users(user_id, signup_date, country)
subscriptions(subscription_id, user_id, plan, started_at, canceled_at, monthly_price_usd)
events(event_id, user_id, event_name, event_time)
```

Write a PostgreSQL query to calculate, by signup month and country, the 30-day paid activation rate and average first-month revenue.

Definitions:

- A user is in a cohort by `date_trunc('month', signup_date)`.
- Paid activation means the user starts any paid subscription within 30 days after signup.
- If a user has multiple subscriptions in the first 30 days, count the user once and use the earliest `started_at`.
- First-month revenue is the sum of `monthly_price_usd` for subscriptions started within 30 days after signup, per activated user.
- Output columns: `cohort_month`, `country`, `users_signed_up`, `activated_users`, `activation_rate`, `avg_first_month_revenue`.
- Avoid division by zero.

Also briefly explain two edge cases your query handles.

## Case 03: Privacy Incident Response

Classify this fictional privacy incident and propose a response plan.

Scenario:

A customer support export job accidentally included 4,200 rows of customer tickets in a CSV sent to an internal analytics mailing list. The CSV includes ticket ID, issue category, free-text ticket body, customer email, and account tier. No passwords, payment card numbers, government IDs, or authentication tokens were included. The mailing list has 38 employees and 2 contractors. The file was deleted from the mailing list archive after 90 minutes, but 11 recipients had already downloaded it.

Please provide:

1. Severity classification and reasoning.
2. Immediate containment actions in the first 24 hours.
3. Investigation questions.
4. Customer/legal/compliance communication considerations.
5. Preventive controls.

Do not claim a specific statutory notification deadline unless the scenario provides jurisdiction.

## Case 04: Vendor Contract Risk Memo

Review this fictional vendor clause and write a concise risk memo.

Clause:

"Vendor may process Customer Data using subprocessors selected at Vendor's sole discretion. Vendor will use commercially reasonable efforts to maintain availability. Customer's sole remedy for service interruption is a service credit capped at one month of fees. Vendor may use de-identified Customer Data to improve any Vendor product or service. Either party may terminate for convenience with 10 days' notice. Vendor's aggregate liability under this Agreement will not exceed fees paid in the prior one month."

Write the memo for a SaaS buyer. Include:

- Top five risks, ranked.
- Why each risk matters.
- Proposed redline or negotiation position for each risk.
- A final recommendation.

Do not invent governing law or say the clause is illegal.

## Case 05: Experiment Analysis

Analyze this product experiment.

An onboarding team tested a new checklist for new users.

Control:

- Users: 12,000
- Activated within 7 days: 3,000
- Median time to activation: 44 hours
- Support tickets per 1,000 users: 28

Treatment:

- Users: 11,800
- Activated within 7 days: 3,186
- Median time to activation: 39 hours
- Support tickets per 1,000 users: 35

Assume random assignment and no sample-ratio mismatch. Calculate activation rates, absolute lift, relative lift, and give a launch recommendation. Mention at least two risks or follow-up checks.
