# Titanic Data Analysis – Findings Summary

## Before / After Summary

| Metric               | Before Cleaning | After Cleaning |
|----------------------|----------------|----------------|
| **Rows**             | 891            | 891            |
| **Duplicates**       | 0              | 0              |
| **Nulls (total)**    | 866            | 0              |
| **Columns**          | 15             | 16 (+ 4 new features, -3 dropped) |
| **Data types**       | Mixed / object | All correct (numeric + cleaned strings) |

### Cleaning decisions
- **Age** → filled with median (29.7) – robust against outliers.
- **Embarked** → filled with mode ('S') – the most common port.
- **Cabin** → dropped original column (too many missing / unique), but created `has_cabin` flag (1 if cabin existed, else 0).
- **Deck** → dropped entirely (~77% missing).
- **Sex / Embarked** → standardised to lowercase / uppercase, trimmed whitespace.
- **All strings** → stripped of leading/trailing spaces.

---

## 5 Defensible Insights

1. **Gender gap** – Women survived at ~74%, men at ~19%. This is the strongest single predictor.
2. **Class divide** – 1st‑class passengers had ~63% survival vs ~24% for 3rd‑class. Socio‑economic status heavily influenced who got a lifeboat.
3. **Children first** – Children (<18) survived at ~60%, compared to ~38% for adults – the "women and children first" policy is clearly reflected.
4. **Family size matters** – Passengers with moderate family size (2–4) had better survival (~50%) than those alone (~30%) or those in large families (~20%). Small families likely stuck together and found seats, while large families were separated.
5. **Wealth = survival** – Fare correlates positively with survival (r ≈ 0.26). Higher fare meant better cabins and closer access to lifeboats.

---

## Ethics & PII Consideration

Although this is a historical, publicly available dataset, it still contains **names** and **ticket numbers**. In a professional setting (especially health, government, or finance), such columns would be treated as Personally Identifiable Information (PII). I would:

- Anonymise or remove `name`, `passenger_id`, and `ticket` before publishing.
- Ensure the cleaned dataset contains only aggregate or anonymised features.
- Never commit raw personal data to a public repository.

This judgment is part of the engineer's responsibility – not an afterthought.