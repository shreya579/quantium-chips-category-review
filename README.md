# 📦 Quantium — Chips Category Review

## Table of contents

1. [Project summary](#project-summary)  
2. [What’s included (files to upload)](#whats-included-files-to-upload)  
3. [Repository structure (recommended)](#repository-structure-recommended)  
4. [Key findings (executive highlights)](#key-findings-executive-highlights)  
5. [Strategic recommendations (summary)](#strategic-recommendations-summary)  
6. [Visuals & deliverables](#visuals--deliverables)  
7. [License](#license)  
8. [Appendix — Useful code snippets](#appendix---useful-code-snippets)

---

## Project summary

This repository contains the analysis and deliverables produced for the **Chips Category Review** prepared for Julia (Category Manager). The work answers two core questions:

- **Who buys chips and how do they shop?** — segment-level behaviour, pack-size and price-point preferences.  
- **Did the new layout trial work?** — evaluation of trial stores **77, 86 and 88** against matched control stores; uplift analysis and driver diagnostics.

Deliverables include reproducible Jupyter notebooks, cleaned data exports, visual assets and a client-ready presentation.

---

## What’s included (files to upload manually)

> Place the following files in the repository as-is (or upload them via the GitHub web UI). Keep filenames unchanged to preserve notebook paths.

```

data/
├─ QVI\_transaction\_data.xlsx
├─ QVI\_purchase\_behaviour.csv
└─ cleaned\_chip\_data.csv

notebooks/
├─ Report.ipynb
├─ Retail\_Analytics.ipynb
└─ Retail\_Strategy.ipynb

outputs/
└─ trial\_analysis\_plots.pdf

```



---

## Key findings — executive highlights

- **Primary customer contributors:** *Older Families (Budget)* and *Young Singles/Couples (Mainstream)* are the dominant drivers of chips volume and sales.
- **Pack-size behaviour:** Singles/Couples tend to purchase larger pack sizes (~185g). Families purchase more units per transaction (higher quantity).
- **Price preference:** Budget and Mainstream tiers account for the majority of sales; Premium appeals mainly to older singles and retirees.
- **Trial evaluation summary (Feb–Apr 2019):**
  - **Store 77**: ~**+5.2% uplift** in sales compared to its matched control; uplift primarily driven by **increased transactions** rather than increased customer reach.
  - **Stores 86 & 88**: No statistically meaningful uplift vs matched controls.
- **Primary implication:** A targeted rollout is recommended — expand the new layout to stores whose customer and sales profile closely match Store 77; defer or re-test in stores resembling 86 and 88.

---

## Strategic recommendations (summary)

1. **Rollout strategy**
   - Pilot rollout to a cohort of stores matched to Store 77 (demographic mix, life-stage distribution, affluence).
   - Ensure matched-control measurement and at least two quarters of post-rollout monitoring before broader rollout.

2. **Merchandising & pack strategy**
   - Introduce family-value bundles and multi-save packs targeted at Older & Young Families.
   - Offer premium, single-serve or large-pack lines positioned for Singles/Couples and Retirees.

3. **Marketing & personalization**
   - Leverage loyalty and life-stage signals to target promotions and communications.
   - Run A/B tests for promotional placement, pack sizes and pricing across matched stores.

4. **Analytics & measurement**
   - Continue matched-control experiments for any future layout changes.
   - Track both reach (unique customers) and intensity (transactions per customer) to surface the driver of any sales change.

---

## Visuals & deliverables

Place or generate the following assets in `outputs/`:

- `trial_analysis_plots.pdf` — trial vs control visualisations and diagnostics  
- `total_sales_by_segment.png` — sales by `LIFESTAGE × PREMIUM_CUSTOMER`  
- `average_pack_size_by_segment.png` — pack-size behaviour by segment  
- `Chips_Final_Report_Presentation.pptx` — client-ready presentation (if available)  
- `Chips_Category_Strategic_Report.docx` — formatted written report (if available)

_Note:_ keep visual style consistent across slides (colours, fonts, series ordering) for client-facing deliverables.

---

## License

This repository is provided under the **MIT License**. See `LICENSE` in the repository root for full terms.

---

## Appendix — Useful code snippets

### Convert Excel serial date to `datetime`
```python
import pandas as pd
df['DATE'] = pd.to_datetime(df['DATE'], origin='1899-12-30', unit='D')
````

### Extract pack size (grams) from product name

```python
df['PACK_SIZE'] = df['PROD_NAME'].str.extract(r'(\d+)\s?g', expand=False).astype(float)
```

### Compute monthly store metrics

```python
df['MONTH'] = df['DATE'].dt.to_period('M')
monthly_metrics = df.groupby(['STORE_NBR','MONTH']).agg(
    total_sales=('TOT_SALES','sum'),
    unique_customers=('LYLTY_CARD_NBR','nunique'),
    transactions=('TXN_ID','nunique')
).reset_index()
monthly_metrics['avg_txn_per_customer'] = monthly_metrics['transactions'] / monthly_metrics['unique_customers']
```

### Match control store by Pearson correlation (example)

```python
from scipy.stats import pearsonr

def match_control(trial_series, candidate_series):
    aligned = pd.concat([trial_series, candidate_series], axis=1).dropna()
    if len(aligned) < 6:
        return None
    corr, _ = pearsonr(aligned.iloc[:,0], aligned.iloc[:,1])
    return corr
```
