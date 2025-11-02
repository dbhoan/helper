import pandas as pd
import numpy as np

def adjust_monthly_shrinkage(
    df,
    special_dates,
    col_date="date",
    col_lob="LOB",
    col_shrinkage="shrinkage",
    col_offered="offered",
    col_aht="AHT",
    col_month="month",
    col_shrinkage_monthly="shrinkage_monthly",
    col_adjusted="adjusted_shrinkage"
):
    """
    Adjust shrinkage values to ensure:
      1. Special dates remain fixed (no adjustment)
      2. Monthly weighted average shrinkage equals shrinkage_monthly
      3. Weekly shrinkage pattern (relative ratios) is preserved

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns for shrinkage, weights, and month info.
    special_dates : list of pd.Timestamp
        Dates where shrinkage was manually overridden and must remain unchanged.
    col_* : str
        Customizable column names (defaults match previous implementation).

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'adjusted_shrinkage' column.
    """

    df = df.copy()
    df["__weight__"] = df[col_offered] * df[col_aht]
    df[col_adjusted] = df[col_shrinkage]

    def _adjust_group(g):
        # Compute target total weighted shrinkage for the group
        target_total = g[col_shrinkage_monthly].iloc[0] * g["__weight__"].sum()

        special_mask = g[col_date].isin(special_dates)
        special_total = (g.loc[special_mask, col_adjusted] * g.loc[special_mask, "__weight__"]).sum()
        regular_mask = ~special_mask

        remaining_weight = g.loc[regular_mask, "__weight__"].sum()
        if remaining_weight == 0:
            return g

        target_regular_total = target_total - special_total
        current_regular_total = (g.loc[regular_mask, col_shrinkage] * g.loc[regular_mask, "__weight__"]).sum()

        scale = target_regular_total / current_regular_total if current_regular_total != 0 else 1.0
        g.loc[regular_mask, col_adjusted] = g.loc[regular_mask, col_shrinkage] * scale
        return g

    df = df.groupby([col_lob, col_month], group_keys=False).apply(_adjust_group)
    df = df.drop(columns="__weight__")
    return df
