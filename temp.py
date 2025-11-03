import pandas as pd
import numpy as np

def adjust_monthly_shrinkage(
    df,
    col_date="date",
    col_lob="LOB",
    col_shrinkage="shrinkage",
    col_offered="offered",
    col_aht="AHT",
    col_month="month",
    col_shrinkage_monthly="shrinkage_monthly",
    col_special_flag="special_date",
    col_adjusted="adjusted_shrinkage"
):
    """
    Adjust shrinkage values to ensure:
      1. Special dates (where col_special_flag == 1) remain fixed
      2. Monthly weighted average shrinkage equals shrinkage_monthly
      3. Weekly shrinkage pattern (relative ratios) is preserved

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns for shrinkage, weights, month info, and special flag.
    col_* : str
        Customizable column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'adjusted_shrinkage' column.
    """

    df = df.copy()
    df["__weight__"] = df[col_offered] * df[col_aht]
    df[col_adjusted] = df[col_shrinkage]

    def _adjust_group(g):
        # target total weighted shrinkage for this (LOB, month)
        target_total = g[col_shrinkage_monthly].iloc[0] * g["__weight__"].sum()

        # identify special and regular rows
        special_mask = g[col_special_flag].fillna(0) == 1
        regular_mask = ~special_mask

        # weighted totals
        special_total = (g.loc[special_mask, col_adjusted] * g.loc[special_mask, "__weight__"]).sum()
        current_regular_total = (g.loc[regular_mask, col_shrinkage] * g.loc[regular_mask, "__weight__"]).sum()
        remaining_weight = g.loc[regular_mask, "__weight__"].sum()

        if remaining_weight == 0 or current_regular_total == 0:
            return g

        # compute target total for regulars
        target_regular_total = target_total - special_total

        # proportional scaling factor
        scale = target_regular_total / current_regular_total

        # apply scaling only to non-special days
        g.loc[regular_mask, col_adjusted] = g.loc[regular_mask, col_shrinkage] * scale
        return g

    df = df.groupby([col_lob, col_month], group_keys=False).apply(_adjust_group)
    df = df.drop(columns="__weight__")

    return df


def adjust_monthly_offered(
    df,
    col_lob="LOB",
    col_offered="offered",
    col_month="month",
    col_monthly_offered="monthly_offered",
    col_adjusted="adjusted_offered"
):
    """
    Adjust offered values to ensure:
      1. Monthly total offered equals 'monthly_offered'
      2. Daily distribution pattern (ratios between weekdays) is preserved
      3. No special-date exceptions are applied

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns for offered, month, and LOB.
    col_* : str
        Customizable column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'adjusted_offered' column.
    """

    df = df.copy()
    df[col_adjusted] = df[col_offered]

    def _adjust_group(g):
        # Target monthly total for this (LOB, month)
        target_total = g[col_monthly_offered].iloc[0]
        current_total = g[col_offered].sum()

        if current_total == 0:
            return g  # nothing to adjust

        # Scaling factor to match target total
        scale = target_total / current_total

        # Apply proportional adjustment
        g[col_adjusted] = g[col_offered] * scale
        return g

    # Apply adjustment for each LOB × month
    df = df.groupby([col_lob, col_month], group_keys=False).apply(_adjust_group)

    return df
