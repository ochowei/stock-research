# EXP-24 Review: Gap Quality Filter (Volume & Context)

## 1. Executive Summary
*   **Status**: ✅ **Success** (Significant Insight Found)
*   **Primary Finding**: The hypothesis that large gaps (>2%) are "Breakaway Gaps" to be avoided was **firmly rejected**. In fact, **Large Gaps (>3%)** provided the highest performance (Win Rate 57.3%, Avg Return 0.68%), suggesting they are powerful mean-reversion setups (likely "Exhaustion Gaps" or simply offer better risk/reward).
*   **Secondary Finding**: Extreme relative volume (>3.0x Previous Day) does degrade performance (48.4% Win Rate), supporting the idea that massive volume indicates momentum continuation to some extent.

## 2. Detailed Results (Test Period: 2024-2025)

### 2.1 Baseline Performance
*   **Strategy**: V6.2.4.RC (Base + Context Features)
*   **Win Rate**: 50.21%
*   **Avg Return**: +0.18%
*   **Signal Count**: 8,947
*   *Note*: The baseline performance on 2024-2025 data is lower than the historical training performance (~53%), indicating a harder market regime.

### 2.2 Gap Size Analysis (The "Breakaway" Myth)
Contrary to the hypothesis, performance **improves** as Gap Size increases.

| Gap Size Bin | Win Rate | Avg Return | Count | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **0.5% - 1.0%** | 48.03% | +0.04% | 2,942 | Low quality, noise. |
| **1.0% - 2.0%** | 48.35% | +0.06% | 3,065 | Mediocre. |
| **2.0% - 3.0%** | 50.54% | +0.13% | 1,286 | Improved. |
| **> 3.0%** | **57.29%** | **+0.68%** | 1,653 | **Superior Alpha.** |

*   **Insight**: We were "cutting our winners" by fearing large gaps. Gaps > 3% are not trends to be feared; they are over-extensions to be shorted.

### 2.3 Volume Ratio Analysis
| Vol Ratio (vs MA20) | Win Rate | Avg Return | Count | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **< 1.0x** (Low) | 48.99% | +0.12% | 5,170 | Low conviction. |
| **1.0x - 2.0x** | 51.92% | +0.26% | 3,120 | **Sweet Spot.** |
| **2.0x - 3.0x** | **53.21%** | **+0.39%** | 436 | High conviction. |
| **> 3.0x** (Extreme) | 48.42% | -0.12% | 221 | **Danger Zone.** |

*   **Insight**: Momentum/Breakaway risk exists primarily in **Extreme Volume (>3x)** scenarios, not just "High" volume. Moderate-High volume (1x-3x) confirms the setup.

## 3. Conclusion & Recommendations
1.  **Reject** the filter "Gap < 2%". Implementing this would reduce Win Rate to 48.2%.
2.  **Adopt** a preference for **Large Gaps**. The model feature `Gap_Pct` is already monotonic (larger = better score usually), but we should ensure we don't artificially cap it in production safety checks.
3.  **Consider** a hard filter for **Extreme Volume (>3.0x)**, although it affects only ~2% of trades.
4.  **Strategic Shift**: The "Gap Quality" is defined by **Magnitude** (bigger is better) and **Volume Moderation** (avoid extreme outliers).

## 4. Next Steps
*   Update `global_learning_log` with the "Large Gap" discovery.
*   Investigate if we can explicitly target the >3% Gap segment with a specialized model or simply trust the existing Ensemble to weight it heavily.
*   **New Experiment Idea**: "Large Gap Specialization" - Can we train a model *specifically* on Gaps > 2% to push the WR from 57% to 60%+?
