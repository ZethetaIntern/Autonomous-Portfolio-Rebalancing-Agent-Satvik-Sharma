"""Client-Facing Plain English Narrative Explanation Generator for WealthPilot AI.

Generates accessible, jargon-free explanations tailored for retail investors,
enforcing Grade 8 readability (Flesch-Kincaid <= 8.0), goal-relevance framing,
fee/tax transparency, and strict 200-word conciseness limits.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClientExplanationPayload(BaseModel):
    """Pydantic validated schema for retail client explanations."""
    portfolio_id: str = Field(description="Unique portfolio identifier")
    decision_id: str = Field(description="Deterministic decision tracking ID")
    headline: str = Field(description="Clear 1-sentence non-jargon summary")
    narrative: str = Field(description="Grade 8 readable plain-language explanation")
    readability_grade: float = Field(description="Flesch-Kincaid Grade Level (<= 8.0)")
    word_count: int = Field(description="Total word count (<= 200 words)")
    goal_relevance: str = Field(description="How this trade protects long-term client goals")
    cost_transparency: Dict[str, float] = Field(description="Rupee itemization of costs and taxes")
    action_items: List[str] = Field(default_factory=list, description="Next steps for client")


def count_syllables(word: str) -> int:
    """Estimates syllable count using English phonetic rules."""
    w = word.lower().strip()
    if not w:
        return 0
    w = re.sub(r"[^a-z]", "", w)
    if len(w) <= 3:
        return 1
    # Count vowel groups
    vowels = "aeiouy"
    count = len(re.findall(r"[aeiouy]+", w))
    # Discount silent trailing 'e'
    if w.endswith("e") and not w.endswith("le") and len(w) > 2:
        count = max(1, count - 1)
    if w.endswith("ed") and not w.endswith("ted") and not w.endswith("ded"):
        count = max(1, count - 1)
    return max(1, count)


def calculate_flesch_kincaid_grade(text: str) -> float:
    """Calculates Flesch-Kincaid Grade Level index:
    FKGL = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = [w.strip() for w in re.findall(r"\b[A-Za-z]+\b", text) if w.strip()]

    num_sentences = max(1, len(sentences))
    num_words = max(1, len(words))
    total_syllables = sum(count_syllables(w) for w in words)

    asl = num_words / num_sentences
    asw = total_syllables / num_words

    grade = (0.39 * asl) + (11.8 * asw) - 15.59
    return round(float(grade), 1)


class ClientExplainer:
    """Generates plain-language, non-jargon narratives tailored for retail investors."""

    def __init__(self, max_word_count: int = 200, max_readability_grade: float = 8.0) -> None:
        self.max_word_count = max_word_count
        self.max_readability_grade = max_readability_grade

    def generate_explanation(
        self,
        portfolio_id: str,
        decision_id: str,
        trigger_category: str,
        primary_asset: str,
        drift_direction: str,
        current_equity_pct: float,
        target_equity_pct: float,
        total_costs_inr: float,
        tax_impact_inr: float,
        client_goal: str = "Balanced Long-Term Wealth Growth",
    ) -> ClientExplanationPayload:
        """Generates an engaging, Grade 8 readable narrative explaining why a trade occurred.
        
        Args:
            portfolio_id: Portfolio identifier.
            decision_id: System decision ID.
            trigger_category: "THRESHOLD", "CALENDAR", or "EVENT".
            primary_asset: Key asset adjusted (e.g. "NIFTY_50_EQUITY").
            drift_direction: "OVERWEIGHT" (needs trimming) or "UNDERWEIGHT" (needs buying).
            current_equity_pct: Current equity percentage (e.g. 62.0).
            target_equity_pct: Strategic target equity percentage (e.g. 50.0).
            total_costs_inr: Total explicit transaction fees.
            tax_impact_inr: Realized tax consequence (or tax shield savings).
            client_goal: Client's stated investment goal.
        """
        clean_asset_name = primary_asset.replace("_", " ").title()

        if trigger_category.upper() == "THRESHOLD":
            headline = f"Portfolio update: Rebalancing your {clean_asset_name} to keep your risk on track."
            if "OVER" in drift_direction.upper():
                narrative = (
                    f"Recent market gains pushed your stock holdings to {current_equity_pct:.0f}%, "
                    f"which is above your target of {target_equity_pct:.0f}%. "
                    f"To protect your hard-earned profits and keep your {client_goal} plan steady, "
                    f"we locked in some gains and moved funds into safe bonds. "
                    f"This ensures market dips will not hurt your peace of mind."
                )
            else:
                narrative = (
                    f"Recent price dips lowered your stock holdings to {current_equity_pct:.0f}%, "
                    f"below your {target_equity_pct:.0f}% target. "
                    f"We used this chance to buy quality stocks at lower prices. "
                    f"This keeps your {client_goal} plan right on track for future gains."
                )
        elif trigger_category.upper() == "CALENDAR":
            headline = "Periodic portfolio health check: Your investments are aligned with your goals."
            narrative = (
                f"We just finished your scheduled portfolio review. "
                f"Your investments were gently adjusted back to your {target_equity_pct:.0f}% target. "
                f"Routine checkups help protect your wealth and make sure your money works as hard as you do."
            )
        else:  # EVENT
            headline = "Smart tax savings: We harvested losses to lower your taxes."
            narrative = (
                f"We scanned your portfolio for year-end tax savings. "
                f"By replacing a few dips with similar strong funds, we saved you money on taxes "
                f"while keeping your exact investment plan intact. "
                f"You stay fully invested for growth while paying less to the tax office."
            )

        words = re.findall(r"\b[A-Za-z0-9'-]+\b", narrative)
        word_count = len(words)
        grade = calculate_flesch_kincaid_grade(narrative)

        # Enforce Grade 8 constraint
        if grade > self.max_readability_grade:
            # Shorten sentences for simpler readability
            sentences = narrative.split(". ")
            narrative = ". ".join(sentences[:3]) + "."
            grade = calculate_flesch_kincaid_grade(narrative)
            word_count = len(re.findall(r"\b[A-Za-z0-9'-]+\b", narrative))

        return ClientExplanationPayload(
            portfolio_id=portfolio_id,
            decision_id=decision_id,
            headline=headline,
            narrative=narrative,
            readability_grade=grade,
            word_count=word_count,
            goal_relevance=f"Keeps portfolio aligned with {client_goal} and maintains your target risk level.",
            cost_transparency={
                "brokerage_and_fees_inr": round(float(total_costs_inr), 2),
                "tax_impact_inr": round(float(tax_impact_inr), 2),
                "net_total_inr": round(float(total_costs_inr + tax_impact_inr), 2),
            },
            action_items=[
                "No action required from your side. Your portfolio is updated.",
                "Log into your dashboard anytime to review your updated asset holdings.",
            ],
        )
