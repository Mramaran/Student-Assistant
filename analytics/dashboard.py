"""
analytics/dashboard.py — Dashboard Data Aggregator
==================================================
Calculates high-level metrics for the Streamlit dashboard.
"""

from typing import Dict, Any
from feedback import analytics, impact, tutor_ranker

class DashboardLogic:
    @staticmethod
    def get_summary_metrics() -> Dict[str, Any]:
        """
        Returns a flat dict of key metrics for UI cards.
        """
        summ = analytics.summary()
        return {
            "total_queries": summ.get("total", 0),
            "usi": summ.get("user_satisfaction_index", 0.0),
            "match_success": summ.get("match_success_rate", 0.0),
            "avg_latency": summ.get("avg_latency", 0.0)
        }

    @staticmethod
    def get_roi_data() -> Dict[str, Any]:
        """
        Returns ROI / Before-After comparison data.
        """
        return impact.generate_before_after_summary()

    @staticmethod
    def get_leaderboard() -> list:
        """
        Returns the ranked list of tutors.
        """
        report = tutor_ranker.get_ranking_report()
        return report.get("tutor_rankings", [])
