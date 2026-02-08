"""Jiwa — AI relapse predictor berbasis digital phenotyping.

Prototype ini mencontohkan 10 fitur utama untuk memprediksi risiko kekambuhan
(6-12 bulan) pada pasien bipolar/psikosis yang stabil pasca rawat inap.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Dict


@dataclass
class PatientSnapshot:
    # Data linguistik (chat/voice note)
    baseline_speech_rate_wpm: float
    current_speech_rate_wpm: float
    baseline_pause_seconds: float
    current_pause_seconds: float
    baseline_emotion_valence: float  # -1 (negatif) s.d. +1 (positif)
    current_emotion_valence: float
    baseline_disorganization: float  # 0 (terstruktur) s.d. 1 (sangat tidak terstruktur)
    current_disorganization: float

    # Data perilaku (tidur/aktivitas)
    baseline_sleep_hours: float
    current_sleep_hours: float
    sleep_variability_hours: float  # SD tidur 14 hari
    baseline_activity_steps: float
    current_activity_steps: float
    circadian_shift_hours: float  # pergeseran midpoint tidur terhadap baseline

    # Data klinis/adhesi
    medication_adherence_ratio: float  # 0..1
    missed_followup_ratio: float  # 0..1

    # Data engagement digital
    baseline_daily_messages: float
    current_daily_messages: float


class JiwaRelapsePredictor:
    """Model rule-based ringan untuk triase awal risiko relapse."""

    def __init__(self) -> None:
        # Bobot fitur untuk komputasi risiko relapse
        self.weights = {
            "speech_rate_shift": 1.1,
            "pause_change": 0.6,
            "emotion_valence_drop": 1.2,
            "language_disorganization_rise": 1.4,
            "sleep_deviation": 1.1,
            "sleep_irregularity": 0.9,
            "activity_shift": 0.7,
            "circadian_disruption": 1.0,
            "medication_nonadherence": 1.5,
            "digital_withdrawal": 0.8,
        }

    @staticmethod
    def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _relative_change(current: float, baseline: float) -> float:
        if baseline == 0:
            return 0.0
        return abs(current - baseline) / abs(baseline)

    def extract_10_features(self, p: PatientSnapshot) -> Dict[str, float]:
        """10 fitur digital phenotyping utama."""
        speech_rate_shift = self._bounded(
            self._relative_change(p.current_speech_rate_wpm, p.baseline_speech_rate_wpm)
        )
        pause_change = self._bounded(
            self._relative_change(p.current_pause_seconds, p.baseline_pause_seconds)
        )
        emotion_valence_drop = self._bounded(
            max(0.0, p.baseline_emotion_valence - p.current_emotion_valence) / 2.0
        )
        language_disorganization_rise = self._bounded(
            max(0.0, p.current_disorganization - p.baseline_disorganization)
        )
        sleep_deviation = self._bounded(
            abs(p.current_sleep_hours - p.baseline_sleep_hours) / 4.0
        )
        sleep_irregularity = self._bounded(p.sleep_variability_hours / 3.0)
        activity_shift = self._bounded(
            self._relative_change(p.current_activity_steps, p.baseline_activity_steps)
        )
        circadian_disruption = self._bounded(abs(p.circadian_shift_hours) / 4.0)
        medication_nonadherence = self._bounded(1.0 - p.medication_adherence_ratio)
        digital_withdrawal = self._bounded(
            max(0.0, p.baseline_daily_messages - p.current_daily_messages)
            / max(1.0, p.baseline_daily_messages)
        )

        return {
            "speech_rate_shift": speech_rate_shift,
            "pause_change": pause_change,
            "emotion_valence_drop": emotion_valence_drop,
            "language_disorganization_rise": language_disorganization_rise,
            "sleep_deviation": sleep_deviation,
            "sleep_irregularity": sleep_irregularity,
            "activity_shift": activity_shift,
            "circadian_disruption": circadian_disruption,
            "medication_nonadherence": medication_nonadherence,
            "digital_withdrawal": digital_withdrawal,
        }

    def relapse_probability_6_12_months(self, features: Dict[str, float]) -> float:
        """Probabilitas kekambuhan 6-12 bulan (0..1)."""
        score = sum(features[k] * self.weights[k] for k in self.weights)
        calibrated = score - 3.5
        return 1 / (1 + exp(-calibrated))

    def estimate_outcomes(self, p: PatientSnapshot) -> Dict[str, float]:
        """Outcome utama sesuai PICO: relapse, readmission, intervensi dini, adherence, false alarm."""
        features = self.extract_10_features(p)
        relapse_risk = self.relapse_probability_6_12_months(features)

        # Readmission risk diasumsikan subset dari relapse risk
        readmission_risk = self._bounded(relapse_risk * 0.72 + 0.08)

        # Waktu intervensi dini (hari): makin tinggi risiko, makin singkat jendela aman
        early_intervention_window_days = max(3.0, 60.0 * (1.0 - relapse_risk))

        # Prediksi kepatuhan obat 3 bulan ke depan
        predicted_med_adherence_3m = self._bounded(
            p.medication_adherence_ratio - 0.25 * relapse_risk - 0.15 * p.missed_followup_ratio
        )

        # Estimasi false alarm rate (konservatif): naik saat cutoff sensitif
        # Di sini diasumsikan mode sensitif saat risiko > 0.55
        false_alarm_rate = 0.18 if relapse_risk > 0.55 else 0.10

        return {
            "relapse_risk_6_12m": relapse_risk,
            "readmission_risk": readmission_risk,
            "early_intervention_window_days": early_intervention_window_days,
            "predicted_medication_adherence_3m": predicted_med_adherence_3m,
            "estimated_false_alarm_rate": false_alarm_rate,
            "features": features,
        }

    def compare_with_routine_followup(self, p: PatientSnapshot) -> Dict[str, float]:
        """Perbandingan kasar AI monitoring vs follow-up rutin tanpa AI."""
        ai = self.estimate_outcomes(p)

        # Hipotesis: tanpa AI, deteksi lebih lambat + readmission lebih tinggi
        routine_relapse_risk = self._bounded(ai["relapse_risk_6_12m"] + 0.08)
        routine_readmission_risk = self._bounded(ai["readmission_risk"] + 0.12)
        routine_intervention_window_days = max(1.0, ai["early_intervention_window_days"] - 14.0)

        return {
            "ai_relapse_risk_6_12m": ai["relapse_risk_6_12m"],
            "routine_relapse_risk_6_12m": routine_relapse_risk,
            "ai_readmission_risk": ai["readmission_risk"],
            "routine_readmission_risk": routine_readmission_risk,
            "ai_early_intervention_window_days": ai["early_intervention_window_days"],
            "routine_early_intervention_window_days": routine_intervention_window_days,
            "ai_estimated_false_alarm_rate": ai["estimated_false_alarm_rate"],
        }


def demo() -> None:
    patient = PatientSnapshot(
        baseline_speech_rate_wpm=120,
        current_speech_rate_wpm=165,
        baseline_pause_seconds=1.2,
        current_pause_seconds=0.5,
        baseline_emotion_valence=0.1,
        current_emotion_valence=-0.4,
        baseline_disorganization=0.22,
        current_disorganization=0.55,
        baseline_sleep_hours=7.0,
        current_sleep_hours=4.5,
        sleep_variability_hours=2.1,
        baseline_activity_steps=7000,
        current_activity_steps=3200,
        circadian_shift_hours=2.8,
        medication_adherence_ratio=0.78,
        missed_followup_ratio=0.30,
        baseline_daily_messages=35,
        current_daily_messages=12,
    )

    predictor = JiwaRelapsePredictor()
    outcomes = predictor.estimate_outcomes(patient)
    comparison = predictor.compare_with_routine_followup(patient)

    print("=== 10 Fitur Digital Phenotyping ===")
    for name, value in outcomes["features"].items():
        print(f"- {name}: {value:.3f}")

    print("\n=== Outcome AI Monitoring ===")
    for k, v in outcomes.items():
        if k != "features":
            print(f"- {k}: {v:.3f}")

    print("\n=== Perbandingan vs Follow-up Rutin ===")
    for k, v in comparison.items():
        print(f"- {k}: {v:.3f}")


if __name__ == "__main__":
    demo()
