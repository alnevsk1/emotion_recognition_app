from typing import Dict, List, Union, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)

EmotionProbs = Dict[str, float]
Segment = Dict[str, Union[int, EmotionProbs]]

# Configuration constants for better maintainability
class FuzzyConfig:
    # Emotion to valence mapping (circumplex model inspired)
    VALENCE_MAP = {"angry": -0.8, "sad": -0.7, "neutral": 0.0, "positive": 0.8, "other": 0.0}
    
    # Emotion to arousal mapping
    AROUSAL_MAP = {"angry": 0.8, "sad": 0.2, "neutral": 0.3, "positive": 0.6, "other": 0.5}
    
    # Rule thresholds
    MIN_RULE_SCORE = 0.30
    HIGH_CONFIDENCE_THRESHOLD = 0.70
    DOMINANT_EMOTION_THRESHOLD = 0.60
    
    # Fuzzy function parameters
    DEFAULT_WIDTH = 0.2
    APPROX_EQUAL_REL = 0.15
    APPROX_EQUAL_ABS = 0.05
    
    # Segment validation
    MIN_SEGMENT_DURATION_MS = 1
    MAX_SEGMENT_DURATION_MS = 300000  # 5 minutes
    
    # Confidence calculation weights
    PEAK_WEIGHT = 0.6
    ENTROPY_WEIGHT = 0.4

def fuzzy_mood(
    data: Union[Segment, List[Segment]],
    return_details: bool = False,
) -> Union[str, Dict[str, object]]:
    """
    Compute an interpretable average mood over audio using fuzzy rules and a valence–arousal view.
    
    Input:
      - data: a dict {"start_ms": int, "end_ms": int, "probabilities": {"angry": float, "sad": float, "neutral": float, "positive": float, "other": float}}
               or a list of such dicts (e.g., multiple frames/chunks).
      - return_details: if True, returns a dict with diagnostics (valence, arousal, chosen rule, confidence).
    
    Output:
      - By default, a single mood phrase (str), e.g., "calm positive mood." or "negative background with irritation."
      - If return_details=True, returns a dict:
          {
            "mood": str,
            "valence": float in [-1,1],
            "arousal": float in [0,1],
            "confidence": float in [0,1],
            "avg_probs": dict,
            "fired_rule": str,
            "alternatives": List[Tuple[str, float]]
          }
    """
    
    # Input validation
    if not data:
        logger.warning("Empty data provided to fuzzy_mood")
        return _create_fallback_result(return_details)
    
    # Normalize input to a list
    segments = data if isinstance(data, list) else [data]
    
    # Validate segments
    if not segments or len(segments) == 0:
        logger.warning("No valid segments provided")
        return _create_fallback_result(return_details)

    # Ensure required keys and sane values
    def safe_probs(p: EmotionProbs) -> EmotionProbs:
        """Normalize and validate probability distributions."""
        if not p or not isinstance(p, dict):
            logger.warning("Invalid probability dict provided")
            return {k: 0.2 for k in ["angry", "sad", "neutral", "positive", "other"]}
            
        keys = ["angry", "sad", "neutral", "positive", "other"]
        q = {k: float(max(0.0, min(1.0, p.get(k, 0.0)))) for k in keys}
        s = sum(q.values())
        if s > 0:
            q = {k: v / s for k, v in q.items()}
        else:
            # Fallback to uniform distribution if all probabilities are 0
            q = {k: 0.2 for k in keys}
        return q

    # Duration-weighted pooling across segments
    total_w = 0.0
    acc = {k: 0.0 for k in ["angry", "sad", "neutral", "positive", "other"]}
    valid_segments = 0
    
    for seg in segments:
        if not isinstance(seg, dict):
            logger.warning(f"Invalid segment format: {seg}")
            continue
            
        start = int(seg.get("start_ms", 0))
        end = int(seg.get("end_ms", start))
        
        # Validate segment timing
        duration = end - start
        if duration <= 0:
            logger.warning(f"Invalid segment timing: start={start}, end={end}")
            continue
        
        if duration > FuzzyConfig.MAX_SEGMENT_DURATION_MS:
            logger.warning(f"Segment too long: {duration}ms, truncating")
            end = start + FuzzyConfig.MAX_SEGMENT_DURATION_MS
            
        w = max(FuzzyConfig.MIN_SEGMENT_DURATION_MS, duration)
        probs = safe_probs(seg.get("probabilities", {}))
        
        for k, v in probs.items():
            acc[k] += w * v
        total_w += w
        valid_segments += 1

    if total_w <= 0 or valid_segments == 0:
        logger.warning("No valid segments found for processing")
        return _create_fallback_result(return_details)

    avg = {k: acc[k] / total_w for k in acc.keys()}

    # Map emotions to valence [-1,1] and arousal [0,1] (circumplex-inspired)
    valence = sum(avg[k] * FuzzyConfig.VALENCE_MAP[k] for k in avg.keys())
    arousal = sum(avg[k] * FuzzyConfig.AROUSAL_MAP[k] for k in avg.keys())

    # Improved confidence calculation
    def entropy(p: Dict[str, float]) -> float:
        """Calculate Shannon entropy of probability distribution."""
        eps = 1e-12
        return -sum(pi * math.log(pi + eps) for pi in p.values() if pi > 0)

    def calculate_confidence(probs: Dict[str, float]) -> float:
        """Calculate confidence based on entropy and peak concentration."""
        H = entropy(probs)
        Hmax = math.log(len(probs))  # log 5
        peak = max(probs.values()) if probs else 0.0
        
        # Combine peak concentration and entropy-based uncertainty
        peak_confidence = peak
        entropy_confidence = 1.0 - (H / Hmax if Hmax > 0 else 1.0)
        
        # Weighted combination with emphasis on peak
        return FuzzyConfig.PEAK_WEIGHT * peak_confidence + FuzzyConfig.ENTROPY_WEIGHT * entropy_confidence

    confidence = calculate_confidence(avg)

    # Fuzzy helper functions
    def clip01(x: float) -> float:
        """Clip value to [0, 1] range."""
        return max(0.0, min(1.0, x))

    def soft_more(x: float, thresh: float, width: float = None) -> float:
        """Soft threshold: 0 at x<=thresh, 1 at x>=thresh+width."""
        if width is None:
            width = FuzzyConfig.DEFAULT_WIDTH
        return clip01((x - thresh) / max(1e-6, width))

    def soft_less(x: float, thresh: float, width: float = None) -> float:
        """Soft threshold: 1 at x<=thresh, 0 at x>=thresh+width."""
        if width is None:
            width = FuzzyConfig.DEFAULT_WIDTH
        return clip01((thresh + width - x) / max(1e-6, width))

    def approx_equal(a: float, b: float, rel: float = None, abs_tol: float = None) -> float:
        """Check if two values are approximately equal."""
        if rel is None:
            rel = FuzzyConfig.APPROX_EQUAL_REL
        if abs_tol is None:
            abs_tol = FuzzyConfig.APPROX_EQUAL_ABS
        denom = max(abs_tol, rel * (a + b) / 2.0 + 1e-6)
        return clip01(1.0 - abs(a - b) / denom)

    # Calculate derived features
    p = avg
    max_prob = max(p.values()) if p else 0.0
    pos_deg = clip01((valence + 1.0) / 2.0)  # 0..1
    neg_deg = 1.0 - pos_deg
    low_ar_deg = soft_less(arousal, 0.45, 0.25)
    hi_ar_deg = soft_more(arousal, 0.60, 0.20)

    # Define fuzzy rules with improved logic
    rules: List[Tuple[str, float]] = []

    # High confidence rules (dominant emotions)
    if max_prob > FuzzyConfig.DOMINANT_EMOTION_THRESHOLD:
        dominant_emotion = max(p.keys(), key=lambda k: p[k])
        if dominant_emotion == "positive":
            rules.append(("ярко выраженное позитивное настроение.", max_prob))
        elif dominant_emotion == "sad":
            rules.append(("явно подавленное настроение.", max_prob))
        elif dominant_emotion == "angry":
            rules.append(("четко выраженное раздражение.", max_prob))
        elif dominant_emotion == "neutral":
            rules.append(("спокойное, нейтральное настроение.", max_prob))

    # Complex mood rules
    rules.append((
        "раздраженное негативное настроение.",
        clip01(((p["sad"] + p["angry"]) - 0.50) / 0.20)
    ))

    rules.append((
        "спокойное, позитивное настроение.",
        min(approx_equal(p["neutral"], p["positive"]), low_ar_deg, pos_deg)
    ))

    rules.append((
        "преимущественно нейтральное настроение.",
        clip01((p["neutral"] - 0.60) / 0.20)
    ))

    rules.append((
        "бодрое, позитивное настроение.",
        min(clip01((p["positive"] - 0.60) / 0.20), hi_ar_deg)
    ))

    rules.append((
        "подавленное, унылое настроение.",
        min(clip01((p["sad"] - 0.40) / 0.20), soft_less(arousal, 0.40, 0.20))
    ))

    rules.append((
        "напряженное и раздражительное настроение.",
        min(clip01((p["angry"] - 0.40) / 0.20), hi_ar_deg)
    ))

    rules.append((
        "смешанные противоречивые эмоции.",
        min(clip01((p["positive"] - 0.30) / 0.20), clip01((p["sad"] - 0.30) / 0.20))
    ))

    rules.append((
        "оптимистичное настроение с ноткой напряжения.",
        min(clip01((p["positive"] - 0.50) / 0.20), clip01((p["angry"] - 0.25) / 0.20))
    ))

    rules.append((
        "теплое, спокойное и уравновешенное настроение.",
        min(clip01((p["positive"] - 0.40) / 0.20), clip01((p["neutral"] - 0.30) / 0.20), 
            clip01((0.30 - (p["sad"] + p["angry"])) / 0.20))
    ))

    rules.append((
        "неопределенное настроение с разными эмоциями.",
        max(clip01((p["other"] - 0.50) / 0.20), clip01((0.35 - max_prob) / 0.15))
    ))

    # Pick best-scoring rule
    rules_sorted = sorted(rules, key=lambda x: x[1], reverse=True)
    best_label, best_score = rules_sorted[0]

    if best_score < FuzzyConfig.MIN_RULE_SCORE:
        if valence > 0.20:
            best_label = "в целом положительное настроение."
        elif valence < -0.20:
            best_label = "в целом грустное настроение."
        else:
            best_label = "в целом нейтральное настроение."

    if return_details:
        return {
            "mood": best_label,
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "confidence": round(confidence, 4),
            "avg_probs": {k: round(v, 4) for k, v in avg.items()},
            "fired_rule": best_label,
            "alternatives": [{"label": lab, "score": round(sc, 4)} for lab, sc in rules_sorted[1:4]],
        }
    return best_label


def _create_fallback_result(return_details: bool) -> Union[str, Dict[str, object]]:
    """Create fallback result for error cases."""
    fallback_mood = "неопределенное настроение."
    if return_details:
        return {
            "mood": fallback_mood,
            "valence": 0.0,
            "arousal": 0.5,
            "confidence": 0.0,
            "avg_probs": {},
            "fired_rule": "fallback",
            "alternatives": []
        }
    return fallback_mood


def validate_emotion_model_compatibility(emotion_labels: List[str]) -> bool:
    required_emotions = set(FuzzyConfig.VALENCE_MAP.keys())
    model_emotions = set(emotion_labels)
    
    if not required_emotions.issubset(model_emotions):
        missing = required_emotions - model_emotions
        logger.error(f"Missing required emotions: {missing}")
        return False
    
    return True


def get_emotion_statistics(segments: List[Segment]) -> Dict[str, float]:
    if not segments:
        return {}
    
    # Calculate average probabilities
    total_weight = 0.0
    emotion_totals = {k: 0.0 for k in FuzzyConfig.VALENCE_MAP.keys()}
    
    for seg in segments:
        if not isinstance(seg, dict):
            continue
            
        start = seg.get("start_ms", 0)
        end = seg.get("end_ms", start)
        duration = max(1, end - start)
        
        probs = seg.get("probabilities", {})
        for emotion in emotion_totals.keys():
            emotion_totals[emotion] += probs.get(emotion, 0.0) * duration
        
        total_weight += duration
    
    if total_weight > 0:
        return {k: v / total_weight for k, v in emotion_totals.items()}
    
    return {k: 0.0 for k in emotion_totals.keys()}
