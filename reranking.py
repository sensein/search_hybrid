# -*- coding: utf-8 -*-
"""
Re-ranking module: Apply multiple re-rankers to refine candidate set
- LLM-based reranker (OpenRouter API for semantic similarity)
- Late-Interaction (ColBERT-like token-level matching)
- Biomedical-context reranker (domain-specific term boosting)

Configuration via environment variables or API parameters:
- RERANKER_TYPE: comma-separated list or single reranker type
  Options: 'llm', 'late_interaction', 'biomedical', 'ensemble'
  Default: 'ensemble'

- LLM_WEIGHT: weight for LLM reranker (default: 0.5)
- LATE_INTERACTION_WEIGHT: weight for late-interaction (default: 0.3)
- BIOMEDICAL_WEIGHT: weight for biomedical context (default: 0.2)

- OPENROUTER_API_KEY: OpenRouter API key (can be passed via environment or API request)
- OPENROUTER_MODEL: Model name from OpenRouter (default: 'openrouter/auto')
  Examples: 'google/gemini-2.0-flash-001', 'anthropic/claude-3.5-sonnet:beta', 'meta-llama/llama-2-70b-chat'

- LATE_INTERACTION_MODEL: Hugging Face model name (default: 'jinaai/jina-colbert-v2')
"""

import logging
import os
import json
import time
import requests
from typing import Any, List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


def _get_reranker_config(openrouter_api_key: Optional[str] = None) -> Dict[str, any]:
    """Read reranker configuration from environment variables or parameters
    
    Args:
        openrouter_api_key: Optional OpenRouter API key (overrides environment)
        
    Returns:
        Configuration dictionary with reranker settings
    """
    api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
    
    return {
        "type": os.getenv("RERANKER_TYPE", "ensemble").lower(),
        "llm_weight": float(os.getenv("LLM_WEIGHT", "0.5")),
        "late_interaction_weight": float(os.getenv("LATE_INTERACTION_WEIGHT", "0.3")),
        "biomedical_weight": float(os.getenv("BIOMEDICAL_WEIGHT", "0.2")),
        "openrouter_api_key": api_key,
        "openrouter_model": os.getenv("OPENROUTER_MODEL", "openrouter/auto"),
        "late_interaction_model": os.getenv("LATE_INTERACTION_MODEL", "jinaai/jina-colbert-v2"),
    }


@dataclass
class RerankingResult:
    """Result from re-ranking with scores from multiple models"""
    class_uri: str
    preferred_label: str
    ontology_id: str
    original_score: float = 0.0
    llm_score: float = 0.0
    late_interaction_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> Dict:
        return {
            "ontology_id": self.class_uri,
            "preferred_label": self.preferred_label,
            "ontology": self.ontology_id,
            "original_score": float(self.original_score),
            "llm_score": float(self.llm_score),
            "late_interaction_score": float(self.late_interaction_score),
            "final_score": float(self.final_score),
            "rank": self.rank,
        }


class LLMReranker:
    """LLM-based re-ranking using OpenRouter API for semantic similarity scoring
    
    Supports dynamic API key passing via constructor, environment variables, or API requests.
    """

    def __init__(self, api_key: str = "", model_name: str = "openrouter/auto"):
        """
        Initialize LLM reranker
        
        Args:
            api_key: OpenRouter API key (can be empty, uses env var as fallback)
            model_name: Model name from OpenRouter
              Examples:
                - 'openrouter/auto' (default, auto-selects best available)
                - 'google/gemini-2.0-flash-001'
                - 'anthropic/claude-3.5-sonnet:beta'
                - 'meta-llama/llama-2-70b-chat'
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model_name = model_name
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("LLMReranker initialized without API key - scoring will fail at runtime")
        else:
            logger.info(f"LLMReranker initialized with model: {model_name}")

    def _score_candidate_pair(self, query: str, candidate_label: str, api_key: str, model_name: str) -> float:
        """
        Score a candidate using LLM via OpenRouter

        Args:
            query: Query text
            candidate_label: Candidate label to score
            api_key: OpenRouter API key to use
            model_name: OpenRouter model to use

        Returns:
            Similarity score between 0 and 1
        """
        if not api_key:
            logger.warning("Cannot score without API key")
            return 0.0

        try:
            prompt = f"""Rate the semantic similarity between these two biomedical terms on a scale of 0 to 1.
Respond with ONLY a single decimal number between 0 and 1.

Query: {query}
Candidate: {candidate_label}

Similarity score:"""

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 10
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"].strip()
                try:
                    score = float(content)
                    return max(0.0, min(1.0, score))  # Clamp to [0, 1]
                except ValueError:
                    logger.warning(f"Could not parse score from LLM response: {content}")
                    return 0.0
            return 0.0
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request error: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"LLM scoring error: {e}")
            return 0.0

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """
        Re-rank candidates using LLM via OpenRouter

        Args:
            query: Query text
            candidates: List of candidate dicts with 'preferred_label' and optional 'definition'
            top_k: Return only top-k results
            api_key: Override OpenRouter API key (takes priority over instance key)
            model_name: Override OpenRouter model (takes priority over instance model)

        Returns:
            List of (candidate_index, score) tuples, sorted by score descending
        """
        effective_api_key = api_key or self.api_key
        effective_model = model_name or self.model_name

        if not candidates or not effective_api_key:
            return [(i, 0.0) for i in range(len(candidates))]

        try:
            from concurrent.futures import ThreadPoolExecutor

            def _score(i: int, candidate: Dict[str, Any]):
                label = candidate.get("preferred_label", "")
                if not label:
                    return i, 0.0
                evidence = label
                if candidate.get("definition"):
                    evidence += " - " + candidate["definition"]
                return i, self._score_candidate_pair(query, evidence, effective_api_key, effective_model)

            max_workers = min(len(candidates), 8)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_score, i, c) for i, c in enumerate(candidates)]
                results = [f.result() for f in futures]

            results.sort(key=lambda x: x[1], reverse=True)

            if top_k:
                results = results[:top_k]

            logger.debug(f"LLM reranking complete: {len(results)} candidates")
            return results
        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            return [(i, 0.0) for i in range(len(candidates))]


class LateInteractionReranker:
    """Late-Interaction (ColBERT-like) re-ranking"""

    def __init__(self, model_name: str = "jinaai/jina-colbert-v2"):
        """
        Initialize late-interaction reranker
        
        Args:
            model_name: Hugging Face model name (ColBERT or derivative)
        """
        self.tokenizer = None
        self.model = None
        self.model_name = model_name
        logger.info(f"LateInteractionReranker initialized (token-level matching, no model load)")

    def _get_late_interaction_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """
        Compute simple late-interaction score (MaxSim)
        This is a simplified version; production would use more sophisticated matching
        """
        score = 0.0
        for q_token in query_tokens:
            best_match = 0.0
            for d_token in doc_tokens:
                # Simple token-level similarity (would use embedding similarity in production)
                if q_token.lower() == d_token.lower():
                    best_match = 1.0
                    break
            score += best_match
        return score / len(query_tokens) if query_tokens else 0.0

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Re-rank using late-interaction scoring
        
        Args:
            query: Query text
            candidates: List of candidate dicts
            top_k: Return only top-k results
            
        Returns:
            List of (candidate_index, score) tuples, sorted by score descending
        """
        if not candidates:
            return []

        try:
            query_tokens = query.lower().split()
            results = []
            
            for i, candidate in enumerate(candidates):
                evidence = candidate.get("preferred_label", "")
                if candidate.get("definition"):
                    evidence += " " + candidate["definition"]
                
                doc_tokens = evidence.lower().split()
                score = self._get_late_interaction_score(query_tokens, doc_tokens)
                results.append((i, float(score)))
            
            results.sort(key=lambda x: x[1], reverse=True)
            
            if top_k:
                results = results[:top_k]
            
            logger.debug(f"Late-interaction reranking complete: {len(results)} candidates")
            return results
        except Exception as e:
            logger.error(f"Late-interaction reranking failed: {e}")
            return [(i, 0.0) for i in range(len(candidates))]


class BiomedicalContextReranker:
    """Reranker that boosts biomedical terms and context"""

    def __init__(self):
        """Initialize biomedical context reranker"""
        # Biomedical term weights (can be expanded)
        self.biomedical_keywords = {
            "disease": 1.2,
            "disorder": 1.2,
            "syndrome": 1.1,
            "cancer": 1.3,
            "diabetes": 1.2,
            "therapy": 1.1,
            "treatment": 1.1,
            "drug": 1.1,
            "medication": 1.1,
            "protein": 1.2,
            "gene": 1.2,
            "mutation": 1.2,
            "anatomical": 1.1,
            "pathway": 1.1,
        }
        logger.info("BiomedicalContextReranker initialized")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Re-rank with biomedical context boost
        
        Args:
            query: Query text (may be biomedical term or sentence)
            candidates: List of candidate dicts
            top_k: Return only top-k results
            
        Returns:
            List of (candidate_index, score) tuples
        """
        results = []
        query_lower = query.lower()
        
        for i, candidate in enumerate(candidates):
            score = 1.0
            
            # Boost if query contains biomedical keywords
            for keyword, boost in self.biomedical_keywords.items():
                if keyword in query_lower:
                    score *= boost
                    break  # Only apply one boost per query
            
            # Boost candidates with matching biomedical terms
            evidence = (candidate.get("preferred_label", "") + " " + 
                       candidate.get("definition", "")).lower()
            for keyword, boost in self.biomedical_keywords.items():
                if keyword in evidence:
                    score *= (1 + (boost - 1.0) * 0.5)  # Smaller boost for candidate match
                    break
            
            results.append((i, float(score)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results


class EnsembleReranker:
    """Ensemble reranker combining two or three re-rankers with weighted voting.

    ``components`` controls which rerankers are active.  Any subset of
    ``{"llm", "late_interaction", "biomedical"}`` is valid; weights for
    excluded components are ignored and the active weights are re-normalised.

    Examples
    --------
    All three (default)::

        EnsembleReranker()

    LLM + late-interaction only::

        EnsembleReranker(components={"llm", "late_interaction"})

    Late-interaction + biomedical (no API key needed)::

        EnsembleReranker(components={"late_interaction", "biomedical"})
    """

    def __init__(
        self,
        llm_weight: Optional[float] = None,
        late_interaction_weight: Optional[float] = None,
        biomedical_weight: Optional[float] = None,
        openrouter_api_key: Optional[str] = None,
        openrouter_model: Optional[str] = None,
        late_interaction_model: Optional[str] = None,
        components: Optional[set] = None,
    ):
        config = _get_reranker_config(openrouter_api_key)

        # Which rerankers to use (default: all three)
        _all = {"llm", "late_interaction", "biomedical"}
        self.components = set(components) if components else _all

        api_key  = openrouter_api_key or config["openrouter_api_key"]
        model    = openrouter_model   or config["openrouter_model"]
        li_model = late_interaction_model or config["late_interaction_model"]

        # Only instantiate what we need
        self.llm             = LLMReranker(api_key=api_key, model_name=model) if "llm" in self.components else None
        self.late_interaction = LateInteractionReranker(model_name=li_model)  if "late_interaction" in self.components else None
        self.biomedical      = BiomedicalContextReranker()                    if "biomedical" in self.components else None

        # Weights for active components only, then normalise
        raw = {
            "llm":              (llm_weight              if llm_weight              is not None else config["llm_weight"]),
            "late_interaction": (late_interaction_weight if late_interaction_weight is not None else config["late_interaction_weight"]),
            "biomedical":       (biomedical_weight       if biomedical_weight       is not None else config["biomedical_weight"]),
        }
        active = {k: v for k, v in raw.items() if k in self.components}
        total  = sum(active.values()) or 1.0
        self.llm_weight              = active.get("llm",              0.0) / total
        self.late_interaction_weight = active.get("late_interaction", 0.0) / total
        self.biomedical_weight       = active.get("biomedical",       0.0) / total

        logger.info(
            f"EnsembleReranker initialized (components={sorted(self.components)}): "
            f"LLM={self.llm_weight:.2f}, LI={self.late_interaction_weight:.2f}, Bio={self.biomedical_weight:.2f}"
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        openrouter_api_key: Optional[str] = None,
        openrouter_model: Optional[str] = None,
    ) -> List[RerankingResult]:
        """
        Ensemble re-ranking: combine multiple re-rankers

        Args:
            query: Query text
            candidates: List of candidate dicts with metadata
            top_k: Return top-k results
            openrouter_api_key: Override API key (takes priority over env/instance key)
            openrouter_model: Override model name (takes priority over env/instance model)

        Returns:
            List of RerankingResult objects, ranked by final score
        """
        if not candidates:
            return []

        # Get scores from active rerankers only
        _zero = [(i, 0.0) for i in range(len(candidates))]
        _t0 = time.perf_counter()
        llm_results = self.llm.rerank(query, candidates, api_key=openrouter_api_key, model_name=openrouter_model) if self.llm else _zero
        _t1 = time.perf_counter()
        li_results  = self.late_interaction.rerank(query, candidates) if self.late_interaction else _zero
        _t2 = time.perf_counter()
        bio_results = self.biomedical.rerank(query, candidates)       if self.biomedical else _zero
        _t3 = time.perf_counter()
        logger.info(
            f"[rerank] LLM={1000*(_t1-_t0):.1f}ms  "
            f"LateInteraction={1000*(_t2-_t1):.1f}ms  "
            f"Biomedical={1000*(_t3-_t2):.1f}ms  "
            f"n={len(candidates)}"
        )

        # Convert to score dicts
        llm_scores = dict(llm_results)
        li_scores  = dict(li_results)
        bio_scores = dict(bio_results)

        # Normalize scores to [0, 1]
        def normalize_scores(scores_dict):
            if not scores_dict:
                return {i: 0.0 for i in range(len(candidates))}
            max_score = max(scores_dict.values()) or 1.0
            min_score = min(scores_dict.values()) or 0.0
            if max_score == min_score:
                return {i: 0.5 for i in range(len(candidates))}
            return {
                i: (scores_dict.get(i, 0.0) - min_score) / (max_score - min_score)
                for i in range(len(candidates))
            }

        llm_normalized = normalize_scores(llm_scores)
        li_normalized = normalize_scores(li_scores)
        bio_normalized = normalize_scores(bio_scores)

        # Combine scores
        results = []
        for i, candidate in enumerate(candidates):
            llm_score = llm_normalized.get(i, 0.0)
            li_score = li_normalized.get(i, 0.0)
            bio_score = bio_normalized.get(i, 0.0)
            
            final_score = (
                self.llm_weight * llm_score +
                self.late_interaction_weight * li_score +
                self.biomedical_weight * bio_score
            )

            result = RerankingResult(
                class_uri=candidate.get("class_uri", ""),
                preferred_label=candidate.get("preferred_label", ""),
                ontology_id=candidate.get("ontology_id", ""),
                original_score=candidate.get("original_score", 0.0),
                llm_score=llm_score,
                late_interaction_score=li_score,
                final_score=final_score,
            )
            results.append(result)

        # Sort by final score and assign ranks
        results.sort(key=lambda x: x.final_score, reverse=True)
        for rank, result in enumerate(results[:top_k], 1):
            result.rank = rank

        return results[:top_k]


def create_reranker(
    reranker_type: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    openrouter_model: Optional[str] = None,
) -> Union[LLMReranker, LateInteractionReranker, BiomedicalContextReranker, EnsembleReranker]:
    """
    Factory function to create a reranker based on configuration
    
    Args:
        reranker_type: Override RERANKER_TYPE env variable
                      Options: 'llm', 'late_interaction', 'biomedical', 'ensemble'
        openrouter_api_key: Optional OpenRouter API key (overrides env)
        openrouter_model: Optional OpenRouter model name (overrides env)
                      
    Returns:
        Reranker instance
        
    Examples:
        # Use ensemble (default)
        reranker = create_reranker()
        
        # Use LLM with custom API key
        reranker = create_reranker('llm', openrouter_api_key='your-key')
        
        # Use biomedical context
        reranker = create_reranker('biomedical')
        
        # Use ensemble with custom model
        reranker = create_reranker('ensemble', openrouter_model='google/gemini-2.0-flash-001')
    """
    config = _get_reranker_config(openrouter_api_key)
    reranker_type = reranker_type or config["type"]
    reranker_type = reranker_type.lower().strip()
    api_key = openrouter_api_key or config["openrouter_api_key"]
    model = openrouter_model or config["openrouter_model"]
    
    # Named dual-mode shortcuts + generic comma-separated component list
    _ALIASES = {
        "llm_late":         {"llm", "late_interaction"},
        "llm_biomedical":   {"llm", "biomedical"},
        "dual_late":        {"late_interaction", "biomedical"},  # no LLM, fully local
    }

    # Resolve component set: alias → set, comma list → set, "ensemble" → all three
    if reranker_type in _ALIASES:
        components = _ALIASES[reranker_type]
    elif "," in reranker_type:
        components = {p.strip() for p in reranker_type.split(",")}
    elif reranker_type == "ensemble":
        components = {"llm", "late_interaction", "biomedical"}
    else:
        components = None  # single rerankers handled below

    if components is not None:
        logger.info(f"Creating EnsembleReranker (components={sorted(components)})")
        return EnsembleReranker(
            llm_weight=config["llm_weight"],
            late_interaction_weight=config["late_interaction_weight"],
            biomedical_weight=config["biomedical_weight"],
            openrouter_api_key=api_key,
            openrouter_model=model,
            late_interaction_model=config["late_interaction_model"],
            components=components,
        )

    if reranker_type == "llm":
        logger.info(f"Creating LLMReranker (model: {model})")
        return LLMReranker(api_key=api_key, model_name=model)

    elif reranker_type == "late_interaction":
        logger.info(f"Creating LateInteractionReranker (model: {config['late_interaction_model']})")
        return LateInteractionReranker(model_name=config["late_interaction_model"])

    elif reranker_type == "biomedical":
        logger.info("Creating BiomedicalContextReranker")
        return BiomedicalContextReranker()
    
    else:
        logger.warning(
            f"Unknown reranker type '{reranker_type}', falling back to ensemble"
        )
        return EnsembleReranker(
            openrouter_api_key=api_key,
            openrouter_model=model,
        )
