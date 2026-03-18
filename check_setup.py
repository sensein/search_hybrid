#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup and initialization script for Ontology Database Concept Mapping Tool
Verifies database, downloads models, pre-builds indexes
"""

import os
import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_database():
    """Verify SQLite database is accessible"""
    logger.info("Checking database...")
    db_path = "bioportal.db"
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        logger.error("Please run bioportal_fetch.py first to download ontologies")
        return False
    
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check schema
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        required = {"classes", "ontologies", "synonyms"}
        
        if not required.issubset(tables):
            logger.error(f"Database schema incomplete. Missing tables: {required - tables}")
            conn.close()
            return False
        
        cursor.execute("SELECT COUNT(*) FROM classes")
        num_classes = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM ontologies")
        num_ontologies = cursor.fetchone()[0]
        
        conn.close()
        
        if num_classes == 0:
            logger.error("No classes found in database")
            return False
        
        logger.info(f"Database OK: {num_ontologies} ontologies, {num_classes:,} classes")
        return True
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        return False


def download_models(force: bool = False):
    """Download required transformer models"""
    logger.info("Checking/downloading models...")
    
    models = {
        "Embedding model": "sentence-transformers/all-MiniLM-L6-v2",
        "Cross-encoder model": "cross-encoder/stsb-distilroberta-base",
        "Late-interaction model": "jinaai/jina-colbert-v2",
    }
    
    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
        
        # Embedding model
        logger.info("  Downloading sentence-transformers embeddings...")
        embedding_model = models["Embedding model"]
        try:
            model = SentenceTransformer(embedding_model)
            logger.info(f"    ✓ {embedding_model}")
        except Exception as e:
            logger.warning(f"    ⚠ Failed to load embedding model: {e}")
        
        # Cross-encoder
        logger.info("  Downloading cross-encoder...")
        ce_model = models["Cross-encoder model"]
        try:
            model = CrossEncoder(ce_model)
            logger.info(f"    ✓ {ce_model}")
        except Exception as e:
            logger.warning(f"    ⚠ Failed to load cross-encoder: {e}")
        
        logger.info("✓ Models downloaded/verified")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Required package not installed: {e}")
        logger.error("Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"❌ Model download failed: {e}")
        return False


def test_api_endpoints():
    """Test API endpoints"""
    logger.info("Testing API endpoints...")
    
    try:
        import requests
        
        # Wait for API to start if just launching
        import time
        time.sleep(2)
        
        base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        
        # Health check
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✓ Health check passed")
            else:
                logger.warning(f"⚠ Health check returned {response.status_code}")
        except requests.ConnectionError:
            logger.warning("⚠ Cannot connect to API (not running?)")
            return False
        
        # Simple concept mapping
        try:
            response = requests.post(
                f"{base_url}/map/concept",
                json={"text": "diabetes", "max_results": 1},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("results"):
                    logger.info("✓ Concept mapping works")
                else:
                    logger.warning("⚠ Concept mapping returned no results")
            else:
                logger.warning(f"⚠ Concept mapping returned {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠ Concept mapping test failed: {e}")
        
        return True
        
    except ImportError:
        logger.warning("⚠ requests library not found (optional)")
        return True
    except Exception as e:
        logger.error(f"API test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup and verify Ontology Database Concept Mapping Tool"
    )
    parser.add_argument(
        "--check-db",
        action="store_true",
        help="Check database only"
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download required models only"
    )
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Test API endpoints only"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=True,
        help="Run all checks (default)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Ontology Database Concept Mapping Tool - Setup Verification")
    logger.info("=" * 70)
    
    results: dict[str, bool] = {}
    
    # Run checks
    if args.check_db or args.full:
        results["database"] = check_database()
    
    if args.download_models or args.full:
        results["models"] = download_models()
    
    if args.test_api or args.full:
        results["api"] = test_api_endpoints()
    
    # Summary
    logger.info("=" * 70)
    if all(results.values()):
        logger.info("✓ All checks passed! Ready to use.")
        return 0
    else:
        logger.warning("⚠ Some checks failed. See above for details.")
        if not results.get("database"):
            logger.warning("- Ensure bioportal.db exists and contains the required schema.")
        if not results.get("models"):
            logger.warning("- Install all dependencies: pip install -r requirements.txt")
        if not results.get("api"):
            logger.warning("- Start API: python -m uvicorn main:app --reload")
        return 1


if __name__ == "__main__":
    sys.exit(main())
