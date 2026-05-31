"""
Jarvis Phase 3 — System Validation Suite
========================================
Runs health checks and operational diagnostics across all subsystems.
Outputs results and operational metrics.
"""

import sys
import time
from backend.database import DatabaseManager
from backend.provider_manager import ProviderManager
from backend.lm_studio import LMStudioClient
from backend.llm import OllamaClient
from backend.cloud_llm import CloudLLM
from backend.model_router import ModelRouter
from backend.explainability_engine import ExplainabilityEngine
from backend.analytics_engine import AnalyticsEngine
from state.analytics_manager import AnalyticsManager
from knowledge.knowledge_store import KnowledgeStore
from memory.vector_store import VectorStore
from configs.settings import get_settings

def print_result(name: str, status: str, metrics: list = None):
    print(f"{name.ljust(20)} ... {status}")
    if metrics:
        print("\n  Metrics:")
        for metric in metrics:
            print(f"    - {metric}")
        print()

def main():
    print("========================================")
    print("Jarvis System Validation Suite")
    print("========================================\n")
    
    settings = get_settings()
    db = DatabaseManager(settings=settings)
    
    # --- 1. Database ---
    start_time = time.time()
    try:
        tables = db._connect().execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        db_time = (time.time() - start_time) * 1000
        print_result("Database", "PASS", [f"Response Time: {db_time:.1f}ms", f"Tables found: {len(tables)}"])
    except Exception as e:
        print_result("Database", f"FAIL ({e})")
        sys.exit(1)

    # --- 2. Memory Store (Vector Store) ---
    vs = None
    start_time = time.time()
    try:
        vs = VectorStore(settings=settings)
        count = vs.get_count()
        vs_time = (time.time() - start_time) * 1000
        print_result("Memory Store", "PASS", [f"Response Time: {vs_time:.1f}ms", f"Total Vectors: {count}"])
    except Exception as e:
        print_result("Memory Store", f"FAIL ({e})")

    # --- 3. Knowledge Store ---
    start_time = time.time()
    try:
        ks = KnowledgeStore(settings=settings)
        colls = ks._client.list_collections()
        ks_time = (time.time() - start_time) * 1000
        print_result("Knowledge Store", "PASS", [f"Response Time: {ks_time:.1f}ms", f"Collections: {len(colls)}"])
    except Exception as e:
        print_result("Knowledge Store", f"FAIL ({e})")

    # --- 4. LM Studio ---
    start_time = time.time()
    try:
        lm_client = LMStudioClient(settings=settings)
        health = lm_client.check_health()
        lm_time = (time.time() - start_time) * 1000
        if health["status"] == "healthy":
            print_result("LM Studio", "PASS", [f"Response Time: {lm_time:.1f}ms", f"Available Models: {len(health['available_models'])}"])
        else:
            print_result("LM Studio", f"WARNING ({health.get('error', 'unreachable')})")
    except Exception as e:
        print_result("LM Studio", f"FAIL ({e})")

    # --- 5. Provider Manager (with authenticated health checks) ---
    pm = None
    start_time = time.time()
    try:
        pm = ProviderManager(db=db, settings=settings)
        all_health = pm.check_all_health()
        pm_time = (time.time() - start_time) * 1000
        metrics = [f"Health Check Time: {pm_time:.1f}ms"]
        for pname, pstatus in all_health.items():
            detail = f"Status={pstatus.status}"
            if pstatus.error:
                detail += f", Reason={pstatus.error}"
            if pstatus.latency_ms > 0:
                detail += f", Latency={pstatus.latency_ms:.0f}ms"
            metrics.append(f"  {pname.upper()}: {detail}")
        overall = "PASS" if any(s.status == "healthy" for s in all_health.values()) else "WARNING"
        print_result("Provider Manager", overall, metrics)
    except Exception as e:
        print_result("Provider Manager", f"FAIL ({e})")

    # --- 6. Explainability Engine ---
    start_time = time.time()
    try:
        ee = ExplainabilityEngine(db=db, settings=settings)
        history = ee.get_decision_history(limit=1)
        ee_time = (time.time() - start_time) * 1000
        print_result("Explainability", "PASS", [f"Response Time: {ee_time:.1f}ms", f"History Items: {len(history)}"])
    except Exception as e:
        print_result("Explainability", f"FAIL ({e})")

    # --- 7. Analytics Engine ---
    start_time = time.time()
    try:
        am = AnalyticsManager(db=db)
        ae = AnalyticsEngine(db=db, analytics_manager=am, settings=settings)
        ae_time = (time.time() - start_time) * 1000
        print_result("Analytics Engine", "PASS", [f"Response Time: {ae_time:.1f}ms"])
    except Exception as e:
        print_result("Analytics Engine", f"FAIL ({e})")

    # --- 8. Model Router ---
    router = None
    if pm:
        start_time = time.time()
        try:
            ollama_client = OllamaClient(settings=settings)
            cloud_llm = CloudLLM(db=db, provider_manager=pm, vector_store=vs, settings=settings)
            lm_studio_client = LMStudioClient(settings=settings)
            
            router = ModelRouter(
                provider_manager=pm,
                cloud_llm=cloud_llm,
                ollama_client=ollama_client,
                lm_studio_client=lm_studio_client,
                db=db,
                settings=settings
            )
            router_time = (time.time() - start_time) * 1000
            print_result("Model Router", "PASS", [f"Init Time: {router_time:.1f}ms"])
        except Exception as e:
            print_result("Model Router", f"FAIL ({e})")

    # --- 9. Router Verification Suite ---
    if router:
        print("\n----------------------------------------")
        print("Router Verification Suite")
        print("----------------------------------------\n")

        TEST_CASES = [
            ("write a python web scraper",   "coding",    "Expected: coding route"),
            ("solve x^2 + 5x + 6 = 0",      "math",      "Expected: math route"),
            ("remind me to study tomorrow",  "fast",      "Expected: fast route"),
            ("explain transformers in detail", "reasoning", "Expected: reasoning route"),
        ]

        all_passed = True
        for query, expected_tier, label in TEST_CASES:
            decision = router.route(query, [])
            ok = decision.selected_provider != "disabled"
            status_str = "PASS" if ok else "FAIL (disabled)"
            all_passed = all_passed and ok
            print(f"  [{status_str}] {label}")
            print(f"    Query:    {query}")
            print(f"    Provider: {decision.selected_provider}")
            print(f"    Model:    {decision.selected_model}")
            print(f"    Reason:   {decision.reason}\n")

        print("Router Suite:", "ALL PASS" if all_passed else "FAILURES DETECTED")

    # --- 10. Alias Validation Report ---
    if pm:
        print("\n----------------------------------------")
        print("=== MODEL ALIAS VALIDATION ===")
        print("----------------------------------------")
        tiers = ["fast", "reasoning", "coding", "math", "classifier"]
        local_providers = ["ollama", "lm_studio"]
        local_health_cache = {}
        for prov in local_providers:
            local_health_cache[prov] = pm.check_health(prov)

        for tier in tiers:
            print(f"\n  {tier.upper()}")
            for prov in local_providers:
                model_name = settings.local_models.get_model_for(tier, prov)
                ph = local_health_cache.get(prov)
                label = "Ollama" if prov == "ollama" else "LM Studio"
                if not model_name:
                    print(f"    {label}: (not configured)")
                    continue
                if ph and ph.status == "healthy" and model_name in ph.available_models:
                    status_tag = "FOUND"
                elif ph and ph.status != "healthy":
                    status_tag = "PROVIDER OFFLINE"
                else:
                    status_tag = "NOT FOUND"
                print(f"    {label}: {model_name}  [{status_tag}]")

    print("\n========================================")
    print("Validation Suite Complete.")
    print("========================================\n")

if __name__ == "__main__":
    main()
