import sys
import time
import logging
from configs.settings import get_settings
from backend.s4_roles import S4RoleManager
from backend.llm import OllamaClient
from backend.lm_studio import LMStudioClient

logging.basicConfig(level=logging.WARNING)

def run_diagnostics():
    print("=== JARVIS S4 DIAGNOSTIC SUITE ===\n")
    
    settings = get_settings()
    
    # 1. Initialize Clients
    print("[1] Initializing Provider Clients...")
    try:
        ollama = OllamaClient(settings=settings)
        print("    [OK] OllamaClient instantiated.")
    except Exception as e:
        print(f"    [FAIL] OllamaClient error: {e}")
        return

    try:
        lm_studio = LMStudioClient(settings=settings)
        print("    [OK] LMStudioClient instantiated.")
    except Exception as e:
        print(f"    [FAIL] LMStudioClient error: {e}")
        return

    try:
        rm = S4RoleManager(ollama_client=ollama, lm_studio_client=lm_studio, settings=settings)
        print("    [OK] S4RoleManager instantiated.\n")
    except Exception as e:
        print(f"    [FAIL] S4RoleManager error: {e}")
        return

    # 2. Test connectivity globally
    print("[2] Global Provider Health...")
    ollama_health = ollama.check_health()
    lm_studio_health = lm_studio.check_health()
    print(f"    Ollama: {ollama_health['status'].upper()} (models: {len(ollama_health.get('available_models', []))})")
    print(f"    LM Studio: {lm_studio_health['status'].upper()} (models: {len(lm_studio_health.get('available_models', []))})\n")

    # 3. Test Each Role
    print("[3] Testing S4 Roles...")
    roles = ["chief", "analyst", "engineer", "mentor", "rapid"]
    
    for role_name in roles:
        print(f"\n--- Role: {role_name.upper()} ---")
        role = rm.get_role(role_name)
        if not role:
            print("    Status: [FAIL] Unknown role.")
            continue
            
        print(f"    Provider: {role.provider}")
        print(f"    Target Model: {role.model_id}")
        
        is_avail = rm.is_available(role_name)
        print(f"    Reachable: {'YES' if is_avail else 'NO'}")
        
        if not is_avail:
            print("    Skipping execution due to unreachable provider.")
            continue
            
        print("    Sending test ping...")
        start_t = time.time()
        try:
            result = rm.call_role(
                role_name=role_name,
                message="Diagnostic ping. Reply with 'ACK'.",
                conversation_history=[],
                context_kwargs={}
            )
            
            if result.success:
                print(f"    Status: [SUCCESS]")
                print(f"    Actual Model: {result.model_id}")
                print(f"    Latency: {result.duration_ms}ms")
                print(f"    Response: {result.content.strip()}")
            else:
                print(f"    Status: [FAIL]")
                print(f"    Error: {result.error}")
                if getattr(result, "fallback_used", False):
                    print("    Note: Fallback to Ollama was triggered.")
        except Exception as e:
            print(f"    Status: [CRASH]")
            print(f"    Exception: {str(e)}")

if __name__ == "__main__":
    run_diagnostics()
