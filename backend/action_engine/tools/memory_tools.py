"""
Memory Tools
============

Wraps memory extraction and management into Action Engine tools.
"""
from backend.action_engine.tool_registry import registry, ToolDefinition, ToolParameter
from backend.database import DatabaseManager
from backend.llm import OllamaClient
from configs.settings import get_settings
from memory.extractor import MemoryExtractor
from memory.manager import MemoryManager
from memory.vector_store import VectorStore

_memory_manager_instance = None


def _get_memory_manager() -> MemoryManager:
    global _memory_manager_instance
    if _memory_manager_instance is None:
        settings = get_settings()
        db = DatabaseManager(settings=settings)
        vector_store = VectorStore(settings=settings)
        llm = OllamaClient(settings=settings)
        extractor = MemoryExtractor(llm_client=llm)
        _memory_manager_instance = MemoryManager(
            db=db,
            vector_store=vector_store,
            extractor=extractor,
            settings=settings,
        )
    return _memory_manager_instance


def store_memory(kwargs: dict) -> str:
    mm = _get_memory_manager()
    memory_id = mm.store_manual_memory(
        content=kwargs.get("content"),
        memory_type=kwargs.get("memory_type", "fact"),
        importance=kwargs.get("importance", 0.5),
    )
    return {
        "message": f"Stored memory with ID: {memory_id}",
        "reverse_operation": {
            "intent": "forget_memory",
            "parameters": {"memory_id": memory_id},
        },
    }


def forget_memory(kwargs: dict) -> str:
    mm = _get_memory_manager()
    memory_id = kwargs.pop("memory_id")
    deleted = mm.delete_memory(memory_id)
    if deleted:
        return {"message": f"Forgot memory {memory_id}."}
    return {"message": f"Memory {memory_id} not found."}


registry.register(
    ToolDefinition(
        name="store_memory",
        description="Explicitly store a fact or memory for long-term recall.",
        parameters=[
            ToolParameter("content", "string", "The memory content to store"),
            ToolParameter(
                "memory_type",
                "string",
                "Type of memory (fact, preference, etc)",
                required=False,
                enum=[
                    "fact",
                    "summary",
                    "preference",
                    "goal",
                    "project",
                    "habit",
                    "routine",
                    "deadline",
                    "observation",
                ],
            ),
            ToolParameter(
                "importance", "number", "Importance score from 0.0 to 1.0", required=False
            ),
            ToolParameter(
                "conversation_id", "string", "Optional conversation ID", required=False
            ),
        ],
        handler=store_memory,
        risk_level="low",
    )
)

registry.register(
    ToolDefinition(
        name="forget_memory",
        description="Forget a stored memory.",
        parameters=[ToolParameter("memory_id", "string", "The ID of the memory to delete")],
        handler=forget_memory,
        risk_level="high",
    )
)
