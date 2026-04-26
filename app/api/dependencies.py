from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.memory.service import MemoryService
from app.orchestrator.graph import OrchestrationGraph
from app.retrieval.service import RetrievalService
from app.services.chat_service import ChatService
from app.core.retry import RetryPolicy
from app.storage.memory_repo import MemoryRepository
from app.storage.sqlite import SQLiteStore
from app.tools.executor import ToolExecutor
from app.tools.registry import DefaultToolRegistryFactory


@dataclass(slots=True)
class AppContainer:
    repository: MemoryRepository
    memory_service: MemoryService
    retrieval_service: RetrievalService
    tool_executor: ToolExecutor
    chat_service: ChatService
    graph: OrchestrationGraph


def build_container() -> AppContainer:
    repository = MemoryRepository(SQLiteStore())
    memory_service = MemoryService(repository)
    retrieval_service = RetrievalService(memory_service.corpus)
    tool_registry = DefaultToolRegistryFactory(retrieval_service=retrieval_service).build()
    tool_executor = ToolExecutor(
        tool_registry,
        retry_policy=RetryPolicy(
            max_attempts=1,
            initial_delay_seconds=0.0,
            backoff_factor=1.0,
            max_delay_seconds=0.0,
        ),
    )
    graph = OrchestrationGraph(memory_service=memory_service, retrieval_service=retrieval_service, tool_executor=tool_executor)
    chat_service = ChatService(graph)
    return AppContainer(
        repository=repository,
        memory_service=memory_service,
        retrieval_service=retrieval_service,
        tool_executor=tool_executor,
        chat_service=chat_service,
        graph=graph,
    )


def get_container(request: Request):
    return request.app.state.container


def get_chat_service(request: Request) -> ChatService:
    return get_container(request).chat_service


def get_retrieval_service(request: Request) -> RetrievalService:
    return get_container(request).retrieval_service
