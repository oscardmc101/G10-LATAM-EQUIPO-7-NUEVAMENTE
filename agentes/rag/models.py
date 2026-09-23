from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def document_id(self) -> str:
        if "document_id" not in self.metadata or not self.metadata["document_id"]:
            raise KeyError("Document metadata does not contain a valid 'document_id'")
        return str(self.metadata["document_id"])


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def document_id(self) -> str:
        if "document_id" not in self.metadata or not self.metadata["document_id"]:
            raise KeyError("Chunk metadata does not contain a valid 'document_id'")
        return str(self.metadata["document_id"])


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def document_id(self) -> str:
        if "document_id" not in self.metadata or not self.metadata["document_id"]:
            raise KeyError("SearchResult metadata does not contain a valid 'document_id'")
        return str(self.metadata["document_id"])