from abc import ABC, abstractmethod
from src.models import Market


class BaseFeed(ABC):
    @abstractmethod
    async def fetch(self) -> list[Market]:
        """Fetch and normalize current markets from this source."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up connections."""
        ...
