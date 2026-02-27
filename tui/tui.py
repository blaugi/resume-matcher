from textual import containers, widgets
from textual.app import App, ComposeResult
from textual.message import Message

from core.models import ChunkMatch


class MatchListItem(containers.VerticalGroup):
    """An entry in the match list."""

    can_focus = True

    def __init__(self, match: ChunkMatch) -> None:
        self._match = match
        super().__init__(id=f"match-{match.chunk_id}")

    def compose(self) -> ComposeResult:
        match = self._match
        job_snippet = match.get_job_text()[:60].replace('\n', ' ') + "..."
        
        with containers.Horizontal(classes="match-header"):
            yield widgets.Label(f"Score: {match.similarity:.2f}", classes="match-score")
            yield widgets.Label(match.status, classes=f"match-status {match.status.lower().replace(' ', '-')}")
            
        yield widgets.Label(job_snippet, classes="match-snippet")

    class Selected(Message):
        def __init__(self, widget_instance, data_payload):
            super().__init__()
            self.widget = widget_instance
            self.chunk_id = data_payload

    def on_click(self) -> None:
        self.post_message(self.Selected(self, self._match.chunk_id))

