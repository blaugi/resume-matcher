from textual import containers, widgets
from textual.app import ComposeResult
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
        job_snippet = match.get_job_text()[:60].replace("\n", " ") + "..."

        with containers.Horizontal(classes="match-header"):
            display_score = (
                f"{match.original_similarity:.2f} -> {match.similarity:.2f}"
                if match.similarity != match.original_similarity
                else f"{match.similarity:.2f}"
            )
            yield widgets.Label(
                f"Score: {display_score}", classes="match-score", id="match-score-label"
            )
            yield widgets.Label(
                match.status,
                classes=f"match-status {match.status.lower().replace(' ', '-')}",
                id="match-status-label",
            )

        yield widgets.Label(job_snippet, classes="match-snippet")

    def refresh_match_data(self) -> None:
        """Update the labels with the latest match data."""
        display_score = (
            f"{self._match.original_similarity:.2f} -> {self._match.similarity:.2f}"
            if self._match.similarity != self._match.original_similarity
            else f"{self._match.similarity:.2f}"
        )
        self.query_one("#match-score-label", widgets.Label).update(
            f"Score: {display_score}"
        )
        status_label = self.query_one("#match-status-label", widgets.Label)
        status_label.update(self._match.status)

        status_label.set_classes(
            f"match-status {self._match.status.lower().replace(' ', '-')}"
        )

    class Selected(Message):
        def __init__(self, widget_instance, data_payload):
            super().__init__()
            self.widget = widget_instance
            self.chunk_id = data_payload

    def on_click(self) -> None:
        self.post_message(self.Selected(self, self._match.chunk_id))
