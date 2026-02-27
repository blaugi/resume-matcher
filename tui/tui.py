from textual import containers, widgets
from textual.app import ComposeResult
from textual.message import Message

from core.models import SuggestedEdit


class ItemSelected(Message):
    """Shared message for when a list item is clicked."""
    def __init__(self, widget_instance, data_payload):
        super().__init__()
        self.widget = widget_instance
        self.item_id = data_payload

class EditListItem(containers.VerticalGroup):
    """An entry in the edit list."""

    can_focus = True

    def __init__(self, edit: SuggestedEdit) -> None:
        self._edit = edit
        super().__init__(id=f"edit-{edit.id}")

    def compose(self) -> ComposeResult:
        edit = self._edit
        reason_snippet = edit.reason[:60].replace("\n", " ") + "..."

        with containers.Horizontal(classes="match-header"):
            delta_str = (
                f"+{edit.similarity_delta:.4f}"
                if edit.similarity_delta > 0
                else f"{edit.similarity_delta:.4f}"
            )
            yield widgets.Label(
                f"Impact: [{delta_str}]", classes="match-score", id="match-score-label"
            )
            yield widgets.Label(
                edit.status.upper(),
                classes=f"match-status {edit.status.lower().replace(' ', '-')}",
                id="match-status-label",
            )

        yield widgets.Label(reason_snippet, classes="match-snippet")

    def refresh_edit_data(self) -> None:
        """Update the labels with the latest edit data."""
        status_label = self.query_one("#match-status-label", widgets.Label)
        status_label.update(self._edit.status.upper())

        status_label.set_classes(
            f"match-status {self._edit.status.lower().replace(' ', '-')}"
        )

    def on_click(self) -> None:
        self.focus()
        self.post_message(ItemSelected(self, self._edit.id))


class KeywordListItem(containers.VerticalGroup):
    """An entry in the edit list."""
    can_focus = True

    def __init__(self, keywords: tuple[list[str], list[str]]) -> None:
        self._present_keywords= keywords[0] 
        self._missing_keywords= keywords[1] 
        super().__init__(id="keyword-list-item")

    def compose(self) -> ComposeResult:

        with containers.Horizontal(classes="match-header"):
            yield widgets.Label(
                "Keywords", classes="match-score", id="match-score-label"
            )
        yield widgets.Label("Keyword Information", classes="match-snippet")


    def on_click(self) -> None:
        self.focus()
        self.post_message(ItemSelected(self, "keyword-list-item"))