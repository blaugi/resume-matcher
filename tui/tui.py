from textual import containers, widgets
from textual.app import App, ComposeResult
from textual.message import Message

from core.models import ChunkMatch


class MatchListItem(containers.VerticalGroup):
    """An entry in the match list."""

    can_focus = True

    def __init__(self, match: ChunkMatch) -> None:
        self._match = match
        super().__init__()

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
        self.app.query(MatchListItem).remove_class("selected")
        self.add_class("selected")
        self.post_message(self.Selected(self, self._match.chunk_id))


class MatchWidgetDummy(containers.VerticalGroup):
    """An entry in the match grid select."""

    can_focus = True

    def __init__(self) -> None:
        super().__init__()

    def on_click(self) -> None:
        self.app.query("MatchWidgetDummy").remove_class("selected")
        self.add_class("selected")

    def compose(self) -> ComposeResult:
        yield widgets.Static("Job Chunk:", id="chunk-title-job", classes="chunk-title")
        yield widgets.Label("Lorem job ipsu", id="job-text")
        yield widgets.Rule()
        yield widgets.Static(
            "Resume Chunk:", id="chunk-title-resume", classes="chunk-title"
        )
        yield widgets.Label("Lorem resume ipsu", id="resume-text")
        yield widgets.Static("0.75", id="similarity")


class GridLayoutExample(App):
    CSS = """
Screen {
    layout: grid;
    grid-size: 2;
    grid-columns: 1fr 1fr;
    grid-rows: auto;
    overflow-y: auto;
}

MatchWidgetDummy {
    border: solid $secondary;
    margin: 0 1;
    padding: 0 1;
    height: auto;
}

MatchWidgetDummy:hover {
    background: $accent 10%;
}

/* Style for the single selected widget */
MatchWidgetDummy.selected {
    border: double $success;
    background: $success 10%;
}

#job-text {
    text-style: bold;
}

#resume-text {
    text_style: italic;
}

#similarity {
    text-align: right;
    color: $success;
}

.chunk-title {
    color: $accent;
}
"""
    def compose(self) -> ComposeResult:
        for _ in range(10):
            yield MatchWidgetDummy()


if __name__ == "__main__":
    app = GridLayoutExample()
    app.run()
