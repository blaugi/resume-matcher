from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.widget import Widget
from textual import containers
from core.models import ChunkMatch
from textual import widgets


class MatchWidget(containers.VerticalGroup):
    """An entry in the match grid select."""

    def __init__(self, match: ChunkMatch) -> None:
        self._match = match
        super().__init__()

    def compose(self) -> ComposeResult:
        match = self._match
        yield widgets.Label(match.get_job_text(), id="job-text")
        yield widgets.Label(match.get_resume_text(), id="resume-text")
        yield widgets.Static(str(match.similarity), id="similarity")


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
        yield widgets.Static(str("0.75"), id="similarity")


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
