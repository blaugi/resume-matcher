from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import (
    Button,
    DirectoryTree,
    Header,
    Input,
    Label,
    LoadingIndicator,
    TextArea,
)

from core.engine import ResumeEngine
from tui.tui import MatchListItem


class LoadingScreen(Screen):
    """Screen displayed while the engine processes documents."""

    def compose(self) -> ComposeResult:
        with Vertical(id="loading-container"):
            yield LoadingIndicator()
            yield Label("Processing documents...", id="loading-label")


class InputScreen(Screen):
    """Screen for uploading resume and pasting job description."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="resume-input-container"):
                yield Label("Resume Path", classes="input-label")
                yield Input(
                    placeholder="Enter path to resume (e.g., data/resume.pdf)",
                    id="resume-path",
                )
                yield Label(
                    "Or select from data folder:",
                    classes="input-label",
                    id="tree-label",
                )
                yield DirectoryTree("./data", id="resume-tree")

            with Vertical(id="job-input-container"):
                yield Label("Job Description", classes="input-label")
                yield TextArea(id="job-text")

        yield Button("Find Matches", id="process-btn")

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        """Update the input field when a file is selected in the tree."""
        input_widget = self.query_one("#resume-path", Input)
        input_widget.value = str(event.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        app: ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]
        if event.button.id == "process-btn":
            resume_path = self.query_one("#resume-path", Input).value
            job_offer = self.query_one("#job-text", TextArea).text

            app.push_screen(LoadingScreen())
            self.process_documents(resume_path, job_offer)

    @work(thread=True)
    def process_documents(self, resume_path: str, job_offer: str) -> None:
        app: ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]

        app.resume_eng.load_resume(resume_path)
        app.resume_eng.load_job(job_offer)

        self.app.call_from_thread(self.show_results)

    def show_results(self) -> None:
        self.app.pop_screen()
        self.app.push_screen(ResultsScreen())


class ResultsScreen(Screen):
    """Screen displaying the master-detail view of matches."""

    def compose(self) -> ComposeResult:
        app: ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]
        yield Header()

        current_score, original_score = app.resume_eng.get_overall_similarity()
        display_score = (
            f"{original_score:.2%} -> {current_score:.2%}"
            if current_score != original_score
            else f"{current_score:.2%}"
        )
        yield Label(f"Overall Resume Match: {display_score}", id="overall-score-label")

        with Horizontal(id="results-container"):
            with VerticalScroll(id="match-list"):
                if matches := app.resume_eng.get_matches():
                    for match in matches:
                        yield MatchListItem(match)

            with VerticalScroll(id="match-detail"):
                yield Label("Select a match to view details", id="empty-detail-msg")

        yield Button("Finish", id="finish-btn", variant="primary")

    async def on_match_list_item_selected(
        self, message: MatchListItem.Selected
    ) -> None:
        for widget in self.query(MatchListItem).results():
            widget.set_class(widget.id == message.widget.id, "selected")

        selected_match_id = message.chunk_id
        await self.show_detail(selected_match_id)

    async def show_detail(self, match_id: str) -> None:
        app: ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]
        detail_container = self.query_one("#match-detail", VerticalScroll)

        await detail_container.query("*").remove()

        match = app.resume_eng.get_match_from_uuid(match_id)
        if not match:
            return

        await detail_container.mount(
            Label("Job Requirement", classes="detail-section-title"),
            TextArea(
                text=match.get_job_text(),
                read_only=True,
                classes="detail-textarea job-textarea",
            ),
            Label("Current Resume Chunk", classes="detail-section-title"),
            TextArea(
                text=match.get_resume_text(),
                read_only=True,
                classes="detail-textarea old-textarea",
            ),
            Label("Edit Resume Chunk", classes="detail-section-title"),
            TextArea(
                text=match.get_resume_text(),
                id="edit-textarea",
                classes="detail-textarea new-textarea",
            ),
            Horizontal(
                Button("Generate New Chunk", id="generate-btn", variant="warning"),
                Button("Save Changes", id="save-btn", variant="primary"),
                Button("Cancel", id="cancel-btn", variant="error"),
                id="detail-buttons",
            ),
        )
        self.current_match_id = match_id

    def on_button_pressed(self, event: Button.Pressed) -> None:
        app: ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]
        if event.button.id == "finish-btn":
            app.exit()
        elif event.button.id == "generate-btn":
            self.generate_new_chunk()
        elif event.button.id == "save-btn":
            new_text = self.query_one("#edit-textarea", TextArea).text
            app.resume_eng.update_resume_chunk(self.current_match_id, new_text)

            # Refresh UI
            # 1. Update the overall score label
            current_overall, original_overall = app.resume_eng.get_overall_similarity()
            display_overall = (
                f"{original_overall:.2%} -> {current_overall:.2%}"
                if current_overall != original_overall
                else f"{current_overall:.2%}"
            )
            self.query_one("#overall-score-label", Label).update(
                f"Overall Resume Match: {display_overall}"
            )

            # 2. Update the MatchListItem in the list
            try:
                list_item = self.query_one(
                    f"#match-{self.current_match_id}", MatchListItem
                )
                list_item.refresh_match_data()
            except NoMatches:
                pass  # Widget might have been unmounted

            self.app.notify("Chunk updated successfully!")
        elif event.button.id == "cancel-btn":
            match = app.resume_eng.get_match_from_uuid(self.current_match_id)
            if match:
                self.query_one(
                    "#edit-textarea", TextArea
                ).text = match.get_resume_text()

    @work(thread=True)
    def generate_new_chunk(self) -> None:
        """Call LLM to reformat the chunk and update the view."""
        app: ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]
        match_id = self.current_match_id

        # Get chunk outside thread for safety
        match = app.resume_eng.get_match_from_uuid(match_id)
        if not match:
            return

        # Notify UI we're starting
        self.app.call_from_thread(self._set_loading_state, True)

        try:
            new_text = app.resume_eng.reformat_chunk(match)
            # Send result back (verify we are still on that chunk)
            self.app.call_from_thread(self._finish_gen, match_id, new_text)
        except Exception as e:
            self.app.call_from_thread(
                self.app.notify, f"Error generating: {str(e)}", severity="error"
            )
        finally:
            self.app.call_from_thread(self._set_loading_state, False)

    def _set_loading_state(self, is_loading: bool) -> None:
        """Helper to show/hide loading indicator and toggle button."""
        try:
            gen_btn = self.query_one("#generate-btn", Button)
            gen_btn.disabled = is_loading

            if is_loading:
                if not self.query("#gen-loading"):
                    self.query_one("#match-detail", VerticalScroll).mount(
                        LoadingIndicator(id="gen-loading"), before="#detail-buttons"
                    )
            else:
                self.query("#gen-loading").remove()
        except NoMatches:
            pass

    def _finish_gen(self, match_id: str, new_text: str) -> None:
        """Update the TextArea with generated text if match hasn't changed."""
        if self.current_match_id == match_id:
            try:
                self.query_one("#edit-textarea", TextArea).text = new_text
                self.app.notify("Generation complete!")
            except NoMatches:
                pass


class ResumeMatcherApp(App):
    CSS_PATH = "styles.tcss"

    def __init__(self, resume_eng: ResumeEngine):
        super().__init__()
        self.resume_eng = resume_eng

    def on_mount(self) -> None:
        self.push_screen(InputScreen())
