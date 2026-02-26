from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, TextArea, LoadingIndicator, Label
from textual import work

from core.engine import ResumeEngine
from core.models import ChunkMatch
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
                yield Input(placeholder="Enter path to resume (e.g., data/resume.pdf)", id="resume-path")
            
            with Vertical(id="job-input-container"):
                yield Label("Job Description", classes="input-label")
                yield TextArea(id="job-text")
        
        yield Button("Find Matches", id="process-btn")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        app:ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]
        if event.button.id == "process-btn":
            resume_path = self.query_one("#resume-path", Input).value
            job_offer = self.query_one("#job-text", TextArea).text
            
            # Push loading screen first
            app.push_screen(LoadingScreen())
            
            # Start background processing
            self.process_documents(resume_path, job_offer)

    @work(thread=True)
    def process_documents(self, resume_path: str, job_offer: str) -> None:
        app:ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]
        
        # Run the heavy engine tasks in the background thread
        app.resume_eng.load_resume(resume_path)
        app.resume_eng.load_job(job_offer)
        
        # Once done, tell the main UI thread to update
        self.app.call_from_thread(self.show_results)

    def show_results(self) -> None:
        # Pop the loading screen and push results
        self.app.pop_screen()
        self.app.push_screen(ResultsScreen())

class ResultsScreen(Screen):
    """Screen displaying the master-detail view of matches."""

    def compose(self) -> ComposeResult:
        app:ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]
        yield Header()
        
        with Horizontal(id="results-container"):
            with VerticalScroll(id="match-list"):
                if matches := app.resume_eng.get_matches():
                    for match in matches:
                        yield MatchListItem(match)
            
            with VerticalScroll(id="match-detail"):
                yield Label("Select a match to view details", id="empty-detail-msg")
                
        yield Button("Finish", id="finish-btn", variant="primary")
        yield Footer()

    async def on_match_list_item_selected(self, message: MatchListItem.Selected) -> None:
        selected_match_id = message.chunk_id
        await self.show_detail(selected_match_id)

    async def show_detail(self, match_id: str) -> None:
        app:ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]
        detail_container = self.query_one("#match-detail", VerticalScroll)
        
        # Clear current details
        await detail_container.query("*").remove()
            
        match = app.resume_eng.get_match_from_uuid(match_id)
        if not match:
            return
            
        # Mount new details
        await detail_container.mount(
            Label("Job Requirement", classes="detail-section-title"),
            TextArea(text=match.get_job_text(), read_only=True, classes="detail-textarea job-textarea"),
            Label("Current Resume Chunk", classes="detail-section-title"),
            TextArea(text=match.get_resume_text(), read_only=True, classes="detail-textarea old-textarea"),
            Label("Edit Resume Chunk", classes="detail-section-title"),
            TextArea(text=match.get_resume_text(), id="edit-textarea", classes="detail-textarea new-textarea"),
            Horizontal(
                Button("Generate New Chunk", id="generate-btn", variant="warning"),
                Button("Save Changes", id="save-btn", variant="primary"),
                Button("Cancel", id="cancel-btn", variant="error"),
                id="detail-buttons"
            )
        )
        self.current_match_id = match_id

    def on_button_pressed(self, event: Button.Pressed) -> None:
        app:ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]
        if event.button.id == "finish-btn":
            app.exit()
        elif event.button.id == "generate-btn":
            self.app.notify("Generating new chunk... (Not implemented yet)", severity="warning")
        elif event.button.id == "save-btn":
            new_text = self.query_one("#edit-textarea", TextArea).text
            app.resume_eng.update_resume_chunk(self.current_match_id, new_text)
            self.app.notify("Chunk updated successfully!")
        elif event.button.id == "cancel-btn":
            match = app.resume_eng.get_match_from_uuid(self.current_match_id)
            if match:
                self.query_one("#edit-textarea", TextArea).text = match.get_resume_text()


class ResumeMatcherApp(App):
    CSS_PATH = "styles.tcss" # (Optional) Moving CSS to a separate file is cleaner!

    def __init__(self, resume_eng: ResumeEngine):
        super().__init__()
        self.resume_eng = resume_eng

    def on_mount(self) -> None:
        # Start the app on the Input Screen
        self.push_screen(InputScreen())

if __name__ == "__main__":

    resume_eng = ResumeEngine()
    app = ResumeMatcherApp(resume_eng)
    app.run()