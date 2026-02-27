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
from tui.tui import EditListItem, ItemSelected, KeywordListItem


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

        yield Button("Analyze", id="process-btn")

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
        app.resume_eng.generate_edits()

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
                if keywords := app.resume_eng.keywords:
                    yield KeywordListItem(keywords)
                if edits := app.resume_eng.edit_list:
                    for edit in edits:
                        yield EditListItem(edit)

            with VerticalScroll(id="match-detail"):
                yield Label("Select an edit to view details", id="empty-detail-msg")

        yield Button("Finish", id="finish-btn", variant="primary")

    async def on_item_selected(self, message: ItemSelected) -> None:
        for widget in self.query("EditListItem, KeywordListItem").results():
            widget.set_class(widget.id == message.widget.id, "selected")

        selected_edit_id = message.item_id
        await self.show_detail_edit(selected_edit_id)

    async def show_detail_edit(self, edit_id: str) -> None:
        app: ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]
        detail_container = self.query_one("#match-detail", VerticalScroll)

        await detail_container.query("*").remove()

        if edit_id != "keyword-list-item":

            edit = app.resume_eng.get_edit_from_id(edit_id)
            if not edit:
                return
            await detail_container.mount(
                Label("Reason for Edit", classes="detail-section-title"),
                TextArea(
                    text=edit.reason,
                    read_only=True,
                    classes="detail-textarea job-textarea",
                ),
                Label("Original Text", classes="detail-section-title"),
                TextArea(
                    text=edit.original_text,
                    read_only=True,
                    classes="detail-textarea old-textarea",
                ),
                Label("Suggested New Text", classes="detail-section-title"),
                TextArea(
                    text=edit.new_text,
                    id="edit-textarea",
                    classes="detail-textarea new-textarea",
                ),
                Horizontal(
                    Button("Accept", id="accept-btn", variant="success"),
                    Button("Reject", id="reject-btn", variant="error"),
                    id="detail-buttons",
                ),
            )
            self.current_edit_id = edit_id
        else:
            keywords = app.resume_eng.keywords
            if not keywords:
                return
            await detail_container.mount(
                Label("Present Keywords:", classes="detail-section-title"),
                TextArea(
                    text=' '.join(keywords[0]),
                    read_only=True,
                    classes="detail-textarea job-textarea",
                ),
                Label("Missing Keywords", classes="detail-section-title"),
                TextArea(
                    text=' '.join(keywords[1]),
                    read_only=True,
                    classes="detail-textarea old-textarea",
                ),
            )
            self.current_edit_id = edit_id


    def on_button_pressed(self, event: Button.Pressed) -> None:
        app: ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]
        if event.button.id == "finish-btn":
            app.push_screen(FinalizeScreen())
        elif event.button.id == "accept-btn":
            new_text = self.query_one("#edit-textarea", TextArea).text
            success = app.resume_eng.apply_edit(self.current_edit_id, new_text)

            if success:
                # Refresh UI
                # 1. Update the overall score label
                current_overall, original_overall = (
                    app.resume_eng.get_overall_similarity()
                )
                display_overall = (
                    f"{original_overall:.2%} -> {current_overall:.2%}"
                    if current_overall != original_overall
                    else f"{current_overall:.2%}"
                )
                self.query_one("#overall-score-label", Label).update(
                    f"Overall Resume Match: {display_overall}"
                )

                # 2. Update the EditListItem in the list
                try:
                    list_item = self.query_one(
                        f"#edit-{self.current_edit_id}", EditListItem
                    )
                    list_item.refresh_edit_data()
                except NoMatches:
                    pass  # Widget might have been unmounted

                self.app.notify("Edit applied successfully!")
            else:
                self.app.notify(
                    "Failed to apply edit. Original text not found.", severity="error"
                )
        elif event.button.id == "reject-btn":
            edit = app.resume_eng.get_edit_from_id(self.current_edit_id)
            if edit:
                edit.status = "rejected"
                try:
                    list_item = self.query_one(
                        f"#edit-{self.current_edit_id}", EditListItem
                    )
                    list_item.refresh_edit_data()
                except NoMatches:
                    pass
                self.app.notify("Edit rejected.")


class FinalizeScreen(Screen):
    """Screen for finalizing and exporting the resume."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="finalize-container"):
            with Horizontal(id="format-buttons"):
                yield Button("Raw", id="btn-raw")
                yield Button("Markdown", id="btn-markdown")
                yield Button("Typst", id="btn-typst")
                yield Button("Export", id="btn-export", variant="success")
            yield TextArea(id="preview")

    def on_mount(self) -> None:
        app: ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]
        self.raw_text = app.resume_eng.current_resume.get_full_text()
        self.current_format = "raw"
        self.query_one("#preview", TextArea).text = self.raw_text

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-raw":
            self.current_format = "raw"
            self.query_one("#preview", TextArea).text = self.raw_text
        elif event.button.id == "btn-markdown":
            self.format_text("markdown")
        elif event.button.id == "btn-typst":
            self.format_text("typst")
        elif event.button.id == "btn-export":
            self.export_resume()

    @work(exclusive=True, thread=True)
    def format_text(self, format_type: str) -> None:
        app: ResumeMatcherApp = self.app  # ty:ignore[invalid-assignment]
        self.app.call_from_thread(self.app.notify, f"Formatting as {format_type}...")
        try:
            formatted_text = app.resume_eng.format_resume_text(
                self.raw_text, format_type
            )
            self.app.call_from_thread(self._update_preview, formatted_text, format_type)
        except Exception as e:
            self.app.call_from_thread(
                self.app.notify, f"Error formatting: {str(e)}", severity="error"
            )

    def _update_preview(self, text: str, format_type: str) -> None:
        self.current_format = format_type
        self.query_one("#preview", TextArea).text = text
        self.app.notify(f"Formatted as {format_type} successfully!")

    def export_resume(self) -> None:
        text = self.query_one("#preview", TextArea).text
        ext = "txt"
        if self.current_format == "markdown":
            ext = "md"
        elif self.current_format == "typst":
            ext = "typ"

        filename = f"resume_export.{ext}"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
            self.app.notify(f"Exported to {filename}")
        except Exception as e:
            self.app.notify(f"Error exporting: {str(e)}", severity="error")


class ResumeMatcherApp(App):
    CSS_PATH = "styles.tcss"

    def __init__(self, resume_eng: ResumeEngine):
        super().__init__()
        self.resume_eng = resume_eng

    def on_mount(self) -> None:
        self.push_screen(InputScreen())
