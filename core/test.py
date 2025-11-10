from textual.app import App, ComposeResult
from textual.containers import HorizontalGroup, VerticalScroll
from textual.widgets import Footer, Header, Button, Digits

class TimeDisplay(Digits):
    """A simple time display showing hours, minutes, seconds, and centiseconds."""

class Stopwatch(HorizontalGroup):
    def compose(self) -> ComposeResult:
        yield Button("Start", id="start", variant="success")
        yield Button("Stop", id="stop", variant="error")
        yield Button("Reset", id="reset")
        yield TimeDisplay("00:00:00:00")



class StopwatchApp(App): 

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        yield VerticalScroll(Stopwatch(), Stopwatch(), Stopwatch())

    def action_toggle_dark(self) -> None:
        self.theme = (
            "textual_dark" if self.theme == "textual_light" else "textual_light"
        )

if __name__ == "__main__":
    app = StopwatchApp()
    app.run()