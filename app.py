from core.engine import ResumeEngine
from tui.screens import ResumeMatcherApp

if __name__ == "__main__":
    resume_eng = ResumeEngine()
    app = ResumeMatcherApp(resume_eng)
    app.run()
