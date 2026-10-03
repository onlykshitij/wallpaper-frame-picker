# SPDX-License-Identifier: AGPL-3.0-or-later
"""The "Install dependencies" pop-up, drawn with Tk because Qt may not be
installed yet. Falls back to a terminal prompt when no window can open.

Standard library only.
"""
import queue
import sys
import threading
import time

from .paths import data_dir
from .uvtools import SetupError

BG, PANEL, TEXT, DIM, ACCENT, BAD = "#16181d", "#1d2027", "#e6e8ee", "#9aa1ad", "#4c8dff", "#f0626a"


def ask_and_install(rows, system, install, gui=True, assume_yes=False):
    """Shows what is missing and installs it when the user agrees.

    rows: [(name, purpose, size in MB or None)] of downloads.
    system: (packages, command) of system packages to install, or None.
    install: install(progress) does the work; it raises SetupError.
    Returns True once everything is installed, False if the user quit.
    Every line of output also goes to install.log in the data folder."""
    install = _logged(install)
    if gui and not assume_yes:
        try:
            return _Dialog(rows, system, install).run()
        except _NoWindow:
            pass
    return _console(rows, system, install, assume_yes)


class _NoWindow(Exception):
    pass


def log_path():
    return data_dir() / "install.log"


def _logged(install):
    """Wraps install() so its output is written to install.log as well."""
    def run(progress):
        with open(log_path(), "a", encoding="utf-8") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} installing dependencies\n")

            def both(line):
                log.write(line + "\n")
                log.flush()
                progress(line)
            try:
                install(both)
            except SetupError as e:
                log.write(f"FAILED: {e}\n")
                raise SetupError(f"{e}\n\nThe full log is in {log_path()}") from e
            log.write("done\n")
    return run


def _summary(rows):
    total = sum(r[2] for r in rows if r[2])
    return f"About {total} MB to download." if total else ""


def _console(rows, system, install, assume_yes):
    out = sys.stderr
    print("Wallpaper Frame Picker needs these dependencies before it can start:", file=out)
    for name, purpose, mb in rows:
        print(f"  {name}: {purpose}" + (f", {mb} MB" if mb else ""), file=out)
    if system:
        print(f"  System packages (installed with your password): {', '.join(system[0])}", file=out)
    if _summary(rows):
        print(_summary(rows), file=out)
    if not assume_yes:
        if not (sys.stdin and sys.stdin.isatty()):
            print("Run the app from a terminal, or with --install-deps, to install them.", file=out)
            return False
        if input("Install dependencies now? [Y/n] ").strip().lower() not in ("", "y", "yes"):
            return False
    try:
        install(lambda line: print("  " + line, file=out, flush=True))
    except SetupError as e:
        print(f"Could not install the dependencies: {e}", file=out)
        return False
    return True


class _Dialog:
    def __init__(self, rows, system, install):
        try:
            import tkinter as tk
            from tkinter import font, ttk
            self.root = tk.Tk()
        except Exception as e:   # no tkinter, or no display
            raise _NoWindow from e
        self.tk, self.install, self.ok = tk, install, False
        self.events = queue.Queue()
        root = self.root
        root.withdraw()   # shown once laid out, by _place()
        root.title("Wallpaper Frame Picker")
        try:
            from pathlib import Path
            self.icon = tk.PhotoImage(file=str(Path(__file__).with_name("assets") / "icon.png"))
            root.iconphoto(True, self.icon)
        except Exception:
            pass   # no icon is fine
        root.configure(bg=BG, padx=22, pady=18)
        root.resizable(False, False)
        base = font.nametofont("TkDefaultFont")
        size = abs(base.cget("size")) or 10
        heading = base.copy()
        heading.configure(size=size + 4, weight="bold")
        body = base.copy()
        small = base.copy()
        small.configure(size=max(7, size - 2))
        self.small = small

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Bar.Horizontal.TProgressbar", troughcolor=PANEL, background=ACCENT,
                        bordercolor=PANEL, lightcolor=ACCENT, darkcolor=ACCENT)

        def label(parent, text, f, fg=TEXT, **kw):
            w = tk.Label(parent, text=text, font=f, fg=fg, bg=parent["bg"], justify="left", anchor="w", **kw)
            return w

        label(root, "Install dependencies", heading).pack(anchor="w")
        label(root, "Wallpaper Frame Picker needs these before it can start. They are downloaded once "
                    "and kept for later launches.", body, wraplength=470).pack(anchor="w", pady=(6, 10))

        box = tk.Frame(root, bg=PANEL, padx=12, pady=10)
        box.pack(fill="x")
        for i, (name, purpose, mb) in enumerate(rows):
            label(box, name, small).grid(row=i, column=0, sticky="w", padx=(0, 14))
            label(box, purpose, small, DIM).grid(row=i, column=1, sticky="w")
            label(box, f"{mb} MB" if mb else "", small, DIM).grid(row=i, column=2, sticky="e", padx=(14, 0))
        box.grid_columnconfigure(1, weight=1)
        if system:
            n = len(rows)
            label(box, "System packages", small).grid(row=n, column=0, sticky="nw", padx=(0, 14), pady=(8, 0))
            label(box, ", ".join(system[0]) + "\nInstalled with your password.", small, DIM,
                  wraplength=330).grid(row=n, column=1, columnspan=2, sticky="w", pady=(8, 0))
        if _summary(rows):
            label(root, _summary(rows), small, DIM).pack(anchor="w", pady=(6, 0))

        self.status = label(root, "", small, DIM, wraplength=470)
        self.status.pack(anchor="w", fill="x", pady=(10, 0))
        self.bar = ttk.Progressbar(root, mode="indeterminate", style="Bar.Horizontal.TProgressbar", length=470)

        # install output, shown once installing starts and kept open if it fails
        mono = font.nametofont("TkFixedFont").copy()
        mono.configure(size=max(7, size - 2))
        self.logbox = tk.Frame(root, bg=PANEL)
        self.log = tk.Text(self.logbox, height=11, width=72, font=mono, bg=PANEL, fg=DIM, relief="flat",
                           wrap="char", highlightthickness=0, bd=0, padx=8, pady=6, state="disabled",
                           insertbackground=TEXT)
        scroll = ttk.Scrollbar(self.logbox, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        buttons = tk.Frame(root, bg=BG)
        buttons.pack(fill="x", pady=(14, 0))
        self.go = tk.Button(buttons, text="Install dependencies", command=self._start, bg=ACCENT, fg="white",
                            activebackground="#5d99ff", activeforeground="white", relief="flat",
                            font=body, padx=14, pady=5, cursor="hand2", highlightthickness=0, bd=0)
        self.go.pack(side="right")
        self.quit = tk.Button(buttons, text="Quit", command=root.destroy, bg="#2a2f3a", fg=TEXT,
                              activebackground="#323846", activeforeground=TEXT, relief="flat",
                              font=body, padx=14, pady=5, highlightthickness=0, bd=0)
        self.quit.pack(side="right", padx=(0, 8))
        self.copy = tk.Button(buttons, text="Copy log", command=self._copy_log, bg="#2a2f3a", fg=TEXT,
                              activebackground="#323846", activeforeground=TEXT, relief="flat",
                              font=small, padx=10, pady=5, highlightthickness=0, bd=0)
        self._place()

    def _place(self):
        """Centers the window. Marking it as a dialog matters under XWayland,
        where Tk's idea of the screen does not match the monitors and some
        compositors otherwise open it in a corner."""
        try:
            self.root.attributes("-type", "dialog")   # X11 only
        except self.tk.TclError:
            pass
        self.root.update_idletasks()
        self.root.eval("tk::PlaceWindow . center")

    def run(self):
        self.root.mainloop()
        return self.ok

    def _start(self):
        self.go.configure(state="disabled", text="Installing…")
        self.quit.configure(state="disabled")
        self.status.configure(fg=DIM, text="Starting…")
        self.copy.pack_forget()
        self.bar.pack(anchor="w", pady=(6, 0), before=self.go.master)
        self.logbox.pack(fill="both", expand=True, pady=(8, 0), before=self.go.master)
        self.bar.start(12)
        threading.Thread(target=self._work, daemon=True).start()
        self.root.after(100, self._poll)

    def _work(self):
        try:
            self.install(lambda line: self.events.put(("line", line)))
            self.events.put(("done", None))
        except SetupError as e:
            self.events.put(("error", str(e)))
        except Exception as e:   # anything unexpected still reaches the user
            self.events.put(("error", f"{type(e).__name__}: {e}"))

    def _append(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _copy_log(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log.get("1.0", "end"))
        self.copy.configure(text="Copied")

    def _poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "line":
                    self.status.configure(text=value[-160:])
                    self._append(value)
                elif kind == "done":
                    self.ok = True
                    self.root.destroy()
                    return
                else:
                    self.bar.stop()
                    self.bar.pack_forget()
                    self.status.configure(fg=BAD, text=value.split("\n")[0][:300])
                    self._append("")
                    self._append(value)
                    self.go.configure(state="normal", text="Try again")
                    self.quit.configure(state="normal")
                    self.copy.pack(side="left")
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)
