import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import subprocess
import time
import threading
import os
import webbrowser

steps = [
    ("Scraping reviews", "scrape.py"),
    ("Cleaning data", "clean.py"),
    ("Performing EDA", "eda.py"),
    ("Training model", "train.py"),
    ("Generating report", "report_generator.py")
]

companies_available = ["Flipkart", "Amazon", "Meesho", "Myntra"]
selected_companies = []
log_lines = []
theme_mode = "dark"

def log_message(msg):
    global log_lines
    log_lines.append(msg)
    text_area.insert(tk.END, msg + "\n")
    text_area.see(tk.END)

def run_step(label, script, skip_env=False):
    log_message(f"🔄 {label} started...")
    start = time.time()
    try:
        if not skip_env:
            os.environ["SELECTED_COMPANIES"] = ",".join(selected_companies)
        subprocess.run(["python", script], check=True)
        duration = time.time() - start
        log_message(f"✅ {label} completed in {duration:.2f} seconds.")
        if "report" in script:
            report_path = os.path.abspath("model_output/final_report.pdf")
            if os.path.exists(report_path):
                webbrowser.open_new(report_path)
    except subprocess.CalledProcessError:
        log_message(f"❌ Error during {label}. Check {script}.")
        messagebox.showerror("Error", f"{label} failed.")
        return False
    return True

def run_full_pipeline():
    text_area.delete("1.0", tk.END)
    log_lines.clear()
    for i, (label, script) in enumerate(steps, 1):
        success = run_step(label, script)
        if not success:
            return
        progress_var.set(int((i / len(steps)) * 100))
        root.update_idletasks()
        time.sleep(0.5)
    log_message("\n🎉 All steps completed successfully!")
    messagebox.showinfo("Pipeline Complete", "Pipeline completed and report is available!")

def start_pipeline_thread():
    global selected_companies
    selected_companies = [company for company, var in company_vars.items() if var.get()]
    if not selected_companies:
        messagebox.showwarning("No Company Selected", "Select at least one company to proceed.")
        return
    threading.Thread(target=run_full_pipeline, daemon=True).start()

def create_retry_buttons(frame):
    for label, script in steps:
        b = tk.Button(frame, text=f"↻ Retry: {label}", command=lambda l=label, s=script: run_step(l, s), font=("Segoe UI", 9), bg=colors["button_bg"], fg=colors["button_fg"])
        b.pack(pady=2, fill="x", padx=5)

def export_logs():
    with open("pipeline_logs.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    messagebox.showinfo("Logs Saved", "Logs exported to pipeline_logs.txt")

def toggle_theme():
    global theme_mode, colors
    theme_mode = "light" if theme_mode == "dark" else "dark"
    colors = LIGHT_COLORS if theme_mode == "light" else DARK_COLORS
    apply_theme()

def apply_theme():
    root.config(bg=colors["bg"])
    main_frame.config(bg=colors["bg"])
    title_label.config(bg=colors["bg"], fg=colors["fg"])
    run_btn.config(bg=colors["run_btn_bg"], fg=colors["run_btn_fg"])
    theme_btn.config(bg=colors["button_bg"], fg=colors["button_fg"])
    export_btn.config(bg=colors["button_bg"], fg=colors["button_fg"])
    retry_frame.config(bg=colors["bg"], fg=colors["fg"])
    for widget in retry_frame.winfo_children():
        widget.config(bg=colors["button_bg"], fg=colors["button_fg"])
    for chk in company_checks:
        chk.config(bg=colors["bg"], fg=colors["fg"], selectcolor=colors["selectcolor"])
    text_area.config(bg=colors["log_bg"], fg=colors["log_fg"])

DARK_COLORS = {
    "bg": "#2e2e2e",
    "fg": "white",
    "button_bg": "#444",
    "button_fg": "white",
    "run_btn_bg": "green",
    "run_btn_fg": "white",
    "log_bg": "#1e1e1e",
    "log_fg": "lime",
    "selectcolor": "black"
}
LIGHT_COLORS = {
    "bg": "#f2f2f2",
    "fg": "#222",
    "button_bg": "#ddd",
    "button_fg": "#000",
    "run_btn_bg": "#28a745",
    "run_btn_fg": "white",
    "log_bg": "white",
    "log_fg": "black",
    "selectcolor": "white"
}
colors = DARK_COLORS

# This content will be inserted before main root window creation
import tkinter as tk
from tkinter import ttk

def show_splash():
    splash = tk.Toplevel()
    splash.overrideredirect(True)
    splash.geometry("400x250+500+300")
    splash.configure(bg="black")

    label = tk.Label(splash, text="🚀 Initializing...", font=("Segoe UI", 16, "bold"), fg="lime", bg="black")
    label.pack(pady=60)

    progress = ttk.Progressbar(splash, mode="indeterminate", length=300)
    progress.pack(pady=10)
    progress.start()

    splash.after(2500, splash.destroy)  # Auto-close after 2.5 seconds
    splash.wait_window(splash)


root = tk.Tk()
root.withdraw()

def show_splash():
    splash = tk.Toplevel()
    splash.overrideredirect(True)
    splash.geometry("400x250+500+300")
    splash.configure(bg="black")

    label = tk.Label(splash, text="🚀 Initializing...", font=("Segoe UI", 16, "bold"), fg="lime", bg="black")
    label.pack(pady=60)

    progress = ttk.Progressbar(splash, mode="indeterminate", length=300)
    progress.pack(pady=10)
    progress.start()

    splash.after(2500, splash.destroy)
    splash.wait_window(splash)

show_splash()
root.deiconify()

root.title("Customer Churn Analysis - Pro UI")
root.geometry("780x680")
root.configure(bg=colors["bg"])

main_frame = tk.Frame(root, bg=colors["bg"])
main_frame.pack(pady=10)

title_label = tk.Label(main_frame, text="Customer Churn Pipeline", font=("Segoe UI", 16), bg=colors["bg"], fg=colors["fg"])
title_label.pack(pady=5)

# Company checkboxes
company_vars = {}
company_checks = []
for company in companies_available:
    var = tk.BooleanVar(value=True)
    chk = tk.Checkbutton(main_frame, text=company, variable=var, bg=colors["bg"], fg=colors["fg"], selectcolor=colors["selectcolor"], font=("Segoe UI", 10))
    chk.pack(anchor="w", padx=20)
    company_vars[company] = var
    company_checks.append(chk)

# Run full pipeline
run_btn = tk.Button(main_frame, text="▶ Run Full Pipeline", command=start_pipeline_thread, font=("Segoe UI", 12), bg=colors["run_btn_bg"], fg=colors["run_btn_fg"])
run_btn.pack(pady=10)

# Retry section
retry_frame = tk.LabelFrame(main_frame, text="Manual Retry Steps", font=("Segoe UI", 10, "bold"), bg=colors["bg"], fg=colors["fg"], bd=2, relief="groove")
retry_frame.pack(pady=5, fill="x", padx=10)
create_retry_buttons(retry_frame)

# Bottom controls
bottom_frame = tk.Frame(main_frame, bg=colors["bg"])
bottom_frame.pack(pady=10)

theme_btn = tk.Button(bottom_frame, text="🌗 Toggle Theme", command=toggle_theme, bg=colors["button_bg"], fg=colors["button_fg"])
theme_btn.grid(row=0, column=0, padx=10)

export_btn = tk.Button(bottom_frame, text="📝 Export Logs", command=export_logs, bg=colors["button_bg"], fg=colors["button_fg"])
export_btn.grid(row=0, column=1, padx=10)

# Progress
progress_var = tk.IntVar()
progress = ttk.Progressbar(main_frame, variable=progress_var, maximum=100, length=520)
progress.pack(pady=10)

# Log area
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=20, font=("Consolas", 10), bg=colors["log_bg"], fg=colors["log_fg"])
text_area.pack(padx=10, pady=10)

root.mainloop()