"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LAYER 4 — LIVE TERMINAL DASHBOARD                                           ║
║  dashboard.py                                                                ║
║                                                                              ║
║  A Rich-powered terminal UI that visualises the digital twin system in       ║
║  real time.  Refreshed every tick by main.py.                                ║
║                                                                              ║
║  PANELS:                                                                     ║
║    ① Header      — tick progress bar + system status                        ║
║    ② Queue       — all patients ranked by DPS with live vitals + trends      ║
║    ③ Resources   — ICU / HDU / Ward / CT / Doctor utilisation bars          ║
║    ④ Events      — recent scheduler decisions + clinical event alerts        ║
║    ⑤ Footer      — urgency tier counts                                       ║
║                                                                              ║
║  REQUIRES:  pip install rich                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from rich.console  import Console
from rich.table    import Table
from rich.panel    import Panel
from rich.layout   import Layout
from rich.live     import Live
from rich.text     import Text
from rich          import box

from heuristics    import dps_label, DPS_IMMEDIATE, DPS_URGENT, DPS_SEMI_URGENT
from scheduler     import ResourceType, RESOURCE_LABELS
from dt  import VITAL_RANGES

console = Console()


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def dps_color(dps) -> str:
    if dps is None:                return "white"
    if dps >= DPS_IMMEDIATE:       return "bold red"
    if dps >= DPS_URGENT:          return "bold yellow"
    if dps >= DPS_SEMI_URGENT:     return "yellow"
    return "green"


def vital_color(val, vital: str) -> str:
    """Returns a Rich colour string based on clinical alarm level."""
    if val is None:
        return "white"
    lo_crit, lo_warn, hi_warn, hi_crit = VITAL_RANGES[vital]
    if vital in ("SpO2", "GCS"):
        if val <= lo_crit: return "bold red"
        if val <= lo_warn: return "yellow"
        return "green"
    else:
        if val <= lo_crit or val >= hi_crit: return "bold red"
        if val <= lo_warn or val >= hi_warn: return "yellow"
        return "green"


def trend_icon(direction: str) -> str:
    """Returns a coloured Rich markup arrow for a trend direction."""
    if "rising"  in direction: return "[red]↑[/red]"
    if "falling" in direction: return "[green]↓[/green]"
    if "stable"  in direction: return "[cyan]→[/cyan]"
    return "[dim]?[/dim]"


def colored(val, color: str) -> str:
    return f"[{color}]{val}[/{color}]"


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD CLASS
# ══════════════════════════════════════════════════════════════════════════════

class Dashboard:
    """
    Live terminal dashboard using Rich's Live display.

    Instantiated once in main.py.  Call render() every tick.
    Call stop() at the end to cleanly restore the terminal.

    Layout (approximate):
    ┌─────────────────────── HEADER ──────────────────────────┐
    │ ① Tick progress bar                                      │
    ├──────────────── QUEUE ─────────────┬──── SIDEBAR ────────┤
    │                                    │ ③ Resources          │
    │ ② Patient queue                    ├─────────────────────┤
    │    ranked by DPS                   │ ④ Events & decisions │
    └────────────────────────────────────┴─────────────────────┘
    │ ⑤ Footer — urgency tier summary                          │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(self, manager, scheduler):
        self.manager   = manager
        self.scheduler = scheduler
        self._layout   = self._build_layout()
        self._live     = Live(
            self._layout,
            console=console,
            refresh_per_second=4,
            screen=True,          # full-screen takeover
        )
        self._live.start()

    # ══════════════════════════════════════════════════════════════════════
    #  LAYOUT SKELETON
    # ══════════════════════════════════════════════════════════════════════

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header",  size=3),
            Layout(name="main",    ratio=1),
            Layout(name="footer",  size=3),
        )
        layout["main"].split_row(
            Layout(name="queue",   ratio=3),
            Layout(name="sidebar", ratio=2),
        )
        layout["sidebar"].split_column(
            Layout(name="resources", ratio=2),
            Layout(name="events",    ratio=3),
        )
        return layout

    # ══════════════════════════════════════════════════════════════════════
    #  PUBLIC: render() — called every tick from main.py
    # ══════════════════════════════════════════════════════════════════════

    def render(self, tick: int, max_ticks: int,
               decisions: list, all_events: list):
        """Refresh every panel in the dashboard."""
        self._layout["header"].update(
            self._render_header(tick, max_ticks))
        self._layout["queue"].update(
            self._render_queue())
        self._layout["resources"].update(
            self._render_resources())
        self._layout["events"].update(
            self._render_events(decisions, all_events))
        self._layout["footer"].update(
            self._render_footer(tick))

    def stop(self):
        """Cleanly exit the Rich live display."""
        if self._live:
            self._live.stop()

    # ══════════════════════════════════════════════════════════════════════
    #  ① HEADER — tick progress bar
    # ══════════════════════════════════════════════════════════════════════

    def _render_header(self, tick: int, max_ticks: int) -> Panel:
        pct    = tick / max_ticks
        filled = int(pct * 44)
        bar    = f"[cyan]{'█' * filled}[/cyan][dim]{'░' * (44 - filled)}[/dim]"

        ranked      = self.manager.get_ranked_queue()
        n_improving = sum(1 for t in ranked if t.improving)
        n_det       = len(ranked) - n_improving
        sync_str    = "[green]🟢 ALL TWINS LIVE[/green]"

        text = Text.from_markup(
            f"  [bold white on blue] 🏥 HOSPITAL DIGITAL TWIN [/bold white on blue]"
            f"  [bold]T{tick:03d}/{max_ticks}[/bold]  {bar}  {pct*100:.0f}%  "
            f"   [green]⬆{n_improving} improving[/green]  "
            f"[red]⬇{n_det} deteriorating[/red]  "
            f"   {sync_str}"
        )
        return Panel(text, style="blue", padding=(0, 1))

    # ══════════════════════════════════════════════════════════════════════
    #  ② QUEUE — patient priority table
    # ══════════════════════════════════════════════════════════════════════

    def _render_queue(self) -> Panel:
        table = Table(
            title="[bold]📋  Patient Priority Queue — Sorted by DPS (highest = most urgent)[/bold]",
            box=box.SIMPLE_HEAVY,
            show_lines=False,
            padding=(0, 1),
            expand=True,
        )

        # Column definitions
        table.add_column("#",         width=3,  justify="right",  style="bold dim")
        table.add_column("ID",        width=8,  style="cyan bold")
        table.add_column("Condition", width=22)
        table.add_column("DPS",       width=6,  justify="right")
        table.add_column("Urgency",   width=15)
        table.add_column("HR",        width=5,  justify="right")
        table.add_column("BP",        width=5,  justify="right")
        table.add_column("SpO₂",      width=5,  justify="right")
        table.add_column("RR",        width=4,  justify="right")
        table.add_column("GCS",       width=4,  justify="right")
        table.add_column("Trend",     width=5,  justify="center")
        table.add_column("Traj",      width=11)
        table.add_column("Bed",       width=10)

        ranked = self.manager.get_ranked_queue()

        for rank, twin in enumerate(ranked, 1):
            dps  = twin.dps
            v    = twin.vitals
            sc   = twin.scores

            # ── Trend arrows for SpO2 + GCS (most clinically important) ──
            trends   = twin.get_all_trends(window=10)
            spo2_dir = trends.get("SpO2", {}).get("direction", "?")
            gcs_dir  = trends.get("GCS",  {}).get("direction", "?")
            trend_str = trend_icon(spo2_dir) + trend_icon(gcs_dir)

            # ── Trajectory arrow ──
            traj = "[green]⬆ IMPR[/green]" if twin.improving else "[red]⬇  DET[/red]"

            # ── Bed allocation ──
            alloc   = self.scheduler.get_patient_allocation(twin.patient_id)
            bed_str = ""
            if   ResourceType.ICU_BED   in alloc: bed_str = "[bold red]ICU[/bold red]"
            elif ResourceType.HDU_BED   in alloc: bed_str = "[yellow]HDU[/yellow]"
            elif ResourceType.WARD_BED  in alloc: bed_str = "[green]Ward[/green]"
            else:                                  bed_str = "[dim]waiting[/dim]"
            if ResourceType.DOCTOR     in alloc:  bed_str += " [white]👨‍⚕️[/white]"
            if ResourceType.CT_SCANNER in alloc:  bed_str += " [cyan]CT[/cyan]"

            # ── Colour-coded vital values ──
            def cv(val, key):
                c = vital_color(val, key)
                return f"[{c}]{val:.0f}[/{c}]" if val is not None else "[dim]—[/dim]"

            dps_str = (f"[{dps_color(dps)}]{dps:.1f}[/{dps_color(dps)}]"
                       if dps is not None else "[dim]—[/dim]")

            table.add_row(
                str(rank),
                twin.patient_id,
                twin.condition.replace("_", " "),
                dps_str,
                dps_label(dps),
                cv(v.get("HR"),     "HR"),
                cv(v.get("BP_sys"), "BP_sys"),
                cv(v.get("SpO2"),   "SpO2"),
                cv(v.get("RR"),     "RR"),
                f"[{vital_color(v.get('GCS'), 'GCS')}]{v.get('GCS', '—')}[/]",
                trend_str,
                traj,
                bed_str,
            )

        return Panel(table, title="Queue", border_style="cyan", padding=(0, 0))

    # ══════════════════════════════════════════════════════════════════════
    #  ③ RESOURCES — utilisation bars
    # ══════════════════════════════════════════════════════════════════════

    def _render_resources(self) -> Panel:
        summary = self.scheduler.get_resource_summary()
        lines   = []

        for rtype, data in summary.items():
            used  = data["used"]
            cap   = data["capacity"]
            pct   = data["pct"]

            # Colour: red if ≥80%, yellow if ≥50%, green otherwise
            color = "red" if pct >= 80 else ("yellow" if pct >= 50 else "green")

            # Build block bar (4 chars per slot)
            bar_full  = "█" * used
            bar_empty = "░" * (cap - used)
            bar       = f"[{color}]{bar_full}[/{color}][dim]{bar_empty}[/dim]"

            label    = RESOURCE_LABELS[rtype]
            pct_str  = f"[{color}]{pct:.0f}%[/{color}]"

            # List current occupants compactly
            pts = ", ".join(data["patients"]) if data["patients"] else "—"
            lines.append(
                f"  [bold]{label:<12}[/bold] [{bar}] "
                f"[bold]{used}/{cap}[/bold] {pct_str}\n"
                f"    [dim]{pts}[/dim]"
            )

        return Panel(
            Text.from_markup("\n".join(lines)),
            title="[bold]🏥 Resource Utilisation[/bold]",
            border_style="yellow",
        )

    # ══════════════════════════════════════════════════════════════════════
    #  ④ EVENTS — scheduler decisions + clinical alerts
    # ══════════════════════════════════════════════════════════════════════

    def _render_events(self, decisions: list, all_events: list) -> Panel:
        lines = []

        # ── Recent scheduler decisions ────────────────────────────────────
        recent = self.scheduler.get_recent_decisions(7)
        if recent:
            lines.append("[bold cyan]Scheduler Decisions (latest):[/bold cyan]")
            action_colors = {
                "ADMIT":     "green",
                "STEP_DOWN": "yellow",
                "PREEMPT":   "bold red",
                "ASSIGN":    "cyan",
                "RELEASE":   "dim",
            }
            for d in reversed(recent):
                ac = action_colors.get(d.action, "white")
                truncated_reason = d.reason[:38]
                lines.append(
                    f"  [dim]T{d.tick:03d}[/dim] "
                    f"[{ac}]{d.action:<10}[/{ac}] "
                    f"[cyan]{d.patient_id:<8}[/cyan] "
                    f"[dim]{truncated_reason}[/dim]"
                )

        # ── Recent clinical events ────────────────────────────────────────
        if all_events:
            lines.append("")
            lines.append("[bold red]Clinical Events (latest):[/bold red]")
            for ev in all_events[-5:]:
                lines.append(
                    f"  [dim]T{ev['tick']:03d}[/dim] "
                    f"[red]{ev['event'][:52]}[/red]"
                )

        if not lines:
            lines = ["[dim]  No decisions yet — waiting for first tick...[/dim]"]

        return Panel(
            Text.from_markup("\n".join(lines)),
            title="[bold]📡 Events & Decisions[/bold]",
            border_style="magenta",
        )

    # ══════════════════════════════════════════════════════════════════════
    #  ⑤ FOOTER — urgency tier counts
    # ══════════════════════════════════════════════════════════════════════

    def _render_footer(self, tick: int) -> Panel:
        ranked = self.manager.get_ranked_queue()
        n_imm  = sum(1 for t in ranked if (t.dps or 0) >= DPS_IMMEDIATE)
        n_urg  = sum(1 for t in ranked if DPS_URGENT  <= (t.dps or 0) < DPS_IMMEDIATE)
        n_semi = sum(1 for t in ranked if DPS_SEMI_URGENT <= (t.dps or 0) < DPS_URGENT)
        n_rout = sum(1 for t in ranked if (t.dps or 0) < DPS_SEMI_URGENT)

        n_dec  = len(self.scheduler.decision_log)

        text = Text.from_markup(
            f"  [bold red]🔴 IMMEDIATE: {n_imm}[/bold red]    "
            f"[bold yellow]🟠 URGENT: {n_urg}[/bold yellow]    "
            f"[yellow]🟡 SEMI-URGENT: {n_semi}[/yellow]    "
            f"[green]🟢 ROUTINE: {n_rout}[/green]"
            f"       [dim]Decisions made: {n_dec}  |  "
            f"Tick: {tick}  |  "
            f"Sync: LIVE ✓[/dim]"
        )
        return Panel(text, style="dim", padding=(0, 1))