"""
Periodic Knowledge Base Refresh Scheduler
==========================================
Wraps APScheduler's BackgroundScheduler to periodically re-check registered
URL sources and re-ingest their content into the target knowledge base.

State lives at module level (not st.session_state) because it must persist
across Streamlit's per-interaction script reruns for the life of the running
process. Registered sources are also persisted to disk so they survive app
restarts.
"""

import os
import json

from apscheduler.schedulers.background import BackgroundScheduler

import knowledge_expander

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCES_CONFIG_PATH = os.path.join(_SRC_DIR, "sources_config.json")

_scheduler = None
_last_run_results = []


def _load_sources() -> list:
    if os.path.exists(SOURCES_CONFIG_PATH):
        with open(SOURCES_CONFIG_PATH) as f:
            return json.load(f)
    return []


def _save_sources(sources: list):
    with open(SOURCES_CONFIG_PATH, "w") as f:
        json.dump(sources, f, indent=2)


def get_sources() -> list:
    return _load_sources()


def add_source(kb_name: str, url: str):
    sources = _load_sources()
    if not any(s["kb"] == kb_name and s["url"] == url for s in sources):
        sources.append({"kb": kb_name, "url": url})
        _save_sources(sources)


def remove_source(kb_name: str, url: str):
    sources = _load_sources()
    sources = [s for s in sources if not (s["kb"] == kb_name and s["url"] == url)]
    _save_sources(sources)


def _run_scheduled_refresh():
    global _last_run_results
    results = []
    for src in _load_sources():
        try:
            result = knowledge_expander.ingest_url(src["kb"], src["url"])
            results.append({"kb": src["kb"], "url": src["url"], **result})
        except Exception as e:
            results.append({"kb": src["kb"], "url": src["url"], "status": "error", "reason": str(e)})
    _last_run_results = results


def get_last_run_results() -> list:
    return _last_run_results


def start_scheduler(interval_minutes: int = 60):
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.reschedule_job("kb_refresh", trigger="interval", minutes=interval_minutes)
        return _scheduler

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(_run_scheduled_refresh, "interval", minutes=interval_minutes, id="kb_refresh")
    _scheduler.start()
    return _scheduler


def stop_scheduler():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def is_running() -> bool:
    return _scheduler is not None and _scheduler.running


def run_now():
    """Trigger an immediate refresh of all registered sources, synchronously."""
    _run_scheduled_refresh()
    return _last_run_results
