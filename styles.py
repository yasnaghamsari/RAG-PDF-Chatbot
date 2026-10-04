"""
styles.py
---------
Custom CSS for the Streamlit interface. Kept in its own file so that app.py
only contains application logic. The base colours are defined in
.streamlit/config.toml; this file adds cards, chips and spacing on top.
"""

CUSTOM_CSS = """
<style>
:root {
    --brand: #4F46E5;
    --brand-dark: #3730A3;
    --brand-soft: #EEF2FF;
    --card-bg: #FFFFFF;
    --border: #E5E7EB;
    --muted: #6B7280;
    --ok: #059669;
    --warn: #D97706;
    --err: #DC2626;
}

/* ---------- Page layout ---------- */
.block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 1300px; }

/* ---------- Header banner ---------- */
.hero {
    background: linear-gradient(120deg, var(--brand) 0%, #7C3AED 60%, #DB2777 100%);
    border-radius: 18px;
    padding: 1.6rem 2rem;
    color: #FFFFFF;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
}
.hero h1 { color: #FFFFFF; font-size: 2rem; margin: 0; padding: 0; font-weight: 800; }
.hero p  { color: rgba(255,255,255,0.88); margin: 0.3rem 0 0 0; font-size: 1.02rem; }
.hero .pipeline {
    margin-top: 0.8rem; font-size: 0.8rem; letter-spacing: 0.02em;
    color: rgba(255,255,255,0.8); font-family: ui-monospace, monospace;
}

/* ---------- Statistic cards ---------- */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.9rem;
    margin-bottom: 1.2rem;
}
.stat-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.stat-card .label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
.stat-card .value { font-size: 1.7rem; font-weight: 700; color: #111827; line-height: 1.3; }
.stat-card .icon  { float: right; font-size: 1.4rem; opacity: 0.85; }

/* ---------- Generic panel ---------- */
.panel-title { font-weight: 700; font-size: 1.05rem; margin: 0.2rem 0 0.6rem 0; color: #111827; }

/* ---------- Document list ---------- */
.doc-item {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-left: 4px solid var(--ok);
    border-radius: 10px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    word-break: break-word;
}
.doc-item.warning { border-left-color: var(--warn); }
.doc-item.error   { border-left-color: var(--err); }
.doc-item .name   { font-weight: 600; color: #111827; }
.doc-item .meta   { color: var(--muted); font-size: 0.8rem; }
.doc-item .note   { color: var(--warn); font-size: 0.78rem; margin-top: 0.2rem; }
.doc-item.error .note { color: var(--err); }

/* ---------- Source citation chips ---------- */
.sources { margin-top: 0.6rem; display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
.sources .label { font-size: 0.78rem; color: var(--muted); font-weight: 600; margin-right: 0.2rem; }
.chip {
    background: var(--brand-soft);
    color: var(--brand-dark);
    border: 1px solid #C7D2FE;
    border-radius: 999px;
    padding: 0.15rem 0.65rem;
    font-size: 0.78rem;
    font-weight: 500;
    word-break: break-all;
}
.chip.none { background: #FEF3C7; color: #92400E; border-color: #FDE68A; }

/* ---------- Retrieved context passages ---------- */
.passage {
    background: #F9FAFB;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.7rem;
    font-size: 0.86rem;
    white-space: pre-wrap;
    word-break: break-word;
}
.passage-head { font-size: 0.8rem; color: var(--muted); margin-bottom: 0.35rem; }
.passage-head b { color: var(--brand-dark); }
.score-bar { height: 5px; background: #E5E7EB; border-radius: 4px; margin: 0.25rem 0 0.5rem 0; }
.score-bar > div { height: 5px; background: var(--brand); border-radius: 4px; }

/* ---------- Welcome / empty state ---------- */
.welcome {
    background: var(--card-bg);
    border: 1px dashed #C7D2FE;
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
}
.welcome h3 { margin-top: 0; }
.welcome ol { margin-bottom: 0; }

/* ---------- Sidebar (colours come from [theme.sidebar] in config.toml) ---------- */
[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] .stDownloadButton button {
    width: 100%; border-radius: 10px; font-weight: 600;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { border-radius: 12px; }
.sidebar-brand { font-size: 1.5rem; font-weight: 800; margin-bottom: 0; }
.sidebar-sub   { font-size: 0.82rem; color: #9CA3AF; margin-top: 0; }
.key-ok   { color: #34D399; font-size: 0.85rem; font-weight: 600; }
.key-miss { color: #FBBF24; font-size: 0.85rem; font-weight: 600; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}
.answer-meta { font-size: 0.74rem; color: var(--muted); margin-top: 0.4rem; }

/* ---------- Small screens ---------- */
@media (max-width: 640px) {
    .hero { padding: 1.1rem 1.2rem; }
    .hero h1 { font-size: 1.5rem; }
    .hero .pipeline { display: none; }
    .stat-card .value { font-size: 1.35rem; }
}
</style>
"""
