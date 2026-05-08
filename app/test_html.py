<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>State DOT — Application Tracker</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Barlow+Condensed:wght@300;400;500;600;700&family=Barlow:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0d1117;
    --bg2: #161b22;
    --bg3: #1f2733;
    --border: #2a3441;
    --accent: #f0a500;
    --accent2: #2ea8ff;
    --accent3: #3ddc97;
    --danger: #ff5c5c;
    --warn: #ffb300;
    --text: #e6edf3;
    --muted: #7d8590;
    --card: #1a2231;
  }

  * { margin:0; padding:0; box-sizing:border-box; }

  body {
    font-family: 'Barlow', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* GRID TEXTURE */
  body::before {
    content:'';
    position:fixed; inset:0;
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.18;
    pointer-events:none;
    z-index:0;
  }

  .wrapper { position:relative; z-index:1; }

  /* HEADER */
  header {
    display:flex; align-items:center; justify-content:space-between;
    padding: 18px 36px;
    border-bottom: 1px solid var(--border);
    background: rgba(13,17,23,0.9);
    backdrop-filter: blur(10px);
    position: sticky; top:0; z-index:100;
  }

  .logo {
    display:flex; align-items:center; gap:12px;
  }
  .logo-badge {
    width:36px; height:36px;
    background: var(--accent);
    clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
    display:flex; align-items:center; justify-content:center;
    font-family:'Barlow Condensed', sans-serif;
    font-weight:700; font-size:13px; color:#000;
    flex-shrink:0;
  }
  .logo-text { font-family:'Barlow Condensed', sans-serif; }
  .logo-text span:first-child {
    display:block; font-size:18px; font-weight:700; letter-spacing:2px; color:var(--accent);
  }
  .logo-text span:last-child {
    display:block; font-size:11px; font-weight:400; letter-spacing:3px; color:var(--muted); text-transform:uppercase;
  }

  .header-actions { display:flex; gap:10px; align-items:center; }

  .btn {
    font-family:'Barlow Condensed', sans-serif;
    font-weight:600; letter-spacing:1px; font-size:13px;
    padding:8px 20px; border:none; cursor:pointer;
    transition: all .2s; text-transform:uppercase;
  }
  .btn-primary {
    background: var(--accent); color:#000;
    clip-path: polygon(8px 0%, 100% 0%, calc(100% - 8px) 100%, 0% 100%);
  }
  .btn-primary:hover { background:#ffc030; transform:translateY(-1px); }
  .btn-ghost {
    background:transparent; color:var(--muted);
    border:1px solid var(--border);
  }
  .btn-ghost:hover { color:var(--text); border-color:var(--muted); }

  /* TABS */
  .tabs {
    display:flex; gap:0;
    padding: 0 36px;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }
  .tab {
    font-family:'Barlow Condensed', sans-serif;
    font-weight:600; letter-spacing:1.5px; font-size:13px;
    text-transform:uppercase; color:var(--muted);
    padding:14px 24px; cursor:pointer; border:none;
    background:none; transition:all .2s;
    border-bottom:2px solid transparent;
    position:relative;
  }
  .tab:hover { color:var(--text); }
  .tab.active { color:var(--accent); border-bottom-color:var(--accent); }
  .tab .badge {
    display:inline-block; background:var(--bg3);
    border:1px solid var(--border);
    font-size:10px; padding:1px 6px;
    border-radius:2px; margin-left:6px;
    color:var(--muted);
  }
  .tab.active .badge { background:rgba(240,165,0,0.15); border-color:var(--accent); color:var(--accent); }

  /* VIEWS */
  .view { display:none; padding:28px 36px; }
  .view.active { display:block; animation: fadeIn .3s ease; }
  @keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none} }

  /* DASHBOARD */
  .dash-title {
    font-family:'Barlow Condensed', sans-serif;
    font-size:11px; font-weight:500; letter-spacing:3px;
    text-transform:uppercase; color:var(--muted); margin-bottom:20px;
  }

  .kpi-grid {
    display:grid; grid-template-columns: repeat(4, 1fr);
    gap:14px; margin-bottom:28px;
  }
  .kpi {
    background:var(--card); border:1px solid var(--border);
    padding:20px 22px; position:relative; overflow:hidden;
    transition:transform .2s;
  }
  .kpi:hover { transform:translateY(-2px); }
  .kpi::before {
    content:''; position:absolute;
    top:0; left:0; right:0; height:2px;
  }
  .kpi.yellow::before { background:var(--accent); }
  .kpi.blue::before { background:var(--accent2); }
  .kpi.green::before { background:var(--accent3); }
  .kpi.red::before { background:var(--danger); }

  .kpi-label {
    font-size:11px; letter-spacing:2px; text-transform:uppercase;
    color:var(--muted); font-family:'DM Mono', monospace; margin-bottom:10px;
  }
  .kpi-num {
    font-family:'Barlow Condensed', sans-serif;
    font-size:40px; font-weight:700; line-height:1;
  }
  .kpi.yellow .kpi-num { color:var(--accent); }
  .kpi.blue .kpi-num { color:var(--accent2); }
  .kpi.green .kpi-num { color:var(--accent3); }
  .kpi.red .kpi-num { color:var(--danger); }
  .kpi-sub { font-size:12px; color:var(--muted); margin-top:6px; }

  .dash-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }

  .panel {
    background:var(--card); border:1px solid var(--border); padding:22px;
  }
  .panel-title {
    font-family:'Barlow Condensed', sans-serif;
    font-size:13px; font-weight:600; letter-spacing:2px;
    text-transform:uppercase; color:var(--muted);
    margin-bottom:16px; padding-bottom:10px;
    border-bottom:1px solid var(--border);
  }

  .bar-item { margin-bottom:14px; }
  .bar-label { display:flex; justify-content:space-between; font-size:13px; margin-bottom:5px; }
  .bar-track { height:6px; background:var(--bg3); border-radius:0; overflow:hidden; }
  .bar-fill { height:100%; border-radius:0; transition:width 1s cubic-bezier(.4,0,.2,1); }

  .status-list { display:flex; flex-direction:column; gap:8px; }
  .status-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:10px 14px; background:var(--bg3);
    border:1px solid var(--border);
    font-size:13px;
  }
  .status-dot { width:8px; height:8px; border-radius:50%; margin-right:8px; display:inline-block; }
  .status-row .left { display:flex; align-items:center; }

  /* FILTER BAR */
  .filter-bar {
    display:flex; gap:10px; flex-wrap:wrap;
    margin-bottom:20px; align-items:center;
  }
  .search-wrap { position:relative; flex:1; min-width:220px; }
  .search-wrap input {
    width:100%; background:var(--card);
    border:1px solid var(--border);
    color:var(--text); font-family:'Barlow', sans-serif; font-size:14px;
    padding:9px 12px 9px 36px; outline:none;
    transition:border-color .2s;
  }
  .search-wrap input:focus { border-color:var(--accent); }
  .search-icon {
    position:absolute; left:11px; top:50%; transform:translateY(-50%);
    color:var(--muted); font-size:14px; pointer-events:none;
  }
  select.filter-select {
    background:var(--card); border:1px solid var(--border);
    color:var(--text); font-family:'Barlow', sans-serif; font-size:13px;
    padding:9px 12px; outline:none; cursor:pointer;
    transition:border-color .2s; -webkit-appearance:none; appearance:none;
    padding-right:28px;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%237d8590'/%3E%3C/svg%3E");
    background-repeat:no-repeat; background-position:right 10px center;
  }
  select.filter-select:focus { border-color:var(--accent); }

  /* TABLE */
  .table-wrap { overflow-x:auto; }
  table { width:100%; border-collapse:collapse; }
  thead tr {
    background:var(--bg3);
    border-bottom:2px solid var(--accent);
  }
  thead th {
    font-family:'Barlow Condensed', sans-serif;
    font-weight:600; letter-spacing:1.5px; font-size:12px;
    text-transform:uppercase; color:var(--muted);
    padding:12px 14px; text-align:left; white-space:nowrap;
    cursor:pointer; user-select:none;
  }
  thead th:hover { color:var(--text); }
  tbody tr {
    border-bottom:1px solid var(--border);
    transition:background .15s; cursor:pointer;
  }
  tbody tr:hover { background:var(--bg3); }
  tbody td {
    padding:12px 14px; font-size:13px; vertical-align:middle;
  }
  .app-id { font-family:'DM Mono', monospace; font-size:12px; color:var(--muted); }
  .applicant-name { font-weight:500; }
  .applicant-sub { font-size:11px; color:var(--muted); margin-top:2px; font-family:'DM Mono', monospace; }

  .chip {
    display:inline-block; padding:3px 10px; font-size:11px;
    font-family:'Barlow Condensed', sans-serif; font-weight:600;
    letter-spacing:1px; text-transform:uppercase; border-radius:0;
  }
  .chip-planning  { background:rgba(46,168,255,.15); color:var(--accent2); border:1px solid rgba(46,168,255,.3); }
  .chip-construction { background:rgba(240,165,0,.15); color:var(--accent); border:1px solid rgba(240,165,0,.3); }
  .chip-incident  { background:rgba(255,92,92,.15); color:var(--danger); border:1px solid rgba(255,92,92,.3); }
  .chip-inspection { background:rgba(61,220,151,.15); color:var(--accent3); border:1px solid rgba(61,220,151,.3); }
  .chip-management { background:rgba(190,120,255,.15); color:#be78ff; border:1px solid rgba(190,120,255,.3); }

  .status-chip {
    display:inline-flex; align-items:center; gap:5px;
    padding:4px 10px; font-size:11px;
    font-family:'Barlow Condensed', sans-serif; font-weight:600;
    letter-spacing:1px; text-transform:uppercase;
    border:1px solid; border-radius:0;
  }
  .status-new      { color:#2ea8ff; border-color:rgba(46,168,255,.4); background:rgba(46,168,255,.08); }
  .status-review   { color:var(--warn); border-color:rgba(255,179,0,.4); background:rgba(255,179,0,.08); }
  .status-interview{ color:#be78ff; border-color:rgba(190,120,255,.4); background:rgba(190,120,255,.08); }
  .status-offer    { color:var(--accent3); border-color:rgba(61,220,151,.4); background:rgba(61,220,151,.08); }
  .status-hired    { color:var(--accent3); border-color:var(--accent3); background:rgba(61,220,151,.15); }
  .status-rejected { color:var(--danger); border-color:rgba(255,92,92,.4); background:rgba(255,92,92,.08); }

  .skills-wrap { display:flex; flex-wrap:wrap; gap:4px; }
  .skill-tag {
    background:var(--bg3); border:1px solid var(--border);
    font-size:10px; padding:2px 7px; color:var(--muted);
    font-family:'DM Mono', monospace;
  }

  /* MODAL */
  .modal-overlay {
    display:none; position:fixed; inset:0;
    background:rgba(0,0,0,.75); z-index:200;
    backdrop-filter:blur(4px);
    align-items:flex-start; justify-content:center;
    padding:40px 20px;
    overflow-y:auto;
  }
  .modal-overlay.open { display:flex; animation:fadeIn .2s ease; }
  .modal {
    background:var(--bg2); border:1px solid var(--border);
    width:100%; max-width:660px;
    position:relative;
  }
  .modal-header {
    display:flex; align-items:center; justify-content:space-between;
    padding:18px 24px; border-bottom:1px solid var(--border);
    background:var(--bg3);
  }
  .modal-header h3 {
    font-family:'Barlow Condensed', sans-serif;
    font-size:18px; font-weight:700; letter-spacing:1px; text-transform:uppercase;
  }
  .modal-close {
    background:none; border:none; color:var(--muted); cursor:pointer;
    font-size:20px; line-height:1; padding:4px 8px;
    transition:color .2s;
  }
  .modal-close:hover { color:var(--text); }
  .modal-body { padding:24px; }
  .modal-section { margin-bottom:20px; }
  .modal-section h4 {
    font-family:'Barlow Condensed', sans-serif;
    font-size:11px; font-weight:600; letter-spacing:2.5px;
    text-transform:uppercase; color:var(--muted);
    margin-bottom:10px; padding-bottom:6px;
    border-bottom:1px solid var(--border);
  }
  .field-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .field label {
    display:block; font-size:11px; letter-spacing:1px; text-transform:uppercase;
    color:var(--muted); margin-bottom:5px; font-family:'DM Mono', monospace;
  }
  .field input, .field select, .field textarea {
    width:100%; background:var(--bg3); border:1px solid var(--border);
    color:var(--text); font-family:'Barlow', sans-serif; font-size:13px;
    padding:9px 12px; outline:none; transition:border-color .2s;
  }
  .field input:focus, .field select:focus, .field textarea:focus {
    border-color:var(--accent);
  }
  .field textarea { resize:vertical; min-height:80px; }
  .field.full { grid-column:1/-1; }

  .skills-checklist {
    display:grid; grid-template-columns:repeat(3,1fr); gap:8px;
  }
  .skill-check { display:flex; align-items:center; gap:8px; cursor:pointer; }
  .skill-check input[type=checkbox] { accent-color:var(--accent); cursor:pointer; }
  .skill-check span { font-size:12px; }

  .modal-footer {
    display:flex; justify-content:flex-end; gap:10px;
    padding:16px 24px; border-top:1px solid var(--border);
    background:var(--bg3);
  }

  /* NOTES */
  .notes-thread { display:flex; flex-direction:column; gap:10px; }
  .note-item {
    background:var(--bg3); border:1px solid var(--border);
    padding:12px 14px; font-size:13px; line-height:1.6;
  }
  .note-meta { font-size:11px; color:var(--muted); font-family:'DM Mono', monospace; margin-bottom:5px; }
  .note-add { display:flex; gap:8px; margin-top:10px; }
  .note-add textarea {
    flex:1; background:var(--card); border:1px solid var(--border);
    color:var(--text); font-family:'Barlow', sans-serif; font-size:13px;
    padding:9px 12px; outline:none; resize:none; height:60px;
  }
  .note-add textarea:focus { border-color:var(--accent); }

  .empty-state {
    text-align:center; padding:60px 20px; color:var(--muted);
  }
  .empty-state .big { font-family:'Barlow Condensed',sans-serif; font-size:48px; color:var(--border); }
  .empty-state p { margin-top:8px; font-size:13px; }

  .table-footer {
    display:flex; justify-content:space-between; align-items:center;
    padding:12px 0; font-size:12px; color:var(--muted);
    font-family:'DM Mono', monospace; border-top:1px solid var(--border);
  }
  .pagination { display:flex; gap:4px; }
  .page-btn {
    background:var(--bg3); border:1px solid var(--border);
    color:var(--muted); font-size:12px; padding:4px 10px;
    cursor:pointer; font-family:'DM Mono', monospace;
    transition:all .2s;
  }
  .page-btn:hover, .page-btn.active { background:var(--accent); color:#000; border-color:var(--accent); }

  /* POSITIONS VIEW */
  .positions-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
  .pos-card {
    background:var(--card); border:1px solid var(--border);
    padding:20px; transition:transform .2s; cursor:pointer;
    position:relative; overflow:hidden;
  }
  .pos-card:hover { transform:translateY(-3px); }
  .pos-card::after {
    content:''; position:absolute; top:0; right:0;
    width:60px; height:60px;
    background:radial-gradient(circle at top right, rgba(240,165,0,.08), transparent);
  }
  .pos-category {
    font-size:10px; letter-spacing:2.5px; text-transform:uppercase;
    font-family:'DM Mono', monospace; margin-bottom:10px;
  }
  .pos-title {
    font-family:'Barlow Condensed', sans-serif;
    font-size:20px; font-weight:700; margin-bottom:6px;
  }
  .pos-dept { font-size:12px; color:var(--muted); margin-bottom:14px; }
  .pos-stats { display:flex; gap:16px; }
  .pos-stat { text-align:center; }
  .pos-stat .num {
    font-family:'Barlow Condensed', sans-serif;
    font-size:24px; font-weight:700;
  }
  .pos-stat .lbl { font-size:10px; color:var(--muted); letter-spacing:1px; text-transform:uppercase; }

  .progress-bar { height:4px; background:var(--bg3); margin-top:14px; overflow:hidden; }
  .progress-fill { height:100%; background:var(--accent); transition:width .8s ease; }

  /* TIMELINE badge */
  .timeline-badge {
    font-family:'DM Mono', monospace; font-size:10px;
    padding:2px 8px; background:var(--bg3);
    border:1px solid var(--border); color:var(--muted);
    white-space:nowrap;
  }

  @media(max-width:900px) {
    .kpi-grid { grid-template-columns:1fr 1fr; }
    .dash-grid { grid-template-columns:1fr; }
    .positions-grid { grid-template-columns:1fr 1fr; }
    header { padding:14px 18px; }
    .view { padding:20px 18px; }
  }
</style>
</head>
<body>
<div class="wrapper">

<!-- HEADER -->
<header>
  <div class="logo">
    <div class="logo-badge">DOT</div>
    <div class="logo-text">
      <span>Application Tracker</span>
      <span>Tennessee Department of Transportation</span>
    </div>
  </div>
  <div class="header-actions">
    <button class="btn btn-ghost" onclick="exportCSV()">⬇ Export</button>
    <button class="btn btn-primary" onclick="openNewModal()">+ New Applicant</button>
  </div>
</header>

<!-- TABS -->
<div class="tabs">
  <button class="tab active" onclick="switchTab('dashboard',this)">Dashboard</button>
  <button class="tab" onclick="switchTab('applications',this)">
    Applications <span class="badge" id="tab-count">0</span>
  </button>
  <button class="tab" onclick="switchTab('positions',this)">Open Positions</button>
</div>

<!-- DASHBOARD VIEW -->
<div class="view active" id="view-dashboard">
  <div class="dash-title">// Overview — All Applications</div>
  <div class="kpi-grid">
    <div class="kpi yellow"><div class="kpi-label">Total Applicants</div><div class="kpi-num" id="kpi-total">0</div><div class="kpi-sub">Across all positions</div></div>
    <div class="kpi blue"><div class="kpi-label">Under Review</div><div class="kpi-num" id="kpi-review">0</div><div class="kpi-sub">Awaiting evaluation</div></div>
    <div class="kpi green"><div class="kpi-label">Offers Extended</div><div class="kpi-num" id="kpi-offer">0</div><div class="kpi-sub">Hired + Offers</div></div>
    <div class="kpi red"><div class="kpi-label">Rejected</div><div class="kpi-num" id="kpi-rejected">0</div><div class="kpi-sub">Not proceeding</div></div>
  </div>
  <div class="dash-grid">
    <div class="panel">
      <div class="panel-title">Applications by Position Type</div>
      <div id="bar-chart"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Pipeline Status</div>
      <div class="status-list" id="pipeline-list"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Top Skills in Pool</div>
      <div id="skills-chart"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Recent Activity</div>
      <div id="recent-activity" class="status-list"></div>
    </div>
  </div>
</div>

<!-- APPLICATIONS VIEW -->
<div class="view" id="view-applications">
  <div class="filter-bar">
    <div class="search-wrap">
      <span class="search-icon">🔍</span>
      <input type="text" id="search-input" placeholder="Search by name, ID, position, skills…" oninput="renderTable()">
    </div>
    <select class="filter-select" id="filter-position" onchange="renderTable()">
      <option value="">All Positions</option>
      <option>Planning</option>
      <option>Construction</option>
      <option>Incident Management</option>
      <option>Inspection</option>
      <option>Project Management</option>
    </select>
    <select class="filter-select" id="filter-status" onchange="renderTable()">
      <option value="">All Statuses</option>
      <option>New</option>
      <option>Under Review</option>
      <option>Interview</option>
      <option>Offer Extended</option>
      <option>Hired</option>
      <option>Rejected</option>
    </select>
    <select class="filter-select" id="filter-exp" onchange="renderTable()">
      <option value="">All Experience</option>
      <option>0–2 yrs</option>
      <option>3–5 yrs</option>
      <option>6–10 yrs</option>
      <option>10+ yrs</option>
    </select>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th onclick="sortTable('id')">App ID ↕</th>
          <th onclick="sortTable('name')">Applicant ↕</th>
          <th onclick="sortTable('position')">Position ↕</th>
          <th>Skills</th>
          <th onclick="sortTable('experience')">Exp. ↕</th>
          <th onclick="sortTable('status')">Status ↕</th>
          <th onclick="sortTable('date')">Applied ↕</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody id="table-body"></tbody>
    </table>
  </div>
  <div class="table-footer">
    <span id="table-info">0 applicants</span>
    <div class="pagination" id="pagination"></div>
  </div>
</div>

<!-- POSITIONS VIEW -->
<div class="view" id="view-positions">
  <div class="positions-grid" id="positions-grid"></div>
</div>

<!-- MODAL: New / Edit Applicant -->
<div class="modal-overlay" id="modal-overlay" onclick="closeModalOutside(event)">
  <div class="modal">
    <div class="modal-header">
      <h3 id="modal-title">New Applicant</h3>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-body">
      <div class="modal-section">
        <h4>Personal Information</h4>
        <div class="field-grid">
          <div class="field">
            <label>First Name</label>
            <input type="text" id="f-fname" placeholder="Jane">
          </div>
          <div class="field">
            <label>Last Name</label>
            <input type="text" id="f-lname" placeholder="Doe">
          </div>
          <div class="field">
            <label>Email</label>
            <input type="email" id="f-email" placeholder="jane.doe@email.com">
          </div>
          <div class="field">
            <label>Phone</label>
            <input type="text" id="f-phone" placeholder="555-000-0000">
          </div>
        </div>
      </div>

      <div class="modal-section">
        <h4>Position & Experience</h4>
        <div class="field-grid">
          <div class="field">
            <label>Applying For</label>
            <select id="f-position">
              <option>Planning</option>
              <option>Construction</option>
              <option>Incident Management</option>
              <option>Inspection</option>
              <option>Project Management</option>
            </select>
          </div>
          <div class="field">
            <label>Years of Experience</label>
            <select id="f-experience">
              <option>0–2 yrs</option>
              <option>3–5 yrs</option>
              <option>6–10 yrs</option>
              <option>10+ yrs</option>
            </select>
          </div>
          <div class="field">
            <label>Status</label>
            <select id="f-status">
              <option>New</option>
              <option>Under Review</option>
              <option>Interview</option>
              <option>Offer Extended</option>
              <option>Hired</option>
              <option>Rejected</option>
            </select>
          </div>
          <div class="field">
            <label>Application Date</label>
            <input type="date" id="f-date">
          </div>
        </div>
      </div>

      <div class="modal-section">
        <h4>Skills & Certifications</h4>
        <div class="skills-checklist" id="skills-checklist"></div>
      </div>

      <div class="modal-section">
        <h4>Notes</h4>
        <div class="field">
          <textarea id="f-notes" placeholder="Add notes about this applicant…"></textarea>
        </div>
      </div>

      <!-- Notes thread (edit mode) -->
      <div class="modal-section" id="notes-thread-section" style="display:none">
        <h4>Activity Log</h4>
        <div class="notes-thread" id="notes-thread"></div>
        <div class="note-add">
          <textarea id="new-note-text" placeholder="Add a note…"></textarea>
          <button class="btn btn-primary" onclick="addNote()" style="align-self:flex-end">Add</button>
        </div>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
      <button class="btn btn-primary" id="modal-save-btn" onclick="saveApplicant()">Save Applicant</button>
    </div>
  </div>
</div>

<script>
// ─── DATA ────────────────────────────────────────────────────────────────────

const SKILLS_ALL = [
  'AutoCAD','Civil 3D','GIS / ESRI','Traffic Analysis','Environmental Compliance',
  'Structural Eng.','OSHA Certified','PMP Certified','Contract Admin',
  'Incident Command','MUTCD Standards','Drainage Design','Bridge Inspection',
  'Pavement Mgmt','Stormwater Mgmt','Right-of-Way','Budget Forecasting',
  'Scoping & Design','Construction Mgmt','Scheduling / CPM'
];

const POSITIONS = [
  { title:'Transportation Planner', dept:'Planning Division', category:'Planning', color:'#2ea8ff', open:3 },
  { title:'Construction Inspector', dept:'Field Operations', category:'Construction', color:'#f0a500', open:5 },
  { title:'Incident Manager', dept:'Operations Center', category:'Incident Management', color:'#ff5c5c', open:2 },
  { title:'Bridge Inspector', dept:'Structures Division', category:'Inspection', color:'#3ddc97', open:2 },
  { title:'Project Manager', dept:'Capital Programs', category:'Project Management', color:'#be78ff', open:4 },
  { title:'Traffic Engineer', dept:'Planning Division', category:'Planning', color:'#2ea8ff', open:3 },
];

const SAMPLE_DATA = [
  { id:'DOT-001', fname:'Marcus', lname:'Rivera', email:'m.rivera@email.com', phone:'555-201-3344', position:'Construction', experience:'6–10 yrs', status:'Interview', date:'2025-01-15', skills:['OSHA Certified','AutoCAD','Construction Mgmt','Scheduling / CPM'], notes:'Strong field background. Previous TDOT contractor.', log:[] },
  { id:'DOT-002', fname:'Alicia', lname:'Chen', email:'a.chen@email.com', phone:'555-422-8891', position:'Planning', experience:'3–5 yrs', status:'Under Review', date:'2025-01-17', skills:['GIS / ESRI','Traffic Analysis','Scoping & Design','AutoCAD'], notes:'Masters in Urban Planning.', log:[] },
  { id:'DOT-003', fname:'James', lname:'Okafor', email:'j.okafor@email.com', phone:'555-331-0022', position:'Incident Management', experience:'10+ yrs', status:'Offer Extended', date:'2025-01-10', skills:['Incident Command','OSHA Certified','MUTCD Standards'], notes:'Former state trooper. Excellent ICS credentials.', log:[{text:'Completed panel interview - highly recommended',date:'2025-01-22',user:'H. Mitchell'}] },
  { id:'DOT-004', fname:'Priya', lname:'Nair', email:'p.nair@email.com', phone:'555-108-7760', position:'Inspection', experience:'3–5 yrs', status:'New', date:'2025-01-21', skills:['Bridge Inspection','Structural Eng.','Civil 3D'], notes:'', log:[] },
  { id:'DOT-005', fname:'Derek', lname:'Thornton', email:'d.thornton@email.com', phone:'555-554-2299', position:'Project Management', experience:'10+ yrs', status:'Hired', date:'2025-01-05', skills:['PMP Certified','Contract Admin','Budget Forecasting','Scheduling / CPM','Construction Mgmt'], notes:'Exceptional candidate. 15 years highway project experience.', log:[{text:'Offer accepted. Start date Feb 3.',date:'2025-01-25',user:'R. Evans'}] },
  { id:'DOT-006', fname:'Sofia', lname:'Gutierrez', email:'s.gutierrez@email.com', phone:'555-773-4401', position:'Planning', experience:'0–2 yrs', status:'Rejected', date:'2025-01-08', skills:['GIS / ESRI'], notes:'Insufficient experience for senior role.', log:[] },
  { id:'DOT-007', fname:'Kevin', lname:'Park', email:'k.park@email.com', phone:'555-990-3312', position:'Construction', experience:'3–5 yrs', status:'Under Review', date:'2025-01-19', skills:['AutoCAD','OSHA Certified','Pavement Mgmt','Drainage Design'], notes:'Good technical skills. Schedule phone screen.', log:[] },
  { id:'DOT-008', fname:'Diane', lname:'Blackwell', email:'d.blackwell@email.com', phone:'555-114-8823', position:'Incident Management', experience:'6–10 yrs', status:'Interview', date:'2025-01-14', skills:['Incident Command','MUTCD Standards','Traffic Analysis'], notes:'', log:[] },
  { id:'DOT-009', fname:'Raj', lname:'Patel', email:'r.patel@email.com', phone:'555-339-0051', position:'Inspection', experience:'10+ yrs', status:'Under Review', date:'2025-01-20', skills:['Bridge Inspection','Structural Eng.','Stormwater Mgmt','Right-of-Way'], notes:'PE licensed. Very strong applicant.', log:[] },
  { id:'DOT-010', fname:'Linda', lname:'Morrison', email:'l.morrison@email.com', phone:'555-662-1178', position:'Project Management', experience:'6–10 yrs', status:'Interview', date:'2025-01-12', skills:['PMP Certified','Contract Admin','Budget Forecasting'], notes:'', log:[] },
  { id:'DOT-011', fname:'Tyler', lname:'Bounds', email:'t.bounds@email.com', phone:'555-887-4490', position:'Construction', experience:'0–2 yrs', status:'New', date:'2025-01-22', skills:['AutoCAD','OSHA Certified'], notes:'Recent grad. Entry-level candidate.', log:[] },
  { id:'DOT-012', fname:'Nina', lname:'Voss', email:'n.voss@email.com', phone:'555-445-8812', position:'Planning', experience:'3–5 yrs', status:'New', date:'2025-01-23', skills:['GIS / ESRI','Traffic Analysis','Environmental Compliance'], notes:'', log:[] },
];

let applicants = JSON.parse(localStorage.getItem('dot-apps') || 'null') || SAMPLE_DATA;
let editingId = null;
let sortKey = 'date';
let sortAsc = false;
let currentPage = 1;
const PER_PAGE = 8;

function save() { localStorage.setItem('dot-apps', JSON.stringify(applicants)); }
function nextId() { const n = applicants.length + 1; return 'DOT-' + String(n).padStart(3,'0'); }

// ─── TABS ─────────────────────────────────────────────────────────────────────
function switchTab(view, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('view-' + view).classList.add('active');
  if (view === 'dashboard') renderDashboard();
  if (view === 'applications') renderTable();
  if (view === 'positions') renderPositions();
}

// ─── DASHBOARD ────────────────────────────────────────────────────────────────
function renderDashboard() {
  const total = applicants.length;
  const review = applicants.filter(a => a.status === 'Under Review').length;
  const offer = applicants.filter(a => ['Offer Extended','Hired'].includes(a.status)).length;
  const rej = applicants.filter(a => a.status === 'Rejected').length;

  animNum('kpi-total', total);
  animNum('kpi-review', review);
  animNum('kpi-offer', offer);
  animNum('kpi-rejected', rej);

  // Bar chart
  const posGroups = {};
  applicants.forEach(a => { posGroups[a.position] = (posGroups[a.position]||0) + 1; });
  const posColors = { Planning:'#2ea8ff', Construction:'#f0a500', 'Incident Management':'#ff5c5c', Inspection:'#3ddc97', 'Project Management':'#be78ff' };
  const maxPos = Math.max(...Object.values(posGroups), 1);
  document.getElementById('bar-chart').innerHTML = Object.entries(posGroups).map(([k,v]) =>
    `<div class="bar-item">
      <div class="bar-label"><span>${k}</span><span style="color:${posColors[k]||'#fff'};font-family:'DM Mono',monospace;font-size:12px">${v}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${(v/maxPos*100)}%;background:${posColors[k]||'#888'}"></div></div>
    </div>`
  ).join('');

  // Pipeline
  const statuses = ['New','Under Review','Interview','Offer Extended','Hired','Rejected'];
  const statColors = { New:'#2ea8ff','Under Review':'#ffb300','Interview':'#be78ff','Offer Extended':'#3ddc97','Hired':'#3ddc97','Rejected':'#ff5c5c' };
  document.getElementById('pipeline-list').innerHTML = statuses.map(s => {
    const cnt = applicants.filter(a => a.status === s).length;
    return `<div class="status-row">
      <div class="left"><span class="status-dot" style="background:${statColors[s]}"></span>${s}</div>
      <span style="font-family:'DM Mono',monospace;font-size:13px;color:${statColors[s]}">${cnt}</span>
    </div>`;
  }).join('');

  // Skills chart
  const skillCount = {};
  applicants.forEach(a => (a.skills||[]).forEach(s => { skillCount[s] = (skillCount[s]||0)+1; }));
  const topSkills = Object.entries(skillCount).sort((a,b)=>b[1]-a[1]).slice(0,6);
  const maxSk = topSkills[0]?.[1] || 1;
  document.getElementById('skills-chart').innerHTML = topSkills.map(([k,v]) =>
    `<div class="bar-item">
      <div class="bar-label"><span style="font-size:12px">${k}</span><span style="color:var(--accent3);font-family:'DM Mono',monospace;font-size:12px">${v}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${(v/maxSk*100)}%;background:var(--accent3)"></div></div>
    </div>`
  ).join('');

  // Recent activity
  const recent = [...applicants].sort((a,b) => new Date(b.date) - new Date(a.date)).slice(0,5);
  document.getElementById('recent-activity').innerHTML = recent.map(a =>
    `<div class="status-row" onclick="openEditModal('${a.id}')" style="cursor:pointer">
      <div class="left" style="flex-direction:column;align-items:flex-start;gap:2px">
        <span style="font-size:13px;font-weight:500">${a.fname} ${a.lname}</span>
        <span style="font-size:11px;color:var(--muted);font-family:'DM Mono',monospace">${a.position}</span>
      </div>
      <span class="timeline-badge">${a.date}</span>
    </div>`
  ).join('');
}

function animNum(id, target) {
  let start = 0;
  const el = document.getElementById(id);
  const step = () => {
    start += Math.ceil((target - start) / 6);
    el.textContent = start;
    if (start < target) requestAnimationFrame(step);
    else el.textContent = target;
  };
  requestAnimationFrame(step);
}

// ─── TABLE ────────────────────────────────────────────────────────────────────
function getFiltered() {
  const q = document.getElementById('search-input').value.toLowerCase();
  const pos = document.getElementById('filter-position').value;
  const st = document.getElementById('filter-status').value;
  const exp = document.getElementById('filter-exp').value;
  return applicants.filter(a => {
    const name = (a.fname + ' ' + a.lname + ' ' + a.id + ' ' + a.position + ' ' + (a.skills||[]).join(' ')).toLowerCase();
    return (!q || name.includes(q))
      && (!pos || a.position === pos)
      && (!st || a.status === st)
      && (!exp || a.experience === exp);
  }).sort((a,b) => {
    let av = a[sortKey], bv = b[sortKey];
    if (sortKey === 'name') { av = a.fname; bv = b.fname; }
    if (av < bv) return sortAsc ? -1 : 1;
    if (av > bv) return sortAsc ? 1 : -1;
    return 0;
  });
}

function sortTable(key) {
  if (sortKey === key) sortAsc = !sortAsc;
  else { sortKey = key; sortAsc = true; }
  renderTable();
}

const posChip = { Planning:'chip-planning', Construction:'chip-construction', 'Incident Management':'chip-incident', Inspection:'chip-inspection', 'Project Management':'chip-management' };
const stClass = { New:'status-new','Under Review':'status-review', Interview:'status-interview','Offer Extended':'status-offer', Hired:'status-hired', Rejected:'status-rejected' };

function renderTable() {
  const data = getFiltered();
  const total = data.length;
  const pages = Math.ceil(total / PER_PAGE) || 1;
  if (currentPage > pages) currentPage = 1;
  const slice = data.slice((currentPage-1)*PER_PAGE, currentPage*PER_PAGE);

  document.getElementById('tab-count').textContent = applicants.length;
  document.getElementById('table-info').textContent = `Showing ${slice.length} of ${total} applicant${total!==1?'s':''}`;

  const tbody = document.getElementById('table-body');
  if (!slice.length) {
    tbody.innerHTML = `<tr><td colspan="8"><div class="empty-state"><div class="big">∅</div><p>No applicants match your filters.</p></div></td></tr>`;
  } else {
    tbody.innerHTML = slice.map(a => `
      <tr onclick="openEditModal('${a.id}')">
        <td><span class="app-id">${a.id}</span></td>
        <td>
          <div class="applicant-name">${a.fname} ${a.lname}</div>
          <div class="applicant-sub">${a.email}</div>
        </td>
        <td><span class="chip ${posChip[a.position]||''}">${a.position}</span></td>
        <td><div class="skills-wrap">${(a.skills||[]).slice(0,3).map(s=>`<span class="skill-tag">${s}</span>`).join('')}${a.skills.length>3?`<span class="skill-tag">+${a.skills.length-3}</span>`:''}</div></td>
        <td style="font-family:'DM Mono',monospace;font-size:12px;color:var(--muted)">${a.experience}</td>
        <td><span class="status-chip ${stClass[a.status]||''}">${a.status}</span></td>
        <td style="font-family:'DM Mono',monospace;font-size:12px;color:var(--muted)">${a.date}</td>
        <td><button class="btn btn-ghost" style="padding:4px 12px;font-size:11px" onclick="event.stopPropagation();openEditModal('${a.id}')">Edit</button></td>
      </tr>`).join('');
  }

  // Pagination
  const pg = document.getElementById('pagination');
  pg.innerHTML = '';
  for (let i=1;i<=pages;i++) {
    const b = document.createElement('button');
    b.className = 'page-btn' + (i===currentPage?' active':'');
    b.textContent = i;
    b.onclick = ()=>{ currentPage=i; renderTable(); };
    pg.appendChild(b);
  }
}

// ─── POSITIONS ────────────────────────────────────────────────────────────────
function renderPositions() {
  const grid = document.getElementById('positions-grid');
  grid.innerHTML = POSITIONS.map(p => {
    const apps = applicants.filter(a => a.position === p.category);
    const hired = apps.filter(a => a.status === 'Hired').length;
    const pct = Math.round(hired / p.open * 100);
    return `<div class="pos-card" onclick="filterToPosition('${p.category}')">
      <div class="pos-category" style="color:${p.color}">${p.category}</div>
      <div class="pos-title">${p.title}</div>
      <div class="pos-dept">${p.dept}</div>
      <div class="pos-stats">
        <div class="pos-stat"><div class="num" style="color:${p.color}">${apps.length}</div><div class="lbl">Applied</div></div>
        <div class="pos-stat"><div class="num" style="color:var(--warn)">${apps.filter(a=>a.status==='Interview').length}</div><div class="lbl">Interview</div></div>
        <div class="pos-stat"><div class="num" style="color:var(--accent3)">${hired}</div><div class="lbl">Hired</div></div>
        <div class="pos-stat"><div class="num" style="color:var(--muted)">${p.open}</div><div class="lbl">Open</div></div>
      </div>
      <div class="progress-bar"><div class="progress-fill" style="width:${pct}%;background:${p.color}"></div></div>
    </div>`;
  }).join('');
}

function filterToPosition(pos) {
  document.getElementById('filter-position').value = pos;
  currentPage = 1;
  document.querySelectorAll('.tab')[1].click();
}

// ─── MODAL ────────────────────────────────────────────────────────────────────
function buildSkillsChecklist(selected=[]) {
  document.getElementById('skills-checklist').innerHTML = SKILLS_ALL.map(s =>
    `<label class="skill-check">
      <input type="checkbox" value="${s}" ${selected.includes(s)?'checked':''}>
      <span>${s}</span>
    </label>`
  ).join('');
}

function openNewModal() {
  editingId = null;
  document.getElementById('modal-title').textContent = 'New Applicant';
  document.getElementById('modal-save-btn').textContent = 'Save Applicant';
  document.getElementById('f-fname').value = '';
  document.getElementById('f-lname').value = '';
  document.getElementById('f-email').value = '';
  document.getElementById('f-phone').value = '';
  document.getElementById('f-position').value = 'Planning';
  document.getElementById('f-experience').value = '0–2 yrs';
  document.getElementById('f-status').value = 'New';
  document.getElementById('f-date').value = new Date().toISOString().split('T')[0];
  document.getElementById('f-notes').value = '';
  document.getElementById('notes-thread-section').style.display = 'none';
  buildSkillsChecklist([]);
  document.getElementById('modal-overlay').classList.add('open');
}

function openEditModal(id) {
  const a = applicants.find(x => x.id === id);
  if (!a) return;
  editingId = id;
  document.getElementById('modal-title').textContent = `Edit — ${a.id}`;
  document.getElementById('modal-save-btn').textContent = 'Update Applicant';
  document.getElementById('f-fname').value = a.fname;
  document.getElementById('f-lname').value = a.lname;
  document.getElementById('f-email').value = a.email;
  document.getElementById('f-phone').value = a.phone;
  document.getElementById('f-position').value = a.position;
  document.getElementById('f-experience').value = a.experience;
  document.getElementById('f-status').value = a.status;
  document.getElementById('f-date').value = a.date;
  document.getElementById('f-notes').value = a.notes;
  buildSkillsChecklist(a.skills||[]);

  // Notes log
  document.getElementById('notes-thread-section').style.display = 'block';
  renderNotesThread(a);
  document.getElementById('modal-overlay').classList.add('open');
}

function renderNotesThread(a) {
  document.getElementById('notes-thread').innerHTML = (a.log||[]).length
    ? a.log.map(n => `<div class="note-item"><div class="note-meta">${n.date} · ${n.user}</div>${n.text}</div>`).join('')
    : `<div style="color:var(--muted);font-size:13px">No activity logged yet.</div>`;
}

function addNote() {
  const text = document.getElementById('new-note-text').value.trim();
  if (!text || !editingId) return;
  const a = applicants.find(x => x.id === editingId);
  if (!a) return;
  a.log = a.log || [];
  a.log.push({ text, date: new Date().toISOString().split('T')[0], user: 'Current User' });
  save();
  renderNotesThread(a);
  document.getElementById('new-note-text').value = '';
}

function closeModal() { document.getElementById('modal-overlay').classList.remove('open'); }
function closeModalOutside(e) { if (e.target === document.getElementById('modal-overlay')) closeModal(); }

function getCheckedSkills() {
  return [...document.querySelectorAll('#skills-checklist input:checked')].map(i => i.value);
}

function saveApplicant() {
  const fname = document.getElementById('f-fname').value.trim();
  const lname = document.getElementById('f-lname').value.trim();
  if (!fname || !lname) { alert('Please enter first and last name.'); return; }

  const data = {
    fname, lname,
    email: document.getElementById('f-email').value.trim(),
    phone: document.getElementById('f-phone').value.trim(),
    position: document.getElementById('f-position').value,
    experience: document.getElementById('f-experience').value,
    status: document.getElementById('f-status').value,
    date: document.getElementById('f-date').value,
    notes: document.getElementById('f-notes').value,
    skills: getCheckedSkills()
  };

  if (editingId) {
    const idx = applicants.findIndex(x => x.id === editingId);
    applicants[idx] = { ...applicants[idx], ...data };
  } else {
    applicants.push({ id: nextId(), ...data, log: [] });
  }
  save();
  closeModal();
  renderTable();
  renderDashboard();
  document.getElementById('tab-count').textContent = applicants.length;
}

// ─── EXPORT ───────────────────────────────────────────────────────────────────
function exportCSV() {
  const headers = ['ID','First Name','Last Name','Email','Phone','Position','Experience','Status','Applied','Skills','Notes'];
  const rows = applicants.map(a => [
    a.id, a.fname, a.lname, a.email, a.phone, a.position, a.experience, a.status, a.date,
    (a.skills||[]).join('; '), a.notes.replace(/,/g,' ')
  ]);
  const csv = [headers, ...rows].map(r => r.map(c => `"${c}"`).join(',')).join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'dot-applicants.csv'; a.click();
}

// ─── INIT ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('tab-count').textContent = applicants.length;
  renderDashboard();
});
</script>
</body>
</html>