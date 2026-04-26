import os
from datetime import date, datetime, timedelta
from functools import wraps

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)

import analytics


app = Flask(__name__)

# IMPORTANT: change this to a long random string before using in production
app.secret_key = "CHANGE_ME_TO_RANDOM_SECRET_KEY"


# ---------------------------------------------------------------------------
# Simple user store
# ---------------------------------------------------------------------------
# You (admin) edit this mapping to add/remove users who can log in.
# Keys are usernames, values are plaintext passwords.
USERS: dict[str, str] = {
    "admin": "Admin@123",
    # Example extra user for the guard:
    # "guard": "GuardPassword123",
}


# ---------------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------------
# Host (LAN IP or hostname) used in stream links. Override with CAMERA_HOST env.
# Defaults to the request host so the dashboard works across IP changes.
CAMERA_HOST = os.environ.get("CAMERA_HOST", "").strip()

# (display_name, slug_for_db, port)
CAMERA_DEFS = [
    ("DHA_CAM1", "dha-cam1", 8182),
    ("DHA_CAM2", "dha-cam2", 8186),
    ("MM_CAM1",  "mm-cam1",  8187),
    ("MM_CAM2",  "mm-cam2",  8188),
    ("FSD1",     "fsd1",     8183),
    ("FSD2",     "fsd2",     8184),
    ("FSD3",     "fsd3",     8185),
]


def _camera_host() -> str:
    if CAMERA_HOST:
        return CAMERA_HOST
    # Fall back to the host the user opened the dashboard on
    host = request.host.split(":")[0]
    return host


def _build_camera_list():
    host = _camera_host()
    return [
        {"name": f"{name} ({port})", "slug": slug,
         "url": f"http://{host}:{port}/video_ai"}
        for name, slug, port in CAMERA_DEFS
    ]


CAMERAS_INDEX = {slug: name for name, slug, _ in CAMERA_DEFS}


def login_required(func):
    """Decorator to require a logged-in user for a view."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return func(*args, **kwargs)

    return wrapper


LOGIN_HTML = """
<!doctype html>
<title>AI Cameras Login</title>
<style>
  body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: radial-gradient(circle at top, #0f172a, #020617);
         color: #e5e7eb; margin: 0; height: 100vh; display:flex;
         align-items:center; justify-content:center; }
  .card { background: rgba(15, 23, 42, 0.98); padding: 26px 30px;
          border-radius: 14px; box-shadow: 0 24px 60px rgba(0,0,0,.8);
          width: 320px; border: 1px solid #1f2937; }
  h2 { margin: 0 0 6px 0; font-size: 20px; }
  p.sub { margin: 0 0 18px 0; font-size: 13px; color:#9ca3af; }
  label { display:block; margin-top:10px; font-size: 13px; color:#d1d5db; }
  input { width:100%; padding:9px 10px; margin-top:5px; border-radius:8px;
          border:1px solid #374151; background:#020617; color:#e5e7eb;
          outline:none; font-size:14px; }
  input:focus { border-color:#3b82f6; box-shadow:0 0 0 1px #3b82f6; }
  button { margin-top:18px; width:100%; padding:10px; border-radius:8px;
           border:none; background:linear-gradient(135deg,#3b82f6,#22c55e);
           color:#f9fafb; font-weight:600; cursor:pointer; font-size:14px; }
  button:hover { filter:brightness(1.05); }
  .error { color:#f97373; margin-top:10px; font-size:13px; }
</style>
<div class="card">
  <h2>AI Cameras Login</h2>
  <p class="sub">Sign in to view the clinic camera streams.</p>
  <form method="post">
    <label>Username
      <input name="username" autocomplete="username" required>
    </label>
    <label>Password
      <input name="password" type="password" autocomplete="current-password" required>
    </label>
    <button type="submit">Sign in</button>
    {% if error %}
      <div class="error">{{ error }}</div>
    {% endif %}
  </form>
</div>
"""


CAMERAS_HTML = """
<!doctype html>
<title>AI Cameras</title>
<style>
  body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background:#020617; color:#e5e7eb; margin:0; }
  header { padding:14px 22px; background:#030712; border-bottom:1px solid #111827;
           display:flex; justify-content:space-between; align-items:center;
           position:sticky; top:0; z-index:10; }
  h2 { margin:0; font-size:18px; }
  .small { font-size:12px; color:#9ca3af; margin-right:10px; }
  main { padding:20px 22px 28px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; }
  .cam { background: radial-gradient(circle at top left,#111827,#020617);
         border-radius:14px; padding:14px 16px; border:1px solid #111827;
         box-shadow:0 14px 30px rgba(0,0,0,.65); }
  .cam h3 { margin:0 0 8px 0; font-size:16px; }
  .cam p { margin:0; font-size:12px; color:#9ca3af; }
  .cam a { display:inline-block; margin-top:10px; padding:8px 11px;
           border-radius:999px; background:#10b981; color:#022c22;
           text-decoration:none; font-weight:600; font-size:13px; }
  .cam a:hover { background:#059669; }
  button.logout { background:none; border:1px solid #374151; color:#e5e7eb;
                 padding:6px 11px; border-radius:999px; cursor:pointer;
                 font-size:13px; }
  button.logout:hover { background:#111827; }
</style>
<header>
  <div>
    <h2>AI Cameras</h2>
    <div class="small">Secure access to DHA and Faisalabad counters</div>
  </div>
  <div>
    <a href="{{ url_for('reports') }}" style="color:#93c5fd;margin-right:14px;text-decoration:none;">Reports</a>
    <form method="post" action="{{ url_for('logout') }}" style="display:inline">
      <span class="small">Logged in as {{ user }}</span>
      <button class="logout" type="submit">Logout</button>
    </form>
  </div>
</header>
<main>
  <div class="grid">
    {% for cam in cameras %}
      <div class="cam">
        <h3>{{ cam.name }}</h3>
        <p>{{ cam.url }}</p>
        <a href="{{ cam.url }}" target="_blank" rel="noopener">Open stream</a>
      </div>
    {% endfor %}
  </div>
</main>
"""


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if USERS.get(username) == password:
            session["user"] = username
            return redirect(url_for("cameras"))
        error = "Invalid username or password"
    return render_template_string(LOGIN_HTML, error=error)


@app.post("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/")
@login_required
def cameras():
    return render_template_string(
        CAMERAS_HTML,
        cameras=_build_camera_list(),
        user=session.get("user"),
    )


# ---------------------------------------------------------------------------
# Reports dashboard
# ---------------------------------------------------------------------------


def _today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


def _validate_camera(slug: str) -> str:
    if slug not in CAMERAS_INDEX:
        abort(404, f"Unknown camera: {slug}")
    return slug


def _validate_date(s: str) -> str:
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        abort(400, f"Invalid date: {s}")
    return s


@app.get("/reports")
@login_required
def reports():
    cam = request.args.get("camera") or CAMERA_DEFS[0][1]
    cam = _validate_camera(cam)
    today = _today_str()
    one_month_ago = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    date_str = _validate_date(request.args.get("date") or today)
    from_date = _validate_date(
        request.args.get("from_date") or request.args.get("from") or one_month_ago
    )
    to_date = _validate_date(
        request.args.get("to_date") or request.args.get("to") or today
    )
    year = int(request.args.get("year") or date.today().year)

    sessions = analytics.list_sessions(cam, date_str)
    daily = analytics.daily_summary(cam, from_date, to_date)
    monthly = analytics.monthly_summary(cam, year)

    avg_today = (sum(s["total_wait_s"] for s in sessions) / len(sessions)) if sessions else 0
    return render_template_string(
        REPORTS_HTML,
        user=session.get("user"),
        cameras_index=CAMERAS_INDEX,
        camera=cam,
        camera_name=CAMERAS_INDEX[cam],
        date=date_str,
        from_date=from_date,
        to_date=to_date,
        year=year,
        sessions=sessions,
        avg_today=avg_today,
        daily=daily,
        monthly=monthly,
    )


@app.get("/reports/daily.xlsx")
@login_required
def reports_daily_xlsx():
    cam = _validate_camera(request.args.get("camera", ""))
    date_str = _validate_date(request.args.get("date") or _today_str())
    data = analytics.make_daily_xlsx(cam, date_str)
    return Response(
        data,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{cam}_{date_str}.xlsx"'},
    )


@app.get("/reports/range.xlsx")
@login_required
def reports_range_xlsx():
    cam = _validate_camera(request.args.get("camera", ""))
    from_date = _validate_date(
        request.args.get("from_date") or request.args.get("from") or ""
    )
    to_date = _validate_date(
        request.args.get("to_date") or request.args.get("to") or ""
    )
    data = analytics.make_range_xlsx(cam, from_date, to_date)
    return Response(
        data,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{cam}_daily_{from_date}_{to_date}.xlsx"'},
    )


@app.get("/reports/monthly.xlsx")
@login_required
def reports_monthly_xlsx():
    cam = _validate_camera(request.args.get("camera", ""))
    year = int(request.args.get("year") or date.today().year)
    data = analytics.make_monthly_xlsx(cam, year)
    return Response(
        data,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{cam}_monthly_{year}.xlsx"'},
    )


REPORTS_HTML = """
<!doctype html>
<title>Wait-time reports</title>
<style>
  body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background:#020617; color:#e5e7eb; margin:0; }
  header { padding:14px 22px; background:#030712; border-bottom:1px solid #111827;
           display:flex; justify-content:space-between; align-items:center;
           position:sticky; top:0; z-index:10; }
  h2 { margin:0; font-size:18px; }
  .small { font-size:12px; color:#9ca3af; }
  nav a { color:#93c5fd; margin-right:14px; text-decoration:none; font-size:14px; }
  nav a:hover { text-decoration:underline; }
  main { padding:20px 22px 28px; max-width: 1100px; margin: 0 auto; }
  section { background:#0b1220; border:1px solid #111827; border-radius:12px;
            padding:16px 18px; margin-bottom:16px; }
  section h3 { margin:0 0 10px 0; font-size:15px; color:#e5e7eb; }
  form.controls { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
  form.controls label { font-size:12px; color:#9ca3af; display:flex; flex-direction:column; gap:4px; }
  form.controls select, form.controls input { padding:7px 9px; background:#020617;
        color:#e5e7eb; border:1px solid #374151; border-radius:8px; font-size:13px; }
  form.controls button { padding:7px 14px; border-radius:8px; border:none;
        background:#22c55e; color:#022c22; font-weight:600; cursor:pointer; font-size:13px; }
  table { width:100%; border-collapse: collapse; font-size:13px; }
  th, td { padding: 7px 10px; border-bottom: 1px solid #1f2937; text-align: left; }
  th { color:#9ca3af; font-weight:600; font-size: 12px; text-transform:uppercase; }
  tr:hover td { background:#0a1325; }
  .actions a { display:inline-block; margin-right: 10px; padding: 6px 12px; border-radius: 8px;
        background: #1d4ed8; color: #fff; text-decoration:none; font-size:12px; }
  .actions a:hover { background: #2563eb; }
  .stat { display:inline-block; margin-right:16px; }
  .stat .v { font-size:18px; font-weight:600; }
  .stat .l { font-size:11px; color:#9ca3af; text-transform:uppercase; }
  .empty { color:#6b7280; padding:14px 0; font-style:italic; }
</style>
<header>
  <div>
    <h2>Wait-time reports</h2>
    <div class="small">Detected wait sessions per camera, with daily and monthly aggregates.</div>
  </div>
  <nav>
    <a href="{{ url_for('cameras') }}">Live cameras</a>
    <span class="small">Logged in as {{ user }}</span>
    <form method="post" action="{{ url_for('logout') }}" style="display:inline">
      <button class="logout" type="submit"
              style="background:none;border:1px solid #374151;color:#e5e7eb;padding:6px 11px;border-radius:999px;font-size:13px;">
        Logout
      </button>
    </form>
  </nav>
</header>
<main>

  <section>
    <h3>Camera</h3>
    <form class="controls" method="get" action="{{ url_for('reports') }}">
      <label>Camera
        <select name="camera">
          {% for slug, name in cameras_index.items() %}
            <option value="{{ slug }}" {% if slug == camera %}selected{% endif %}>{{ name }}</option>
          {% endfor %}
        </select>
      </label>
      <label>Date (sessions list)
        <input type="date" name="date" value="{{ date }}">
      </label>
      <label>From
        <input type="date" name="from_date" value="{{ from_date }}">
      </label>
      <label>To
        <input type="date" name="to_date" value="{{ to_date }}">
      </label>
      <label>Year (monthly)
        <input type="number" name="year" min="2020" max="2100" value="{{ year }}" style="width:90px">
      </label>
      <button type="submit">Apply</button>
    </form>
  </section>

  <section>
    <h3>Sessions on {{ date }} — {{ camera_name }}</h3>
    <div class="actions">
      <a href="{{ url_for('reports_daily_xlsx', camera=camera, date=date) }}">Download daily Excel</a>
      <span class="stat"><span class="l">People</span><br><span class="v">{{ sessions|length }}</span></span>
      <span class="stat"><span class="l">Average wait</span><br>
        <span class="v">{% if avg_today > 0 %}{{ "%.1f"|format(avg_today) }} s{% else %}—{% endif %}</span>
      </span>
    </div>
    {% if sessions %}
    <table>
      <thead><tr><th>#</th><th>Person ID</th><th>First seen</th><th>Last seen</th><th>Wait time</th></tr></thead>
      <tbody>
      {% for s in sessions %}
        <tr>
          <td>{{ loop.index }}</td>
          <td>{{ s.label_id }}</td>
          <td>{{ s.first_seen | timestamp_hms }}</td>
          <td>{{ s.last_seen | timestamp_hms }}</td>
          <td>{{ s.total_wait_s | seconds_hms }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    {% else %}
      <div class="empty">No wait sessions logged for this date yet.</div>
    {% endif %}
  </section>

  <section>
    <h3>Daily averages — {{ from_date }} → {{ to_date }}</h3>
    <div class="actions">
      <a href="{{ url_for('reports_range_xlsx', camera=camera, from_date=from_date, to_date=to_date) }}">Download daily-range Excel</a>
    </div>
    {% if daily %}
    <table>
      <thead><tr><th>Date</th><th>People</th><th>Avg wait</th><th>Min</th><th>Max</th></tr></thead>
      <tbody>
      {% for d in daily %}
        <tr>
          <td>{{ d.date }}</td>
          <td>{{ d.people }}</td>
          <td>{{ d.avg_wait | seconds_hms }}</td>
          <td>{{ d.min_wait | seconds_hms }}</td>
          <td>{{ d.max_wait | seconds_hms }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    <canvas id="dailyChart" height="80" style="margin-top:12px"></canvas>
    {% else %}
      <div class="empty">No data for this range.</div>
    {% endif %}
  </section>

  <section>
    <h3>Monthly averages — {{ year }}</h3>
    <div class="actions">
      <a href="{{ url_for('reports_monthly_xlsx', camera=camera, year=year) }}">Download monthly Excel (with chart)</a>
    </div>
    {% if monthly %}
    <table>
      <thead><tr><th>Month</th><th>People</th><th>Avg wait</th></tr></thead>
      <tbody>
      {% for m in monthly %}
        <tr>
          <td>{{ m.month }}</td>
          <td>{{ m.people }}</td>
          <td>{{ m.avg_wait | seconds_hms }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    <canvas id="monthlyChart" height="80" style="margin-top:12px"></canvas>
    {% else %}
      <div class="empty">No data for this year.</div>
    {% endif %}
  </section>

</main>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
  const dailyData = {
    labels: [{% for d in daily %}"{{ d.date }}"{% if not loop.last %},{% endif %}{% endfor %}],
    avgs:   [{% for d in daily %}{{ "%.2f"|format(d.avg_wait or 0) }}{% if not loop.last %},{% endif %}{% endfor %}],
  };
  if (dailyData.labels.length) {
    new Chart(document.getElementById('dailyChart'), {
      type: 'line',
      data: { labels: dailyData.labels,
              datasets: [{ label: 'Avg wait (s)', data: dailyData.avgs,
                           tension: .25, borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,.2)' }] },
      options: { plugins:{legend:{labels:{color:'#e5e7eb'}}}, scales:{
        x:{ ticks:{color:'#9ca3af'}, grid:{color:'#1f2937'} },
        y:{ ticks:{color:'#9ca3af'}, grid:{color:'#1f2937'} } } }
    });
  }

  const monthlyData = {
    labels: [{% for m in monthly %}"{{ m.month }}"{% if not loop.last %},{% endif %}{% endfor %}],
    avgs:   [{% for m in monthly %}{{ "%.2f"|format(m.avg_wait or 0) }}{% if not loop.last %},{% endif %}{% endfor %}],
  };
  if (monthlyData.labels.length) {
    new Chart(document.getElementById('monthlyChart'), {
      type: 'bar',
      data: { labels: monthlyData.labels,
              datasets: [{ label: 'Avg wait (s)', data: monthlyData.avgs,
                           backgroundColor: '#3b82f6' }] },
      options: { plugins:{legend:{labels:{color:'#e5e7eb'}}}, scales:{
        x:{ ticks:{color:'#9ca3af'}, grid:{color:'#1f2937'} },
        y:{ ticks:{color:'#9ca3af'}, grid:{color:'#1f2937'} } } }
    });
  }
</script>
"""


@app.template_filter("timestamp_hms")
def _filter_timestamp_hms(ts):
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except Exception:
        return "—"


@app.template_filter("seconds_hms")
def _filter_seconds_hms(secs):
    try:
        secs = int(max(0, float(secs or 0)))
    except Exception:
        return "—"
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    # For local debugging only; in production use gunicorn as described in README.
    app.run(host="0.0.0.0", port=8000, debug=True)


