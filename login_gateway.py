from functools import wraps

from flask import (
    Flask,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)


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
# These are the links that will be shown on the cameras page after login.
CAMERAS = [
    {
        "name": "DHA_CAM1 (8182)",
        "url": "http://192.168.88.215:8182/video_ai",
    },
    {
        "name": "DHA_CAM2 (8186)",
        "url": "http://192.168.88.215:8186/video_ai",
    },
    {
        "name": "MM_CAM1 (8187)",
        "url": "http://192.168.88.215:8187/video_ai",
    },
    {
        "name": "MM_CAM2 (8188)",
        "url": "http://192.168.88.215:8188/video_ai",
    },
    {
        "name": "FSD1 (8183)",
        "url": "http://192.168.88.215:8183/video_ai",
    },
    {
        "name": "FSD2 (8184)",
        "url": "http://192.168.88.215:8184/video_ai",
    },
    {
        "name": "FSD3 (8185)",
        "url": "http://192.168.88.215:8185/video_ai",
    },
]


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
  <form method="post" action="{{ url_for('logout') }}">
    <span class="small">Logged in as {{ user }}</span>
    <button class="logout" type="submit">Logout</button>
  </form>
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
        cameras=CAMERAS,
        user=session.get("user"),
    )


if __name__ == "__main__":
    # For local debugging only; in production use gunicorn as described in README.
    app.run(host="0.0.0.0", port=8000, debug=True)


