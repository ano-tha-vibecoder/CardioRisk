from collections import Counter
from datetime import timedelta

from flask import Blueprint, redirect, render_template, url_for
from flask_login import current_user, login_required
from sqlalchemy.orm import joinedload

from . import ml
from .extensions import db
from .models import Assessment, Patient, utcnow
from .tenancy import scoped

bp = Blueprint("main", __name__)

RISK_ORDER = ("Low", "Moderate", "High")
WEEKS = 12


@bp.route("/")
def landing():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    return render_template("landing.html")


@bp.route("/healthz")
def healthz():
    db.session.execute(db.text("SELECT 1"))
    return {"status": "ok", "model_version": ml.MODEL_METADATA["model_version"]}


def weekly_counts(timestamps, now, weeks=WEEKS):
    """Bucket timestamps into ``weeks`` 7-day bins ending at ``now`` (oldest first)."""
    counts = [0] * weeks
    for ts in timestamps:
        age_days = (now - ts).days
        index = weeks - 1 - age_days // 7
        if 0 <= index < weeks:
            counts[index] += 1
    starts = [(now - timedelta(days=7 * (weeks - i))).date() for i in range(weeks)]
    return list(zip(starts, counts))


CHART_W, CHART_H, CHART_PAD_TOP = 600, 140, 8


def bar_chart(weekly):
    """SVG geometry for the weekly volume chart: bars rounded only at the data end."""
    peak = max((c for _, c in weekly), default=0) or 1
    slot = CHART_W / len(weekly)
    width = slot - 6  # 6px gap between bars
    bars = []
    for i, (start, count) in enumerate(weekly):
        x = i * slot + 3
        height = (CHART_H - CHART_PAD_TOP) * count / peak
        y = CHART_H - height
        r = min(4.0, width / 2, height)
        path = (f"M{x:.1f},{CHART_H} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
                f"H{x + width - r:.1f} Q{x + width:.1f},{y:.1f} {x + width:.1f},{y + r:.1f} V{CHART_H} Z")
        bars.append({"start": start, "count": count, "path": path if count else None,
                     "hit_x": i * slot, "hit_w": slot, "label_x": x + width / 2, "label_y": y - 4})
    return {"bars": bars, "peak": peak, "w": CHART_W, "h": CHART_H}


@bp.route("/dashboard")
@login_required
def dashboard():
    now = utcnow()
    last_30 = now - timedelta(days=30)
    recent_window = scoped(Assessment).filter(Assessment.created_at >= now - timedelta(days=7 * WEEKS))

    timestamps, bands = [], Counter()
    for created_at, band in recent_window.with_entities(Assessment.created_at, Assessment.risk_band):
        timestamps.append(created_at)
        if created_at >= last_30:
            bands[band] += 1

    total_30 = sum(bands.values())
    risk = [{"band": b, "count": bands[b],
             "share": bands[b] / total_30 if total_30 else 0.0} for b in RISK_ORDER]
    weekly = weekly_counts(timestamps, now)
    stats = {
        "patients": scoped(Patient).count(),
        "assessments_30": total_30,
        "high_30": bands["High"],
        "mine_30": scoped(Assessment).filter(Assessment.clinician_id == current_user.id,
                                             Assessment.created_at >= last_30).count(),
    }
    recent = (scoped(Assessment).options(joinedload(Assessment.patient), joinedload(Assessment.clinician))
              .order_by(Assessment.created_at.desc()).limit(10).all())
    return render_template("dashboard.html", stats=stats, risk=risk, chart=bar_chart(weekly),
                           recent=recent, weeks=WEEKS)
