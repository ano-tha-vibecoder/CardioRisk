"""Display stored UTC timestamps in the clinic's own time zone."""
from datetime import datetime, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo, available_timezones

from flask_login import current_user


@lru_cache(maxsize=1)
def choices() -> list[str]:
    """Zones offered in clinic settings: Africa plus UTC."""
    return ["UTC"] + sorted(z for z in available_timezones() if z.startswith("Africa/"))


def clinic_zone() -> ZoneInfo:
    if current_user and current_user.is_authenticated:
        return ZoneInfo(current_user.organization.timezone)
    return ZoneInfo("UTC")


def localtime(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc).astimezone(clinic_zone())


def init_app(app) -> None:
    app.add_template_filter(localtime, "localtime")

    @app.context_processor
    def zone_name():
        return {"clinic_tz": str(clinic_zone())}
