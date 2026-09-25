import logging
from datetime import date

from flask import Blueprint, abort, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy import or_

from . import audit, ml, scores
from .extensions import db
from .models import Assessment, Patient, utcnow
from .tenancy import get_scoped_or_404, scoped
from .validation import validate

bp = Blueprint("patients", __name__)
log = logging.getLogger("cardiorisk")
PER_PAGE = 25


@bp.route("/patients")
@login_required
def index():
    q = (request.args.get("q") or "").strip()[:100]
    query = scoped(Patient)
    if q:
        like = f"%{q}%"
        query = query.filter(or_(Patient.mrn.ilike(like), Patient.last_name.ilike(like),
                                 Patient.first_name.ilike(like)))
    page = query.order_by(Patient.last_name, Patient.first_name).paginate(
        per_page=PER_PAGE, error_out=False)
    return render_template("patients/index.html", page=page, q=q)


def _patient_form(form, patient_id=None):
    """Validate demographics. Returns (values, errors)."""
    values, errors = {}, {}
    for name, label, limit in (("mrn", "MRN", 64), ("first_name", "First name", 100),
                               ("last_name", "Last name", 100)):
        value = (form.get(name) or "").strip()
        if not value:
            errors[name] = f"{label} is required."
        elif len(value) > limit:
            errors[name] = f"{label} must be at most {limit} characters."
        values[name] = value
    try:
        dob = date.fromisoformat(form.get("date_of_birth") or "")
        if dob > date.today() or dob.year < date.today().year - 120:
            errors["date_of_birth"] = "Enter a valid date of birth."
        values["date_of_birth"] = dob
    except ValueError:
        errors["date_of_birth"] = "Enter the date of birth as YYYY-MM-DD."
    if form.get("sex") not in ("F", "M"):
        errors["sex"] = "Select a sex."
    values["sex"] = form.get("sex")
    if "mrn" not in errors:
        clash = scoped(Patient).filter(Patient.mrn == values["mrn"])
        if patient_id is not None:
            clash = clash.filter(Patient.id != patient_id)
        if db.session.query(clash.exists()).scalar():
            errors["mrn"] = "A patient with this MRN already exists."
    return values, errors


@bp.route("/patients/new", methods=["GET", "POST"])
@login_required
def create():
    if request.method == "GET":
        return render_template("patients/form.html", form={}, errors={}, patient=None)
    values, errors = _patient_form(request.form)
    if errors:
        return render_template("patients/form.html", form=request.form, errors=errors, patient=None), 422
    patient = Patient(org_id=current_user.org_id, created_by_id=current_user.id, **values)
    db.session.add(patient)
    db.session.flush()
    audit.record("patient_created", patient)
    db.session.commit()
    return redirect(url_for("patients.detail", patient_id=patient.id))


@bp.route("/patients/<int:patient_id>/edit", methods=["GET", "POST"])
@login_required
def edit(patient_id):
    patient = get_scoped_or_404(Patient, patient_id)
    if request.method == "GET":
        form = {"mrn": patient.mrn, "first_name": patient.first_name, "last_name": patient.last_name,
                "date_of_birth": patient.date_of_birth.isoformat(), "sex": patient.sex}
        return render_template("patients/form.html", form=form, errors={}, patient=patient)
    values, errors = _patient_form(request.form, patient_id=patient.id)
    if errors:
        return render_template("patients/form.html", form=request.form, errors=errors, patient=patient), 422
    changed = sorted(k for k, v in values.items() if getattr(patient, k) != v)
    for key, value in values.items():
        setattr(patient, key, value)
    audit.record("patient_updated", patient, fields=changed)
    db.session.commit()
    return redirect(url_for("patients.detail", patient_id=patient.id))


@bp.route("/patients/<int:patient_id>")
@login_required
def detail(patient_id):
    patient = get_scoped_or_404(Patient, patient_id)
    assessments = patient.assessments.all()
    audit.record("patient_viewed", patient)
    db.session.commit()
    return render_template("patients/detail.html", patient=patient, assessments=assessments,
                           age=patient.age_on(date.today()))


@bp.route("/patients/<int:patient_id>/assess", methods=["GET", "POST"])
@login_required
def assess(patient_id):
    patient = get_scoped_or_404(Patient, patient_id)
    age = patient.age_on(date.today())
    if request.method == "GET":
        return render_template("assess.html", patient=patient, age=age, form={}, errors={})

    # Age and sex come from the patient record, never from the form.
    submitted = {**request.form.to_dict(), "age": str(age), "sex": "1" if patient.sex == "M" else "0"}
    values, errors, warnings = validate(submitted)
    if errors:
        return render_template("assess.html", patient=patient, age=age,
                               form=request.form, errors=errors), 422
    try:
        result = ml.predict(values)
    except Exception:
        log.exception("prediction failed")  # never log the submitted values (PHI)
        return render_template("assess.html", patient=patient, age=age, form=request.form, errors={},
                               error="The assessment could not be completed. Please try again."), 500
    ptp = scores.esc_2019_ptp(values["age"], values["sex"], values["cp"])

    assessment = Assessment(
        org_id=current_user.org_id, patient_id=patient.id, clinician_id=current_user.id,
        created_at=utcnow(), inputs=values, model_version=result["model_version"],
        probability=result["probability"], risk_band=result["risk"], positive=result["positive"],
        contributions=result["contributions"], warnings=warnings,
        ptp=ptp.percent, ptp_note=ptp.note,
    )
    db.session.add(assessment)
    db.session.flush()
    audit.record("assessment_created", assessment)
    db.session.commit()
    # Post/redirect/get: a browser refresh must not create a duplicate record.
    return redirect(url_for("patients.assessment", assessment_id=assessment.id))


@bp.route("/assessments/<int:assessment_id>")
@login_required
def assessment(assessment_id):
    record = get_scoped_or_404(Assessment, assessment_id)
    if record.patient.org_id != current_user.org_id:  # defence in depth
        abort(404)
    audit.record("assessment_viewed", record)
    db.session.commit()
    return render_template("result.html", a=record, patient=record.patient,
                           discordant=scores.discordant(record.ptp, record.probability),
                           model_metadata=ml.MODEL_METADATA)
