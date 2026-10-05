from flask import Blueprint, request, jsonify
from .models import User
from . import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required, current_user
import logging

bp = Blueprint('auth', __name__)


@bp.route('/api/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')
    if not email or not password:
        return jsonify(success=False, error='missing fields'), 400
    if User.query.filter_by(email=email).first():
        return jsonify(success=False, error='user exists'), 400
    user = User(email=email, name=name)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    login_user(user)
    return jsonify(success=True, id=user.id)


@bp.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email = data.get('email')
    password = data.get('password')
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        login_user(user)
        return jsonify(success=True)
    return jsonify(success=False, error='invalid credentials'), 401


@bp.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify(success=True)


@bp.route('/api/me', methods=['GET'])
def me():
    if not hasattr(current_user, 'is_authenticated') or not current_user.is_authenticated:
        return jsonify(authenticated=False), 401
    return jsonify(authenticated=True, id=current_user.id, email=current_user.email, name=current_user.name)


@bp.route('/api/assessments', methods=['GET'])
@login_required
def list_assessments():
    from .models import Assessment
    items = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.created_at.desc()).limit(100).all()
    out = []
    for it in items:
        out.append({
            'id': it.id,
            'stress_level': it.stress_level,
            'assessment_data': it.assessment_data,
            'created_at': it.created_at.isoformat()
        })
    return jsonify(success=True, items=out)


@bp.route('/api/assessment', methods=['POST'])
@login_required
def create_assessment():
    from .models import Assessment
    data = request.get_json() or {}
    stress_level = data.get('stress_level') or data.get('overall') or data.get('stressLevel')
    assessment_data = data.get('assessment_data') or data.get('metadata') or data.get('data') or data

    if not stress_level:
        return jsonify(success=False, error='missing stress_level'), 400

    entry = Assessment(user_id=current_user.id, stress_level=str(stress_level), assessment_data=assessment_data)
    try:
        db.session.add(entry)
        db.session.commit()
        return jsonify(success=True, id=entry.id)
    except Exception:
        db.session.rollback()
        logging.exception('Failed to save assessment')
        return jsonify(success=False, error='db error'), 500
