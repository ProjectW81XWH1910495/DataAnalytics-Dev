from flask import render_template
from application import application


@application.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@application.errorhandler(504)
def internal_error(error):
    return render_template('504.html'), 504

@application.errorhandler(502)
def internal_error(error):
    return render_template('502.html'), 502

@application.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@application.errorhandler(413)
def internal_error(error):
    return render_template('413.html'), 413