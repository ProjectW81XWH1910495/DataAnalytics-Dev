from application import application

if __name__ == "__main__":
    application.secret_key = 'super secret key'
    application.config['SESSION_TYPE'] = 'filesystem'
    application.run(debug=True)