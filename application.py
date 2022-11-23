from application import application

if __name__ == "__main__":
    application.secret_key = 'super secret key'
    application.config['SESSION_TYPE'] = 'filesystem'
    application.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024
    application.run(debug=True)
    #application.run(host='0,0,0,0',port=5000,debug=True)