from gevent import pywsgi

if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 6000), app)
    server.serve_forever()