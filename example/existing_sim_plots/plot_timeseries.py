#!/usr/bin/env python

import urllib
import os
import sys
import json
import numpy
import webbrowser
import sqlite3
from cStringIO import StringIO
import twisted
from twisted.web import server, resource
from twisted.internet import reactor

SIM_DB_PATH = '/Users/ebaskerv/uchicago/midway_cobey/ccmproject-storage/2015-10-23-simulations/results_gathered.sqlite'

class SQLiteServer(resource.Resource):
    def __init__(self, db):
        self.db = db
    
    isLeaf = True
    def render_POST(self, request):
        request.setHeader('Access-Control-Allow-Origin', 'null')
        
        request_data = json.loads(request.content.read())
        
        query = request_data['query']
        args = request_data['args']
        
        rows = []
        try:
            if args is None:
                c = db.execute(query)
            else:
                c = db.execute(query, args)
            columns = [x[0] for x in c.description]
            print columns
            for row in c:
                rows.append([npy_buffer_to_list(x) if isinstance(x, buffer) else x for x in row])
        
            return json.dumps({
                'exception' : None,
                'columns' : columns,
                'rows' : rows
            })
        except Exception as e:
            return json.dumps({
                'exception' : str(e),
                'columns' : None,
                'rows' : None
            })

def ndarray_to_npy_buffer(x):
    f = StringIO()
    numpy.save(f, x)
    buf = buffer(f.getvalue())
    f.close()
    return buf

def npy_buffer_to_ndarray(x):
    f = StringIO(x)
    arr = numpy.load(f)
    f.close()
    return arr

if __name__ == '__main__':
    if len(sys.argv) > 1:
        db_filename = os.path.abspath(sys.argv[1])
    else:
        db_filename = SIM_DB_PATH
    
    if not os.path.exists(db_filename):
        print('SQLite database {0} does not exist.'.format(db_filename))
        sys.exit(1)
    
    with sqlite3.connect(db_filename, timeout=60) as db:
        site = server.Site(SQLiteServer(db))
        port_obj = reactor.listenTCP(0, site)
        port_num = port_obj.getHost().port
        
        html_path = urllib.pathname2url(os.path.abspath('plot_timeseries.html'))
        html_args = urllib.urlencode({'port' : port_num})
        html_url = 'file://{0}?{1}'.format(html_path, html_args)
        
        print('Running at URL:')
        print(html_url)
        
        webbrowser.open_new(html_url)
        reactor.run()
