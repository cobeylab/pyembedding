import sys
import json
import numpy
import cStringIO
from collections import OrderedDict

class JSONObject(object):
    '''Class to improve syntax when working with JSON-loaded objects, and automatically convert lists from/to
    numpy arrays.

    >>> x = JSONObject([('foo', 5), ('bar', 7)])
    >>> x.foo == 5
    True
    >>> x.bar == 7
    True
    '''

    def __init__(self, name_value_pairs=None, **kwargs):
        '''Initialize object from attribute name-value pairs.

        :param name_value_pairs: Attribute name-value pairs.
        '''
        self.odict = OrderedDict()

        if name_value_pairs is not None:
            if isinstance(name_value_pairs, dict):
                #sys.stderr.write('{0}\n'.format(name_value_pairs))
                name_value_pairs = name_value_pairs.items()

            for name, value in name_value_pairs:
                setattr(self, name, value)

        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setattr__(self, name, value):
        if name == 'odict':
            return super(JSONObject, self).__setattr__(name, value)

        '''Assigns attribute to object, converting to numpy array if appropriate.

        :param name: Name of attribute.
        :param value: Value of attribute.

        >>> x = JSONObject()
        >>> x.assign_attribute('foo', 5)
        >>> x.foo == 5
        True
        >>> x.assign_attribute('bar', 7)
        >>> x.bar == 7
        True
        >>> x.assign_attribute('baz', [1,2,3])
        >>> len(x.baz.shape)
        1
        >>> x.baz.shape[0]
        3
        >>> x.assign_attribute('barf', [[1,2], [2,3,4]])
        >>> isinstance(x.barf, list)
        True
        '''

        if isinstance(value, list):
            try:
                value = numpy.array(value)
            except:
                pass
        elif isinstance(value, dict):
            value = JSONObject(value)

        self.odict[name] = value

    def __getattr__(self, item):
        if item == 'odict':
            return super(JSONObject, self).__getattr__(self, item)
        return self.odict[item]

    def __len__(self):
        return len(self.odict)

    def keys(self):
        return self.odict.keys()

    def iterkeys(self):
        return self.odict.iterkeys()

    def values(self):
        return self.odict.values()

    def itervalues(self):
        return self.odict.itervalues()

    def items(self):
        return self.odict.items()

    def iteritems(self):
        return self.odict.iteritems()

    def update_from_file(self, f_or_s):
        '''

        :param f:
        :return:

        >>> json_obj = JSONObject([('foo', 5), ('bar', 7), ('baz', [1,2,3,4])])
        >>> json_file = cStringIO.StringIO('{"bar" : 8}')
        >>> json_obj.update_from_file(json_file)
        >>> json_obj.foo
        5
        >>> json_obj.bar
        8
        >>> json_obj.baz.tolist()
        [1, 2, 3, 4]
        '''
        json_obj = load_from_file(f_or_s)
        self.odict.update(json_obj.odict)

    def update_from_string(self, s):
        '''

        :param f:
        :return:

        >>> json_obj = JSONObject([('foo', 5), ('bar', 7), ('baz', [1,2,3,4])])
        >>> json_obj.update_from_string('{"bar" : 8}')
        >>> json_obj.foo
        5
        >>> json_obj.bar
        8
        >>> json_obj.baz.tolist()
        [1, 2, 3, 4]
        '''
        json_obj = load_from_string(s)
        self.odict.update(json_obj.odict)

    def dump_to_file(self, f_or_s, indent=None):
        '''
        :return:

        >>> json_obj = JSONObject([('foo', 5), ('bar', 7), ('baz', [1,2,3,4])])
        >>> json_out = cStringIO.StringIO()
        >>> json_obj.dump_to_file(json_out)
        >>> json_obj_check = json.loads(json_out.getvalue())
        >>> json_obj_check['foo']
        5
        >>> json_obj_check['bar']
        7
        >>> json_obj_check['baz']
        [1, 2, 3, 4]
        '''
        if isinstance(f_or_s, basestring):
            f = open(f_or_s, 'w')
            opened_file = True
        else:
            f = f_or_s
            opened_file = False
        json.dump(self, f, cls=JSONObjectEncoder, indent=indent)
        f.write('\n')
        if opened_file:
            f.close()

    def dump_to_string(self, indent=None):
        '''
        :return:

        >>> json_obj = JSONObject([('foo', 5), ('bar', 7), ('baz', [1,2,3,4])])
        >>> json_out = cStringIO.StringIO()
        >>> json_obj_check = json.loads(json_obj.dump_to_string())
        >>> json_obj_check['foo']
        5
        >>> json_obj_check['bar']
        7
        >>> json_obj_check['baz']
        [1, 2, 3, 4]
        '''
        return json.dumps(self, cls=JSONObjectEncoder, indent=indent)


class JSONObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, JSONObject):
            return obj.odict
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_from_file(f_or_s):
    '''Load JSONObject object from JSON file
    >>> json_file = cStringIO.StringIO('{"foo" : 5, "bar" : 7}')
    >>> json_obj = load_from_file(json_file)
    >>> json_obj.foo
    5
    >>> json_obj.bar
    7
    '''
    if isinstance(f_or_s, basestring):
        f = open(f_or_s)
        opened_file = True
    else:
        f = f_or_s
        opened_file = False
    obj = json.load(f, object_pairs_hook=JSONObject)
    if opened_file:
        f.close()
    return obj

def load_from_string(s):
    '''Load JSONObject object from JSON file
    >>> json_str = '{"foo" : 5, "bar" : [7, 8, 9]}'
    >>> json_obj = load_from_string(json_str)
    >>> json_obj.foo
    5
    >>> json_obj.bar.tolist()
    [7, 8, 9]
    '''
    return json.loads(s, object_pairs_hook=JSONObject)

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
