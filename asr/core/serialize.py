import copy
import datetime
import pathlib
import typing

import numpy as np
import simplejson as json
from htwutil.storage import JSONCodec
from ase.io.jsonio import (object_hook as ase_object_hook,
                           default as ase_default)

from asr.core.results import get_object_matching_obj_id

from .results import obj_to_id



class ASRJSONCodec(JSONCodec):
    def encode(self, obj):
        from asr.core.serialize import asr_default
        return asr_default(obj)

    def decode(self, dct):
        from asr.core.serialize import json_hook
        return json_hook(dct)


def asr_default(obj):
    try:
        return object_to_dict(obj)
    except ValueError:
        return ase_default(obj)


def object_to_dict(obj) -> dict:
    if hasattr(obj, '__dict__'):
        cls_id = obj_to_id(obj.__class__)
        obj = {
            'cls_id': cls_id, '__dict__':
            copy.copy(obj.__dict__)
        }
        return obj

    if isinstance(obj, set):
        return {
            '__asr_type__': 'set',
            'value': list(obj),
        }
    elif isinstance(obj, np.ndarray):
        # Out tuple serialization causes trouble with
        # ASE's numpy serialization so here we implement our
        # own to avoid conflict with ASE.
        flatobj = obj.ravel()
        if np.iscomplexobj(obj):
            flatobj.dtype = obj.real.dtype
        return {'__ndarray__': [list(obj.shape),
                                obj.dtype.name,
                                flatobj.tolist()]}
    elif isinstance(obj, datetime.datetime):
        return {
            '__asr_type__': 'datetime.datetime',
            'value': obj.isoformat(),
        }
    elif isinstance(obj, tuple):
        return {
            '__asr_type__': 'tuple',
            'value': list(obj),
        }
    elif isinstance(obj, pathlib.Path):
        return {
            '__asr_type__': 'pathlib.Path',
            'value': str(obj),
        }
    raise ValueError


def json_hook(json_object: dict):
    try:
        return dict_to_object(json_object)
    except ValueError:
        pass
    return ase_object_hook(json_object)


def dict_to_object(json_object) -> typing.Any:
    if 'cls_id' in json_object:
        assert '__dict__' in json_object
        cls = get_object_matching_obj_id(json_object['cls_id'])
        obj = cls.__new__(cls)
        obj.__dict__.update(json_object['__dict__'])
        return obj

    asr_type = json_object.get('__asr_type__')

    if asr_type is not None:
        value = json_object['value']
        if asr_type == 'set':
            return set(value)
        if asr_type == 'datetime.datetime':
            return datetime.datetime.fromisoformat(value)
        elif asr_type == 'tuple':
            return tuple(value)
        elif asr_type == 'pathlib.Path':
            return pathlib.Path(value)
    raise ValueError


class JSONSerializer:
    def serialize(self, obj) -> str:
        """Serialize object to JSON."""
        return json.dumps(obj, tuple_as_array=False, default=asr_default)

    def deserialize(self, serialized: str) -> typing.Any:
        """Deserialize json object."""
        return json.loads(serialized, object_hook=json_hook)
