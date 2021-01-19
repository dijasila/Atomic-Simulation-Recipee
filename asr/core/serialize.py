import copy
import ase
import simplejson as json
import typing
import pathlib
import os
from .config import config
from .results import obj_to_id
from .filetype import ExternalFile, ASRFile
from .utils import only_master


class ASRJSONEncoder(json.JSONEncoder):

    def default(self, obj) -> dict:

        if isinstance(obj, ExternalFile):
            path = pathlib.Path(obj)
            newpath = (
                config.root
                / 'external_files'
                / (','.join([path.name, obj.sha256[:12]]))
            )
            directory = newpath.parent
            if not directory.is_dir():
                only_master(os.makedirs)(directory)
            only_master(path.rename)(newpath)
            obj = ASRFile(newpath)

        if hasattr(obj, '__dict__'):
            cls_id = obj_to_id(obj.__class__)
            obj = {'cls_id': cls_id, '__dict__':
                   copy.copy(obj.__dict__)}
            return obj

        if isinstance(obj, set):
            return {
                '__asr_type__': 'set',
                'value': list(obj),
            }
        elif isinstance(obj, tuple):
            return {
                '__asr_type__': 'tuple',
                'value': list(obj),
            }
        return ase.io.jsonio.MyEncoder.default(self, obj)


def json_hook(json_object: dict):
    from asr.core.results import get_object_matching_obj_id
    from ase.io.jsonio import object_hook

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
        elif asr_type == 'tuple':
            return tuple(value)

    return object_hook(json_object)


class JSONSerializer:

    encoder = ASRJSONEncoder(tuple_as_array=False).encode
    decoder = json.JSONDecoder(object_hook=json_hook).decode

    def serialize(self, obj) -> str:
        """Serialize object to JSON."""
        return self.encoder(obj)

    def deserialize(self, serialized: str) -> typing.Any:
        """Deserialize json object."""
        return self.decoder(serialized)
