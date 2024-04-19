from asr.core import command, ASRResult, prepare_result, read_json
from pathlib import Path
from ase.io import read
from ase import Atoms
import numpy as np
import spglib
import os
from functools import cached_property


class Descriptor:
    def __init__(self, blfolder=None, cell=None):
        self.blfolder = str(Path('.').absolute()) if blfolder is None else blfolder
        
        # If the cell is not provided we get it from either a bilayer or a monolayer structure file
        if cell is None:
            if not Path(f"{self.blfolder}/structure.json").is_file():
                p = Path(self.blfolder).resolve().parents[0]
                self.cell = read(f"{p}/structure.json").cell
            else:
                p = Path(f'{self.blfolder}/structure.json').absolute()
                self.cell = read(str(p)).cell
        else:
            self.cell = cell


    def transformation_matrix(self):
        transformation_matrix_inplane = read_json(f"{self.blfolder}/transformdata.json")["rotation"]
        return transformation_matrix_inplane[0:2,0:2]


    def convert_to_cartesian(self):
        a1, a2, _ = self.cell
        a1 = a1[:2]
        a2 = a2[:2]
        b1 = np.array([a2[1], -a2[0]])
        b2 = np.array([a1[1], -a1[0]])
        b1 /= b1 @ a1
        b2 /= b2 @ a2
        assert np.allclose(b1 @ a2, 0.0)
        assert np.allclose(b2 @ a1, 0.0)

        N = np.array([a1, a2]).T
        B = np.array([b1, b2])
        assert np.allclose(B @ N, np.eye(2))
        assert np.allclose(N @ B, np.eye(2)), N @ B

        return N @ self.transformation_matrix() @ B


    def get_rotations_n(self):
        ''' Symmetry transformations are translated to rotations angles
            Decompose transformation matrix into form b^mc^n
            where b is a rotation matrix and c is a reflection
            Since we remove det=-1 cases, bm is always  pure rotation.
            It should have entries [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
            where theta = 2pi / n for some n '''
        bm = self.convert_to_cartesian()

        if np.allclose(bm, np.eye(2)):
            a1 = 0.0
            a2 = 0.0
        else:
            a1 = np.arctan2(-bm[0, 1], bm[0, 0])
            a2 = np.arctan2(bm[1, 0], bm[1, 1])

        if np.allclose(a1, 0.0):
            return ''
        else:
            n = 2 * np.pi / a1
            return str(abs(round(n)))
        

    def build_notation(self, top_flipped, bot_flipped, tx, ty):
        overlineA = "<span style='text-decoration: overline;'>A</span>"
        overlineB = "<span style='text-decoration: overline;'>B</span>"

        html_descriptor  = 'A' if not bot_flipped else overlineA
        latex_descriptor = 'A' if not bot_flipped else "\overline{A}"
        
        # rotation around z axis
        rotation_n = self.get_rotations_n()

        # top layer A or B or Bbar
        if abs(tx>0.01) or abs(ty>0.01) or top_flipped or rotation_n!='':
            html_descriptor  += 'B' if not top_flipped else overlineB
            latex_descriptor += 'B' if not top_flipped else "\overline{B}"
        else:
            html_descriptor  += 'A'
            latex_descriptor += 'A'

        # top layer rotation
        html_descriptor += f'<sub>{rotation_n}</sub>' if rotation_n!="" else ""
        latex_descriptor += f'_{rotation_n}' if rotation_n!="" else ""

        # top layer displacement
        if abs(tx>0.01) or abs(ty>0.01):
            html_descriptor += f' ({tx:0.2f}, {ty:0.2f})'
            latex_descriptor += f' ({tx:0.2f}, {ty:0.2f})'

        latex_descriptor = "$"+latex_descriptor+"$"
        return html_descriptor, latex_descriptor


    def get_descriptor(self):
        # The existance of these files is a requirement of the recipe so it will be checked there
        # but I raise these errors in case the class was used elsewhere
        if not os.path.isfile(f'{self.blfolder}/transformdata.json'):
            raise FileNotFoundError('transformdata.json file not found')
        if not os.path.isfile(f'{self.blfolder}/translation.json'):
            raise FileNotFoundError('translation.json file not found')

        bot_flipped = read_json(f'{self.blfolder}/transformdata.json')['Bottom_layer_Flipped']
        top_flipped = read_json(f'{self.blfolder}/transformdata.json')['Top_layer_Flipped']

        # We want the translations in fractional units
        translation_inplane = read_json(f"{self.blfolder}/translation.json")["translation_vector"]
        transl = self.cell.scaled_positions(np.array([translation_inplane[0], translation_inplane[1], 0.0]))
        tx = transl[0] if abs(transl[0])>0.01 else 0.00
        ty = transl[1] if abs(transl[1])>0.01 else 0.00

        html_descriptor, latex_descriptor = self.build_notation(top_flipped, bot_flipped, tx, ty)
        return html_descriptor, latex_descriptor


def set_number_of_layers():
    from asr.setinfo import main as setinfo
    setinfo([('numberoflayers', 2)])


def set_monolayer_uid():
    from asr.setinfo import main as setinfo
    ml_uid = Path("..").resolve().name
    setinfo([('monolayer_uid', ml_uid)])
    return ml_uid


@prepare_result
class BilayerDescriptor(ASRResult):
    monolayer_uid: str
    flipped_monolayer_transformation: np.ndarray
    blfolder_name: str
    toplayer_transformation: np.ndarray
    toplayer_translation: np.ndarray
    html_descriptor: str
    latex_descriptor: str  

    key_descriptions = dict(monolayer_uid='UID of source monolayer',
                            flipped_monolayer_transformation='Transformation to get flipped layer',
                            blfolder_name='Name of the bilayer folder in the tree',
                            toplayer_transformation=''.join(['Point group operator',
                                                   ' of top layer relative',
                                                   ' to bottom layer.',
                                                   ' The matrix operators',
                                                   ' on the lattice vectors']),
                            toplayer_translation=''.join(['Translation of top layer',
                                                 ' relative to bottom layer',
                                                 ' in scaled coordinates']),
                            html_descriptor='A full descriptor of a stacking in html notation',
                            latex_descriptor='A full descriptor of a stacking in tex notation')


@command(module='asr.bilayerdescriptor',
         requires=['transformdata.json',
                   'translation.json'],
         returns=BilayerDescriptor)


def main() -> BilayerDescriptor:
    blfolder = str(Path('.').absolute())
    blname = blfolder.split('/')[-1]

    """First we need to establish the monolayer information"""
    monolayer_uid = set_monolayer_uid()
 
    bot_flipped = read_json('transformdata.json')['Bottom_layer_Flipped']
    top_flipped = read_json('transformdata.json')['Top_layer_Flipped']

    if top_flipped or bot_flipped:
        flipped_ml_transform=read_json('../flip_transformation.json')['transformation']
    else:
        flipped_ml_transform=np.eye(3)

    """ Next we read the bilayer information"""
    transform = read_json('transformdata.json')
    rotation = transform['rotation']
 
    translation = read_json('translation.json')['translation_vector']
    t_c = transform['translation'][:2] + translation[:2]

    """ Now we get the full descriptor"""
    descriptor = Descriptor(blfolder=blfolder)
    html_descriptor, latex_descriptor = descriptor.get_descriptor()
    
    set_number_of_layers()

    return BilayerDescriptor.fromdata(monolayer_uid=monolayer_uid,
                                      flipped_monolayer_transformation=flipped_ml_transform,
                                      blfolder_name = blname,
                                      toplayer_transformation=rotation,
                                      toplayer_translation=t_c,
                                      html_descriptor=html_descriptor,
                                      latex_descriptor=latex_descriptor)
