from asr.core import command, option, ASRResult, prepare_result
from typing import List, Dict, Tuple
from ase import Atoms
import numpy as np
import spglib
from ase.io import read
import os
from ase.geometry import wrap_positions
from ase.geometry import get_distances
from asr.core import write_json, read_json
from asr.database.rmsd import get_rmsd
from functools import cached_property
from asr.workflow.bilayerutils import set_info


class StackingError(ValueError):
    """ only 2D materials can be stacked. Rasie error if 0D or 3D """
    pass


class LatticeSymmetries:
    def __init__(self,
                 mlatoms: Atoms,
                 spglib_tol: float = 0.1,
                 remove_unphysical: bool = True):
        """
        Class for computing symmetry operations of a monolayer and its bravias lattice.

        Args:
            mlatoms (Atoms): monolayer atoms
            spglib_tol (float): tolerance for symmetries of monolayer & bravais lattice
            remove_unphysical (bool): True/False. Remove lattice symmetries with det=-1
        """
        self.spglib_tol = spglib_tol
        self.remove_unphysical = remove_unphysical
        self.mlatoms = mlatoms

    @cached_property
    def ml_sym(self) -> Dict[str, np.ndarray]:
        """ Symmetries of the monolayer """
        return spglib.get_symmetry(self.mlatoms, symprec=self.spglib_tol)

    @cached_property
    def bravais_sym(self) -> Dict[str, List]:
        """
        Symmetries of the bravais lattice.
        If asked to remove unphysical symmetries, remove det=-1

        Note: We want to be more strict on the Bravais lattice symmetries to avoid
            generating too many transformations. So we don't use the self.spglib_tol
        """
        pure_cell = Atoms('C', positions=[[0, 0, 0]])
        pure_cell.set_cell(self.mlatoms.get_cell())

        cell_sym = spglib.get_symmetry(pure_cell, symprec=0.01)

        if not self.remove_unphysical:
            return cell_sym

        else:
            all_syms = list(zip(cell_sym['rotations'], cell_sym['translations']))

            pure_rotations = [s for s, t in all_syms if np.linalg.det(s) != -1]
            pure_translations = [t for s, t in all_syms if np.linalg.det(s) != -1]
            return {'rotations': pure_rotations, 'translations': pure_translations}

    @cached_property
    def bravais_not_monolayer_sym(self) -> Dict[str, List]:
        """
        Symmetries of the bravais lattice excluding symmetries of the monolayer.
        We keep the identity transformation even though it is also a monolayer symmetry
        """
        pure_rots = self.bravais_sym['rotations']
        pure_trans = self.bravais_sym['translations']

        ml_rots = self.ml_sym['rotations']

        # Adding identity transformation to create AA stacking first
        rotations = [np.identity(3, dtype=int)]
        translations = [np.zeros(3, dtype=int)]

        for iarr, arr in enumerate(pure_rots):
            if not any([np.array_equal(arr, arr2) for arr2 in ml_rots]):
                rotations.append(arr)
                translations.append(pure_trans[iarr])

        return {'rotations': rotations, 'translations': translations}

    @cached_property
    def bravais_not_monolayer_not_flip_sym(self) -> Dict[str, List]:
        """ Remove flipping transformations from bravais_not_monolayer_sym"""
        res = self.bravais_not_monolayer_sym
        rotations_, translations_ = res['rotations'], res['translations']

        rots = [r for ii, r in enumerate(rotations_) if r[2, 2] != -1]
        trans = [translations_[ii] for ii, r in enumerate(rotations_) if r[2, 2] != -1]

        return {'rotations': rots, 'translations': trans}

    @cached_property
    def ml_needs_flip(self) -> bool:
        """
        Check if monolayer changes with flipping.

        Returns True only when there is a transformation that creates a new structure
        while maintaining the same unit cell by rotating around the first unit vector.

        Logic:
            Check if any bravais symmetry transformation flips the structure by rotating
            around the first axis:
            - Yes: If the monolayer changes with flipping, i.e. none of the monolayer
                   symmetry transformations flip the monolayer, we need to consider the
                   flipped monolayer.
            - No: We won't consider flipping the monolayer
        """

        for mat in self.bravais_sym['rotations']:
            inplane_block = mat[0:2, 0:2]

            # Check if the transformation is physical and flips the structure.
            if np.linalg.det(inplane_block) == -1 and mat[2, 2] == -1:

                # Check if the structure is flipped by rotating around the first axis.
                if mat[0, 0] == 1 and mat[1, 0] == 0:

                    # Here we know the bravais symmetry wants to flip the structure.
                    # Check if flipping actually change the monolayer.
                    return all([s[2, 2] == 1 for s in self.ml_sym['rotations']])

        else:
            return False

    @cached_property
    def cell_type(self) -> str:
        """
        Return the cell type of the monolayer strucutre.
        The use case is when generating special translations for bilayers.
        """
        cell = self.mlatoms.get_cell()
        cell[:, 2] = 0.0
        cell[2, :] = 0.0
        assert cell.rank == 2, cell

        lat = cell.get_bravais_lattice()
        name = lat.name

        if name == 'HEX2D':
            lattice_type = 'hexagonal'
        elif name == 'RECT':
            lattice_type = 'rectangular'
        elif name == 'SQR':
            lattice_type = 'square'
        elif name == 'CRECT':
            lattice_type = 'centered'
        else:
            lattice_type = 'oblique'

        return lattice_type


class InplaneTranslations:
    def __init__(self,
                 mlatoms: Atoms,
                 cell_type: str,
                 flatten_tol: float = 0.2,
                 hexagonal_thirds: bool = False,
                 bridge_points: bool = False):
        """
        Return the in-plane translations of the top with respect to the bottom layer.

        Args:
            mlatoms (Atoms): monolayer atoms
            cell_type: type of cell for cell specific displacements
            flatten_tol (float): tolerance used to find flattened atoms positions
            hexagonal_thirds (bool): do we want 1/3 and 2/3 displacements
            bridge_points (bool): Consider displacements in the middle of 2 atoms.
        """
        self.mlatoms = mlatoms
        self.cell_type = cell_type
        self.flatten_tol = flatten_tol
        self.hexagonal_thirds = hexagonal_thirds
        self.bridge_points = bridge_points

    def flatten(self, atoms: Atoms = None) -> List[np.ndarray]:
        """ Returns a list of in-plane positions of all atoms"""
        if atoms is None:
            atoms = self.mlatoms

        flats = []
        for atom in atoms:
            pos = atom.position[:2]
            if any([np.allclose(x, pos, self.flatten_tol) for x in flats]):
                continue
            else:
                flats.append(pos)
        return flats

    def cell_specific_stacks(self, atoms: Atoms = None) -> List:
        """
        Generate inplane displacements based on cell type
          (1) Middle of the cell vectors and middle of the cell
          (2) Middle of the hexagon in hexagonal lattices like hBN
          (3) The 1/3, 2/3 displacement vectors (for hexagonal cells)

        Note: Option (3) is a more general form that also generates middle of the cell
            as(2), but generates many more structures to study. We ignore it by default

        Note: Despite the possibility of this function to generate many cell specific
            displacements, most of them are by default deactivated because they rarely
            generate a bilayer that will not be created otherwise (with slidestability)
        """
        if atoms is None:
            atoms = self.mlatoms

        final_transforms = []
        positions = atoms.get_positions()
        a, b, c = atoms.cell.lengths()

        def append_helper(x: float, y: float, atoms: Atoms) -> None:
            """
            Brings all the atomsin the structure to specific (x, y) in the cell.
            Example: Generate translations to each atom to the middle of the cell.
            """
            # The for loop is to bring all the atoms to x, y in the cell
            for atom in atoms:
                x_ = x - atom.position[0]
                y_ = y - atom.position[1]
                final_transforms.append(np.array([x_, y_]))

        # Special positions: Middle of the cell vectors and middle of the cell.
        if self.cell_type in ['oblique', 'rectangular', 'square',
                              'centered', 'hexagonal']:
            vec1 = atoms.cell[0]
            vec2 = atoms.cell[1]

            new_vec = vec1 / 2
            append_helper(new_vec[0], new_vec[1], atoms)

            new_vec = vec2 / 2
            append_helper(new_vec[0], new_vec[1], atoms)

            new_vec = (vec1 + vec2) / 2
            append_helper(new_vec[0], new_vec[1], atoms)

        # Special positions: middle of the hexagon in hexagonal lattices like hBN.
        # The logic is that in a hegxagonal cell there are 2 atoms that if you extend
        # along the vector connecting them, you get to the middle of the cell.
        if self.cell_type == 'hexagonal':
            if len(positions) != 1:

                # Avoid cases where the first 2 atoms are on top of each other
                for pos in positions:
                    if np.linalg.norm(positions[0, 0:2] - pos[0:2]) > 0.1:
                        second_atom = pos
                        break

                x = second_atom[0]
                y = second_atom[1]
                append_helper(x, y, atoms)

                x = 2 * second_atom[0]
                y = 2 * second_atom[1]
                append_helper(x, y, atoms)

        # Special positions: Create the 1/3, 2/3 displacement vectors.
        # The purpose was to capture middle of hexagonal cell.
        # Which one points to the middle, depends on the choice of cell vectors
        if self.hexagonal_thirds and self.cell_type in ['hexagonal']:
            vec1 = atoms.cell[0]
            vec2 = atoms.cell[1]

            new_vec = 1 * vec1 / 3 + 2 * vec2 / 3
            append_helper(new_vec[0], new_vec[1], atoms)

            new_vec = 2 * vec1 / 3 + 1 * vec2 / 3
            append_helper(new_vec[0], new_vec[1], atoms)

            new_vec = 1 * vec1 / 3 + 1 * vec2 / 3
            append_helper(new_vec[0], new_vec[1], atoms)

            new_vec = 2 * vec1 / 3 + 2 * vec2 / 3
            append_helper(new_vec[0], new_vec[1], atoms)

        return final_transforms

    @staticmethod
    def vector_in_list_pbc(cell, vec1, veclist,
                           min_distance: float = 0.3) -> bool:
        """
        Check if a vector existis in a list of vectors inside the cell.
        Considers periodic boundary conditions to find the smallest distance.

        Note: If the distance is less than min_distance, don't consider it a new vector
          - To avoid generating too many structures we don't generate similar ones.
          - We assume that the minimums of PES are not that close to each other.
            If the one we removed is the minimum we get to it by DynStab workflow.
        """
        dist = []
        for vec2 in veclist:
            d1, d2 = get_distances([vec1, vec2],
                                   cell=cell,
                                   pbc=[True, True, False])

            # max because we don't want the distance of one atom with itself (0)
            dist.append(np.max(d2.flatten()))

        # If the distance is less than a value we
        return any([d < min_distance for d in dist])

    def pairwise_displacements(self, atoms2: Atoms, atoms1=None):
        """
        Displacement vectors to apply to atoms2 with respect to atoms1

        Args:
            atoms1: Bottom layer Atoms object
            atoms2: Top layer Atoms object that will be displaced with these vectors
                to generate a bilayer. It can be rotated form of atoms1.

        Note: By default the bridge displacements are not considered because we already
            consider displacements that put atoms on top of each other and we don't
            expect local mins both when atoms are on top and in the middle. So we check
            the first and let the other be captured by slide stability.
        """

        def wrap_add_to_list(new_vec):
            """ Brings the vector inside the cell and adds to displacements if new."""
            new_vec = [new_vec[0], new_vec[1], 0]
            new_vec = wrap_positions([new_vec], atoms1.cell,
                                     pbc=True,
                                     center=(0.5, 0.5, 0.5),
                                     pretty_translation=False,
                                     eps=1e-07)[0]

            if not self.vector_in_list_pbc(cell=atoms1.cell,
                                           vec1=new_vec,
                                           veclist=displacements):
                displacements.append(new_vec)

        if atoms1 is None:
            atoms1 = self.mlatoms.copy()

        # Start the list with AA stacking
        displacements = [[0, 0, 0]]

        # Put each atom of top layer on one atom in the bottom layer
        bot_flatten = self.flatten(atoms1)
        top_flatten = self.flatten(atoms2)

        for pos1 in bot_flatten:
            for pos2 in top_flatten:
                move = pos1 - pos2
                wrap_add_to_list(new_vec=move)

        # Cell specific displacements should be applied on each atom at the origin
        for t in self.cell_specific_stacks(atoms=atoms2.copy()):
            wrap_add_to_list(new_vec=t)

        # Displacement vectors in the middle of 2 atoms
        # Add these after previous steps to give it lower priority and remove these
        # rather than previous ones in case of similar displacements.
        if self.bridge_points:
            for pos1 in bot_flatten:
                for pos2 in top_flatten:
                    move = (pos1 - pos2) / 2
                    wrap_add_to_list(new_vec=move)

        return displacements


class BilayerMatcher:
    def __init__(self,
                 atoms_list: List[Atoms],
                 auxs_list: List,
                 transform_layer,
                 bravais_symmetries,
                 vacuum,
                 matcher_tol: float = 0.3,
                 use_rmsd: bool = False):

        self.use_rmsd = use_rmsd
        self.matcher_tol = matcher_tol
        self.atoms_list = atoms_list
        self.auxs_list = auxs_list
        self.vacuum = vacuum
        self.transform_layer = transform_layer
        self.bravais_symmetries = bravais_symmetries

    @classmethod
    def atoms_distance_pbc(cls, cell, pos_1, pos_2):
        """ Return the shortest distance between 2 points in a cell considering pbc"""
        from ase.geometry import get_distances

        d1, d2 = get_distances([pos_1, pos_2],
                               cell=cell,
                               pbc=[True, True, False])

        return np.max(d2.flatten())

    @classmethod
    def atoms_equal(cls,
                    atoms1: Atoms, atoms2: Atoms, matcher_tol: float,
                    vacuum: float = 15, use_rmsd: bool = False) -> bool:
        """
        Check if there is exactly one atom in atoms2 that is at same position
        and has same atomic number as an atom in atoms1

        Note: After transformation, it might be that the role of one atoms is
            played by another atom. For example when flipping the whole structure.
            To capture that effect, we consider the first atom of atoms1 as reference,
            and we bring atoms in atoms2 with the same atomic number to the origin and
            compare the structures. 

        Note: RMSD is not used in this project because it returns None often.
            But there is the option to activate it for testing and benchmark.

        Note: the vacuum in this function is not important any number can be chosen.
        """
        if use_rmsd:
            return (get_rmsd(atoms1.copy(), atoms2.copy()) or 1) < matcher_tol

        numbers1, numbers2 = atoms1.get_atomic_numbers(), atoms2.get_atomic_numbers()

        ref_atomic_number = numbers1[0]
        ref_pos_atoms1 = atoms1[0].position
        same_atomic_number = list(np.where(np.array(numbers2) == ref_atomic_number)[0])

        # We bring atoms with same atomic number to the origin to compare structures
        for index in same_atomic_number:
            atoms2_temp = atoms2.copy()
            temp_ref_pos = atoms2[index].position
            displacement = temp_ref_pos - ref_pos_atoms1
            atoms2_new_pos = [atom.position - displacement for atom in atoms2]
            atoms2_temp.positions = np.array(atoms2_new_pos)
            atoms2_temp.wrap(pbc=[1, 1, 1])
            atoms2_temp = BuildBilayer.adjust_vacuum(vacuum=vacuum,
                                                     layer=atoms2_temp.copy())

            poss1, poss2 = atoms1.get_positions(), atoms2_temp.get_positions()

            # comparing structures atom by atom checking positins and atomic numbers
            matched_atoms = []
            for (pos1, n1) in zip(poss1, numbers1):
                dist = []
                for (pos2, n2) in zip(poss2, numbers2):
                    if n1 != n2:
                        continue

                    dist.append(cls.atoms_distance_pbc(atoms1.cell, pos1, pos2))

                if len(dist) != 0 and min(dist) < matcher_tol:
                    matched_atoms.append(True)

            if len(matched_atoms) == len(atoms1):
                return True

        return False

    def bilayers_similarity_family(self):
        """
        For each bilayer we generate a family with members that are rotated/transformed
        with respect to it

        Returns:
            A list which contains a dictionary for each bilayer in the atoms_list.
            Each dictionary contains all the bravais transformations of the bilayer.
        """
        mats = self.atoms_list
        auxs = self.auxs_list

        family = []
        for ii, (mat, aux) in enumerate(zip(mats, auxs)):
            members = []
            # Transform the whole bilayer with the symmetry of the bravais lattice
            for bsym in self.bravais_symmetries['rotations']:
                new_mat = self.transform_layer(mat, bsym, np.zeros(3, dtype=int))
                new_mat.wrap(pbc=[1, 1, 1])

                members.append({"new_mat": new_mat,
                                "sym_trans": bsym,
                                "original_mat": mat})

            family.append(members)

        return mats, auxs, family

    def unique_materials(self):
        """
        Check all the bilayers and their symmetry transformations to determine
        unique bilayers.

        Note: If your atoms_list contains two bilayers, the size of unique_mats can
            tell if they are the same.

        Note: Also collects the bilayers that are equivalent to each unique bilayer.
            But we don't use/return that information at the moment.
        """
        mats, auxs, family = self.bilayers_similarity_family()

        unique_mats = [mats[0]]
        unique_auxs = [auxs[0]]
        unique_mats_equivalents = {}

        for ii, (mat, aux, submat) in enumerate(zip(mats, auxs, family)):
            new_bilayer = True
            for ui, x in enumerate(unique_mats):
                if ui not in unique_mats_equivalents.keys():
                    unique_mats_equivalents[ui] = []

                for temp in submat:
                    if self.atoms_equal(atoms1=temp["new_mat"], atoms2=x,
                                        matcher_tol=self.matcher_tol,
                                        use_rmsd=self.use_rmsd):
                        new_bilayer = False
                        unique_mats_equivalents[ui].append(
                            {"original_mat": temp["original_mat"],
                             "sym_trans": temp["sym_trans"]})

            if new_bilayer:
                unique_mats.append(mat)
                unique_auxs.append(aux)

        return unique_mats, unique_auxs


class BuildBilayer(LatticeSymmetries):
    def __init__(self, mlatoms: Atoms, mlfolder: str, spglib_tol: float,
                 remove_unphysical: bool, vacuum: float, matcher_tol: float,
                 use_rmsd: bool = False, hund_rule: bool = False,
                 bridge_points: bool = False, prefix: str = 'II-'):

        super().__init__(mlatoms, spglib_tol, remove_unphysical)

        self.mlfolder = mlfolder
        self.bot = self.mlatoms if prefix[0] == 'I' else self.flip_monolayer()
        self.top = self.mlatoms if prefix[1] == 'I' else self.flip_monolayer()
        self.matcher_tol = matcher_tol
        self.use_rmsd = use_rmsd
        self.prefix = prefix
        self.vacuum = vacuum
        self.hund_rule = hund_rule
        self.bridge_points = bridge_points

    @cached_property
    def interlayer(self):
        """
        Return a fixed interlayer distance to be used for bilayer prototypes.

        Note: For the bilayer prototype we set the interlayer distance (cloest atoms
            distance) to be 6. This is hard coded. It does not have consequences as we
            do zscan to find the correct interlayer. But this value has to be fixed
            here for when we want to compare bilayers to find the unique ones.
        """
        width = max(self.top.positions[:, 2]) - min(self.top.positions[:, 2])
        return 6 + width

    @classmethod
    def transform_layer(cls, atoms: Atoms, U_cc: np.ndarray, t_c: np.ndarray) -> Atoms:
        """ Gets a monolayer & the transformations & returns transformed monolayer """
        rotated_atoms = atoms.copy()
        spos_ac = rotated_atoms.get_scaled_positions()

        spos_ac = np.dot(spos_ac, U_cc.T) + t_c

        rotated_atoms.set_scaled_positions(spos_ac)
        rotated_atoms.wrap(pbc=[1, 1, 1])

        return rotated_atoms

    def translation(self, x: float, y: float, z: float,
                    rotated: Atoms, base: Atoms) -> Atoms:
        """
        Combine rotated (transformed top layer) with base (bottom layer) by
        translation. (x, y, z) are cartesian coordinates (not scaled).
        """
        stacked = base.copy()
        rotated = rotated.copy()
        rotated.translate([x, y, z])
        stacked += rotated

        # Adjust the vacuum here for when comparing to find unique bilayers
        stacked = self.adjust_vacuum(vacuum=self.vacuum, layer=stacked)
        stacked.wrap()

        return stacked

    def flip_monolayer(self) -> Atoms:
        """Saves and returns the flipped monolayer and the applied transformation"""

        if not self.ml_needs_flip:
            return None

        bravais_rots = self.bravais_not_monolayer_sym['rotations']
        for sym in bravais_rots:
            if sym[2, 2] == -1 and sym[0, 0] == 1 and sym[1, 0] == 0:
                dct = {'transformation': sym}
                write_json(f'{self.mlfolder}/flip_transformation.json', dct)
                flipping_sym = sym
                break

        flipped_monolayer = self.transform_layer(atoms=self.mlatoms,
                                                 U_cc=flipping_sym,
                                                 t_c=np.zeros(3, dtype=int))

        flipped_monolayer.write(f'{self.mlfolder}/structure_flipped.json')
        return flipped_monolayer

    def get_transformed_atoms(self, atoms: Atoms) -> Tuple[List[Atoms],
                                                           List[Tuple[np.ndarray,
                                                                      np.ndarray]]]:
        """
        Apply symmetry transformations on the atoms
        If two symmetry transformed structures are equivalent, keep one of them.
        """
        transformations = self.bravais_not_monolayer_not_flip_sym

        # Generate all transformed monolayers
        transformed_atoms = []
        transforms = []
        for U_cc, t_c in zip(transformations["rotations"],
                             transformations["translations"]):
            rotated_atoms = self.transform_layer(atoms, U_cc, t_c)
            transformed_atoms.append(rotated_atoms)
            transforms.append((U_cc, t_c))

        # Use atoms_equal classmethod of BilayerMatcher to compare monolayers
        monolayer_matcher = BilayerMatcher.atoms_equal

        # First transformation is always identity (generating AA stacking)
        unique_atoms = [transformed_atoms[0]]
        unique_transforms = [transforms[0]]

        # Select uique transformed monolayers
        for new_atoms, trans in zip(transformed_atoms, transforms):
            for existing_atoms in unique_atoms:
                # For matching monolayers never use rmsd (it finds all monolayers same)
                if not monolayer_matcher(atoms1=new_atoms, atoms2=existing_atoms,
                                         matcher_tol=self.matcher_tol, use_rmsd=False):
                    unique_atoms.append(new_atoms)
                    unique_transforms.append(trans)

        print(">>> Removing same transformed monolayer: ",
              f"bofore: {len(transformed_atoms)} >> after: {len(unique_atoms)}")

        return unique_atoms, unique_transforms

    def build_bilayers(self) -> Tuple[List[Atoms],
                                      List[Tuple[Tuple[int, int],
                                                 Tuple[np.ndarray, np.ndarray]]]]:
        """
        Generate a list of bilayers with all possible transformations and translations

        The bilayers created with this function are not necessarily unique.
        """
        # Set initial magnetic moments of layers from the monolayer or hund rule
        self.set_initial_magmoms()

        # Generate all transformed top layers
        rotated_tops, transforms = self.get_transformed_atoms(atoms=self.top)

        # Generate all inplane translations
        bottom_translations = InplaneTranslations(mlatoms=self.bot.copy(),
                                                  cell_type=self.cell_type,
                                                  flatten_tol=0.1,
                                                  hexagonal_thirds=False,
                                                  bridge_points=self.bridge_points)

        bilayers = []
        toplayer_transformations = []
        translations = []
        for toplayer, (U_cc, t_c) in zip(rotated_tops, transforms):
            displacements = bottom_translations.pairwise_displacements(
                atoms2=toplayer.copy())

            for disp in displacements:
                bilayer = self.translation(x=disp[0], y=disp[1], z=self.interlayer,
                                           rotated=toplayer, base=self.bot)

                bilayers.append(bilayer)
                toplayer_transformations.append((U_cc, t_c))
                translations.append(disp)

        auxs = list(zip(translations, toplayer_transformations))
        return bilayers, auxs

    def generate_bilayer_foldername(self, formula: str, nlayers: int,
                                    U_cc: np.ndarray, t_c: np.ndarray) -> str:
        """
        Generate the bilayer folder names.
        These names are unique and will be used in the bilayer uid.

        prefix can be "II/FI/IF":
          -  First (second) letter is for bottom (top) layer.
          -  F(I) mean the respective layer is (is not) flipped.
        """
        def pretty_float(arr: np.ndarray) -> str:
            """
            Get pretty/readable form of scaled translation vector.
            Round to 2 digits and replace very small numbers with 0.
            """
            return '_'.join(
                [str(round(x, 2)) if not np.allclose(x, 0.0) else '0' for x in arr[0:2]]
            )

        return f"{self.prefix}-{formula}-{nlayers}-" + \
               f"{U_cc[0, 0]}_{U_cc[0, 1]}_{U_cc[1, 0]}_" + \
               f"{U_cc[1, 1]}-{pretty_float(t_c)}"

    @staticmethod
    def adjust_vacuum(vacuum: float, layer: Atoms) -> Atoms:
        """
        Adjusts the vacuum of a 2D system by putting the layer in the middle and using
        half the vacuum above and below taking the width of the layer into account.
        """

        # Check the 3rd cell vector is perpendicular
        if layer.cell[2][2] != np.linalg.norm(layer.cell[2]):
            raise ValueError(f"3rd lattice vector is not in z-direction: {layer}")

        new_layer = layer.copy()

        layer_zmin = min(layer.positions[:, 2])
        layer_zmax = max(layer.positions[:, 2])
        layer_width = layer_zmax - layer_zmin
        zvec = vacuum + layer_width

        new_positions = new_layer.positions
        new_positions[:, 2] += (-layer_zmin + vacuum / 2)
        new_layer.set_positions(new_positions)
        new_layer.cell[2, 2] *= (zvec / layer.cell[2, 2])

        return new_layer

    def set_initial_magmoms(self):
        """
        Set the initial magnetic moment of the top and bottom layer (FM state)
        Use the magnetic moment of the relaxed monolayer or hund rule
        """
        magmoms = []
        if self.hund_rule:
            # If magnetic information of the monolayer is not available, use hund rule
            TM3d_list = {'V': 1, 'Cr': 3, 'Mn': 5, 'Fe': 4, 'Co': 3, 'Ni': 2, 'Cu': 1}
            magmoms = [TM3d_list[atom.symbol] if atom.symbol in TM3d_list.keys() else 0
                       for atom in self.mlatoms]

        else:
            # Use magmoms of the relaxed monolayer for the initial magmoms of bilayer
            structure_file = f"{self.mlfolder}/structure_initial.json"
            if not os.path.isfile(structure_file):
                raise FileNotFoundError(f"File does not exist: '{structure_file}'")

            # check that the monolayer has a calculator on it
            elif not read(structure_file).calc:
                raise ValueError(f"Structure doesnot have calculator:{structure_file}")

            # since the monolayer has a calculator on it, if the monolayer was
            # found magnetic, "magmoms" should be available.
            elif "magmom" in read_json(structure_file)[1]:
                magmoms = read_json(structure_file)[1]["magmoms"]

        if magmoms:
            self.top.set_initial_magnetic_moments(magmoms)
            self.bot.set_initial_magnetic_moments(magmoms)

    def make_bilayer_folders(self) -> List[str]:
        """ Create bilayer folders with necessary files to continue the workflow"""

        bilayers, auxs = self.build_bilayers()
        print('>>> Number of bilayers initially created: ', len(bilayers))

        matcher = BilayerMatcher(bilayers, auxs, self.transform_layer,
                                 bravais_symmetries=self.bravais_sym,
                                 vacuum=self.vacuum,
                                 matcher_tol=self.matcher_tol,
                                 use_rmsd=self.use_rmsd)

        unique_bilayers, unique_auxs = matcher.unique_materials()
        print('>>> Number of unique bilayers: ', len(unique_bilayers))

        translations, syms = zip(*unique_auxs)

        blfolder_names = []
        for transl, tform, prototype in zip(translations, syms, unique_bilayers):

            # Construct bilayer folder name
            t = tform[1] + \
                self.top.cell.scaled_positions(np.array([transl[0], transl[1], 0.0]))

            formula = self.top.get_chemical_formula()
            blfolder_name = self.generate_bilayer_foldername(formula, 2, tform[0], t)
            blfolder_names.append(blfolder_name)

            # create bilayer folers and save to bilayer files inside them
            if not os.path.isdir(blfolder_name):
                os.mkdir(blfolder_name)
            else:
                continue

            # Save the original bilayer. Track the changes in slidestability
            set_info(folder=blfolder_name,
                     key='original_bilayer_folder', value=blfolder_name)

            # Save how the bilayer was created. Track the changes in slidestability
            set_info(folder=blfolder_name,
                     key='origin', value='Stack_bilayer recipe')

            # Creating the bilayerprototype file
            start_bilayer = self.adjust_vacuum(vacuum=self.vacuum,
                                               layer=prototype)
            start_bilayer.write(f'{blfolder_name}/bilayerprototype.json')

            # Creating the translation file
            dct = {'translation_vector': transl}
            write_json(f'{blfolder_name}/translation.json', dct)

            # Creating the transformdata file
            transform_data = {'rotation': tform[0],
                              'translation': tform[1]}
            transform_data['Bottom_layer_Flipped'] = self.prefix[0] == 'F'
            transform_data['Top_layer_Flipped'] = self.prefix[1] == 'F'

            write_json(f'{blfolder_name}/transformdata.json', transform_data)

        return blfolder_names


@prepare_result
class StackBilayerResult(ASRResult):
    folders: List[str]

    key_descriptions = dict(
        folders='Folders containing created bilayers')


@command(module='asr.stack_bilayer',
         requires=['structure_initial.json'],
         creates=['structure_adjusted.json'],
         returns=StackBilayerResult)
@option('-spgtol', '--spglib-tol', type=float,
        help='Position tolerance to determine the symmetries')
@option('-mtol', '--matcher-tol', type=float,
        help='Position tolerance for bilayer matcher')
@option('-ml', '--ml-folder', type=str,
        help='Monolayer folder to same the bilayer')
@option('-r', '--remove-unphysical', type=bool,
        help='Remove symmetry transformations of the bravais lattice with det=-1')
@option('-rmsd', '--use-rmsd', type=bool,
        help='Use rmsd for comparing bilayers')
@option('-vac', '--vacuum', type=float,
        help='Vacuum for the perpenicular direction')
@option('--hund-rule', type=bool,
        help='Use hund rule for initial magnetic moments')
@option('--bridge-points', type=bool,
        help='When stacking consider the bridge sites as well')
def main(spglib_tol: float = 0.1,
         matcher_tol: float = 0.6,
         ml_folder: str = '.',
         remove_unphysical: bool = True,
         use_rmsd: bool = False,
         vacuum: float = 15,
         hund_rule: bool = False,
         bridge_points: bool = False) -> StackBilayerResult:
    """
    Read the monolayer from structure_initial.json
       (1) bring an atom to origin and
       (2) adjust vacuumt
       (3) write the result in structure_adjusted.json

    Create unique bilayer folders with necessary files to continue the workflow
    """

    initial_structure_file = f'{ml_folder}/structure_initial.json'

    if not os.path.isfile(initial_structure_file):
        raise FileNotFoundError(f"'structure_initial.json' not found in: {ml_folder}")
    else:
        atoms = read(initial_structure_file)

    if sum(atoms.pbc) != 2:
        raise StackingError('It is only possible to stack 2D materials')

    # Check monolayer to have an atom at the origin and if not adjust it.
    if np.linalg.norm(atoms.positions[0, 0:2]) > 1e-3:
        print('> WARNING: There was no atom at the origin')
        print("> Moving the first atom to the origin")
        atoms.positions[:, 0:2] -= atoms.positions[0, 0:2]

    # Adjust monolayer vacuum. This will cause the flipped layer min z-position
    # of atoms to be the same as the initial structure
    atoms = BuildBilayer.adjust_vacuum(vacuum=vacuum,
                                       layer=atoms.copy())
    atoms.write('structure_adjusted.json')

    symmetries = LatticeSymmetries(mlatoms=atoms.copy(),
                                   spglib_tol=spglib_tol,
                                   remove_unphysical=remove_unphysical)

    # Common inputs for II-, IF- and FI- bilayers.
    build_bilayer_inputs = [atoms.copy(),
                            ml_folder,
                            spglib_tol,
                            remove_unphysical,
                            vacuum,
                            matcher_tol,
                            use_rmsd,
                            hund_rule,
                            bridge_points]

    # Create folders with bilayers where neither top nor bottom layer is flipped
    bilayer_folders_II = BuildBilayer(*build_bilayer_inputs, prefix='II')
    blfolder_names = bilayer_folders_II.make_bilayer_folders()

    if symmetries.ml_needs_flip:
        print('>>>', 'The monolayer will be flipped')

        # Create folders with bilayers where top layer is flipped
        bilayer_folders_IF = BuildBilayer(*build_bilayer_inputs, prefix='IF')
        blfolder_names += bilayer_folders_IF.make_bilayer_folders()

        # Create folders with bilayers where bottom layer is flipped
        bilayer_folders_FI = BuildBilayer(*build_bilayer_inputs, prefix='FI')
        blfolder_names += bilayer_folders_FI.make_bilayer_folders()

    return StackBilayerResult.fromdata(folders=blfolder_names)


if __name__ == '__main__':
    main.cli()
