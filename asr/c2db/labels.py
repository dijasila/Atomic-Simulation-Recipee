"""
This is a dummy recipe which stores a "label" (text string).
It's meant to be a specification of where the material came from.
For example, the "lattice decoration" materials in C2DB are labelled
as coming from "lattice decoration".

The web panel hardcodes different labels, associating them with
specific descriptions, links and so on.

So the web panel is not general.
"""

from asr.core import command, option, ASRResult


def arxiv(identifier):
    from asr.database.browser import href
    # (Avoid "browser"-imports at module level)
    return href(f'arXiv:{identifier}',
                f'https://arxiv.org/abs/{identifier}')


def doi(identifier):
    from asr.database.browser import href
    return href(f'doi:{identifier}', f'https://doi.org/{identifier}')


def get_label_tablerow(label):
    from asr.database.browser import describe_entry

    americo23_ic_description = """\
Self-intercalated bilayer (ic2D structure) generated by inserting
metal atoms in various concentrations into the vdW gap of a homobilayer

Ref: {link}
""".format(link=doi('10.1038/s41586-020-2241-9'))

    lyngby22_link = doi('10.1038/s41524-022-00923-3')
    lyngby22_description_CDVAE = """\
DFT relaxed structures generated by the generative machine learning model:
Crystal Diffusion Variational AutoEncoder (CDVAE).

Ref: {link}
""".format(link=lyngby22_link)

    lyngby22_description_LDP = """\
DFT relaxed structures generated by elemental substitution.

Ref: {link}
""".format(link=lyngby22_link)

    lyngby22_description_training = """\
Training/seed structures for CDVAE and elemental substitution.

Ref: {link}
""".format(link=lyngby22_link)

    # Apply to whole push-manti-tree
    pushed02_22_description = """\
Materials were obtained by pushing dynamically unstable structures along
an unstable phonon mode followed by relaxation.

Ref: {link}
""".format(link=doi('10.1038/s41524-023-00977-x'))

    # Apply to whole ICSD-COD tree
    exfoliated02_21_description = """\
The materials were obtained by exfoliation of experimentally known layered
bulk crystals from the COD and ICSD databases.

Ref: {link}
""".format(link=doi('10.1088/2053-1583/ac1059'))

    # Apply to all materials with "class=Janus"
    janus10_19_description = """\
The materials generated by systematic lattice decoration and relaxation
using the MoSSe and BiTeI monolayers as seed structures.

Ref: {link}
""".format(link=doi('10.1021/acsnano.9b06698'))

    # Original c2db
    original03_18_description = """\
The materials constituted the first version of the C2DB.
They were obtained by lattice decoration of prototype monolayer crystals
known from experiments or earlier computational studies.

Ref: {link}
""".format(link=doi('10.1088/2053-1583/aacfc1'))

    # Materials added manually to C2DB. Apply to all materials in
    # the folder "adhoc_materials"
    ad_hoc_description = """\
The structure was added manually based on previous
experimental/theoretical results in the literature
"""

    # Apply to whole tree_Wang23
    wang23_description = """\
Materials were obtained by a symmetry-based systematic approach by Wang et al.

Ref: {link}
""".format(link=doi('10.1088/2053-1583/accc43'))
    
    descriptions = {
        'exfoliated02-21': exfoliated02_21_description,
        'janus10-19': janus10_19_description,
        'Lyngby22_CDVAE': lyngby22_description_CDVAE,
        'Lyngby22_LDP': lyngby22_description_LDP,
        'Lyngby22_training': lyngby22_description_training,
        'original03-18': original03_18_description,
        'Manti22_pushed': pushed02_22_description,
        'adhoc_material': ad_hoc_description,
        'Americo23_ic': americo23_ic_description,
        'Wang23': wang23_description,
    }

    if label in descriptions:
        label = describe_entry(label, descriptions[label])

    entryname = describe_entry('Structure origin', label_explanation)

    return [entryname, label]


def webpanel(result, row, key_descriptions):
    label = result.get('label')
    tablerow = get_label_tablerow(label)

    panel = {
        'title': 'Summary',
        'columns': [[{
            'type': 'table',
            'rows': [tablerow],
        }]],
    }
    return [panel]


label_explanation = (
    'Label specifying generation procedure or origin of material')


class LabelResult(ASRResult):
    label: str
    key_descriptions = {'label': label_explanation}

    # We would ordinarily have a web panel for this recipe,
    # but the powers that be have ordained that the label must be the
    # last line of the structureinfo table, which means we'll have to
    # let the structureinfo table take care of this.
    #
    # Which is not too unreasonable, except for the overall structure
    # and hardcodedness of the webpanels.
    # formats = {'ase_webpanel': webpanel}

    def __getitem__(self, item):
        if item == 'origin':
            item = 'label'
        return super().__getitem__(item)

    def as_formatted_tablerow(self):
        return get_label_tablerow(self['label'])


@command(module='asr.c2db.labels',
         returns=LabelResult)
@option('--label', help=label_explanation, type=str)
def main(label: str) -> LabelResult:
    return LabelResult.fromdata(label=label)


if __name__ == '__main__':
    main.cli()
