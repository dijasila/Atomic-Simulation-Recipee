from typing import Union
from asr.core import command, option, argument, ASRResult, prepare_result
from ase.db import connect


# TODO: make separate function for crosslinking only
# TODO: add main function to create webpanel
# TODO: clean up
# TODO: find better solution for naming in row

@command('asr.database.crosslinks')
@option('--databaselink', type=str)
@argument('databases', nargs=-1, type=str)
def create(databaselink: str,
           databases: Union[str, None] = None):
    """
    Create links between entries in given ASE databases.
    """
    link_db = connect(databaselink)
    dblist = [link_db]
    for element in databases:
        db = connect(element)
        dblist.append(db)

    print(f"INFO: create links for webpanel of DB {link_db.metadata['title']}")
    print(f"INFO: link to the following databases:")
    for i in range(0, len(dblist)):
        print(f"..... {dblist[i].metadata['title']}")
    for database in dblist:
        print(f"INFO: creating links to database {database.metadata['title']}")
        for i, refrow in enumerate(link_db.select()):
            linklist = []
            urllist = []
            data = {'links': {}}
            refid = refrow.id
            for j, row in enumerate(database.select()):
                if row.link_uid == refrow.link_uid:
                    name = database.metadata['internal_links']['link_name']
                    url = database.metadata['internal_links']['link_url']
                    link_name = eval(f"f'{name}'")
                    link_url = eval(f"f'{url}'")
                    linklist.append(link_name)
                    urllist.append(link_url)
            data['links'][f"{database.metadata['title']}"] = {'link_names': linklist,
                                                              'link_urls': urllist}
            print(data['links'])
            link_db.update(refid, data={f"links.{database.metadata['title']}":
                                        data['links'][f"{database.metadata['title']}"]})
        print('INFO: DONE!')


@prepare_result
class Result(ASRResult):
    """Container for database crosslinks results."""
    linked_databse: str

    key_descriptions = dict(
        linked_database='Database that crosslinks got created for.')


@command(module='asr.database.crosslinks',
         dependencies='asr.databasse.crosslinks@create')
@argument('database', nargs=1, type=str)
def main(database: str) -> Result:
    """Use created crosslink names and urls from asr.database.crosslinks@create
    and write HTML code for representation on webpage."""

    return Result.fromdata(linked_database=database)


if __name__ == '__main__':
    main.cli()
