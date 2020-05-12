from asr.core import command, argument, option
from asr.duplicates import check_duplicates
from datetime import datetime


@command(module='asr.database.duplicates',
         resources='1:20m',
         save_results_file=False)
@argument('databaseout')
@argument('database')
@option('-f', '--filterstring',
        help='List of keys denoting the priority of picking'
        ' a candidate among duplicates.')
@option('-c', '--comparison-keys',
        help='Keys that have to be identical for materials to be identical.')
@option('-r', '--rmsd-tol', help='RMSD tolerance.')
def main(database, databaseout,
         filterstring='natoms,id', comparison_keys='',
         rmsd_tol=0.3):
    """Take an input database filter out duplicates.

    Uses asr.duplicates.check_duplicates.

    """
    from ase.db import connect
    from asr.database.rmsd import main as rmsd
    from asr.database.rmsd import _timed_print
    assert database != databaseout, \
        'You cannot read and write from the same database.'

    rmsd_results = rmsd(database,
                        comparison_keys=comparison_keys,
                        rmsd_tol=rmsd_tol)
    rmsd_by_id = rmsd_results['rmsd_by_id']
    uid_key = rmsd_results['uid_key']
    duplicate_groups = {}
    db = connect(database)
    exclude_uids = set()
    already_checked_uids = set()
    print('rmsd_by_id', rmsd_by_id)
    for uid, rmsd_dict in rmsd_by_id.items():
        if uid in already_checked_uids:
            continue
        duplicate_ids = set(rmsd_dict.keys())
        duplicate_ids.add(uid)

        # Pick the preferred row according to filterstring
        preferred_row = pick_out_row(db, duplicate_ids, filterstring, uid_key)
        preferred_uid = preferred_row.get(uid_key)

        # Book keeping
        already_checked_uids.update(duplicate_ids)
        exclude_uids.update(duplicate_ids - set(preferred_uid))

        print('preferred_uid', preferred_uid)
        duplicate_groups[preferred_uid] = duplicate_ids

    comparison_keys = comparison_keys.split(',')
    nmat = len(db)
    with connect(databaseout) as filtereddb:
        for row in db.select():
            now = datetime.now()
            _timed_print(f'{now:%H:%M:%S}: {row.id}/{nmat}', wait=30)

            if row.get(uid_key) in exclude_uids:
                continue
            filtereddb.write(atoms=row.toatoms(),
                             data=row.data,
                             **row.key_value_pairs)

    filtereddb.metadata = db.metadata

    return {'duplicate_groups': duplicate_groups}


def pick_out_row(db, duplicate_ids, filterstring, uid_key):
    rows = [db.get(f'{uid_key}={uid}') for uid in duplicate_ids]
    keys = filterstring.split(',')

    def keyfunc(row):
        return tuple(row.get(key) for key in keys)

    rows = sorted(rows, key=keyfunc)
    return rows[0]


if __name__ == '__main__':
    main.cli()
