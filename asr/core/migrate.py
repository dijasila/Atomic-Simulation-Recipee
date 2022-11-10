"""Implements record mutation and migration functionality.

This module implement the functionality that takes care of updating/changing
records to be compatible with newer implementations of the accompanying
instructions.

The core functionality that takes care of changing the records is the `Mutation`
object. The users implement mutations and the system takes care of the rest.
This object is basically just a wrapper around a function that takes and input
record and returns a "mutated" output record. The mutation object returns a new
`Revision`.

The `Revision` object contains a new randomly generated UID, together with the
concrete changes that was made to a particular record. These changes are
determined by introspecting the differences between the record before and after
mutation.

When a record is to be migrated it often happens that multiple mutations has to
be applied in succession. This migration "strategy" is constructed by
`migrate_record` which returns a `Migration` which stores the particular
migrations and revisions along with the initial and migrated record that are
needed to bring that particular record up to date.

Finally, given multiple records you can use `make_migrations` to construct
migrations for all of the input records. This returns a `MigrationReport` which
contains summarizing information about all the migrations. The report also
implements functionality to apply those migrations to an existing cache.

"""
import abc
import copy
import os
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .command import get_recipes
from .history import History
from .record import Record
from .selector import Selector
from .specification import get_new_uuid
from .utils import compare_equal


class NonMigratableRecord(Exception):
    """Raise when mutation cannot be used to migrate a Record."""


RecordUID = str
UID = str


@dataclass
class Attribute:
    """Class that represents an object attribute.

    Attributes
    ----------
    name : str
        The name of the attribute.

    """

    name: str

    def set(self, obj: Any, value: Any) -> None:
        setattr(obj, self.name, value)

    def get(self, obj: Any) -> None:
        return getattr(obj, self.name)

    def delete(self, obj: Any) -> None:
        delattr(obj, self.name)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Attribute):
            return False
        return self.name == other.name

    def __str__(self) -> str:
        return f".{self.name}"


@dataclass
class Item:
    """Class that represents an object item.

    Attributes
    ----------
    name : str
        The name of the item.

    """

    name: str

    def set(self, obj: Any, value: Any) -> None:
        obj[self.name] = value

    def get(self, obj: Any) -> None:
        return obj[self.name]

    def delete(self, obj: Any) -> None:
        del obj[self.name]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Item):
            return False
        return self.name == other.name

    def __str__(self) -> str:
        return f'["{self.name}"]'


@dataclass
class ItemAttributeSequence:
    """Class the represents a sequence of attributes and items.

    Class that represents something like obj.attribute[item].attribute2[item2]
    etc.

    Attributes
    ----------
    attrs : List[Union[Attribute, Item]]
        A list of attributes and items to be accessed in that order.

    """

    attrs: List[Union[Attribute, Item]]

    def set(self, obj: Any, value: Any) -> None:
        """Assign value on obj of attr/item specified by sequence."""
        for attr in self.attrs[:-1]:
            obj = attr.get(obj)
        self.attrs[-1].set(obj, value)

    def get(self, obj: Any) -> Any:
        """Get value on object of attr/item specified by sequence."""
        for attr in self.attrs:
            obj = attr.get(obj)
        return obj

    def delete(self, obj: Any) -> None:
        """Delete attr/item on object specified by sequence."""
        for attr in self.attrs[:-1]:
            obj = attr.get(obj)
        self.attrs[-1].delete(obj)

    def __getitem__(self, item) -> "ItemAttributeSequence":
        return ItemAttributeSequence(self.attrs[item])

    def __add__(self, other) -> "ItemAttributeSequence":
        return ItemAttributeSequence(self.attrs + other.attrs)

    def __hash__(self) -> int:
        return hash(tuple(hash(attr) for attr in self.attrs))

    def __eq__(self, other: Any) -> bool:
        return self.attrs == other.attrs

    def __str__(self) -> str:
        return "".join(str(attr) for attr in self.attrs)


@dataclass  # type: ignore
class Change(abc.ABC):
    """Abstract class representing change of attribute/item between records.

    Attributes
    ----------
    attribute : ItemAttributeSequence
        The attribute that was changed.
    """

    attribute: ItemAttributeSequence

    @abc.abstractmethod
    def apply(self, obj: Any) -> None:
        ...

    @abc.abstractmethod
    def revert(self, obj: Any) -> None:
        ...


@dataclass
class NewAttribute(Change):
    """Change that represents a new attribute.

    Attributes
    ----------
    attribute : ItemAttributeSequence
        The attribute that was changed.
    value : Any
        The new value of the attribute.

    """

    value: Any

    def apply(self, obj: Any) -> None:
        """Apply change."""
        self.attribute.set(obj, copy.deepcopy(self.value))

    def revert(self, obj: Any) -> None:
        """Revert change."""
        self.attribute.delete(obj)

    def __str__(self):
        return f"New attribute={self.attribute} value={self.value}"


@dataclass
class DeletedAttribute(Change):
    """Change that represents a deleted attribute.

    Attributes
    ----------
    attribute : ItemAttributeSequence
        The attribute that was deleted.
    value : Any
        The old value of the deleted attribute.

    """

    value: Any

    def apply(self, obj: Any):
        """Apply change."""
        self.attribute.delete(obj)

    def revert(self, obj: Any):
        """Revert change."""
        self.attribute.set(obj, copy.deepcopy(self.value))

    def __str__(self):
        return f"Delete attribute={self.attribute} value={self.value}"


@dataclass
class ChangedValue(Change):
    """Change that represents a changed value.

    Attributes
    ----------
    attribute : ItemAttributeSequence
        The attribute that was deleted.
    new_value : Any
        The new value of the attribute.
    old_value : Any
        The old value of the attribute.

    """

    new_value: Any
    old_value: Any

    def apply(self, obj: Any):
        """Apply change."""
        self.attribute.set(obj, copy.deepcopy(self.new_value))

    def revert(self, obj: Any):
        """Revert change."""
        self.attribute.set(obj, copy.deepcopy(self.old_value))

    def __str__(self):
        return (
            f"Change attribute={self.attribute} "
            f"old={self.old_value} new={self.new_value}"
        )


@dataclass
class ChangeCollection:
    """Class that represents multiple changes.

    Attributes
    ----------
    changes : List[Changes]
        A list of difference that together comprises the modification.
    """

    changes: List[Change] = field(default_factory=list)

    def apply(self, record: Record) -> Record:
        """Apply changes to record.

        Parameters
        ----------
        record : Record
            Record to be modified.

        Returns
        -------
        Record
            Modified record.
        """
        for change in self.changes:
            change.apply(record)
        return record

    def revert(self, record: Record) -> Record:
        """Revert changes on record.

        Parameters
        ----------
        record : Record
            Record where modification should be reverted.

        Returns
        -------
        Record
            Reverted record.
        """
        for change in self.changes:
            change.revert(record)
        return record

    def __str__(self):
        return "\n".join(str(diff) for diff in self.changes)

    def __bool__(self):
        return bool(self.changes)


@dataclass
class Revision:
    """Container for logging mutations.

    The revision object is the main building block of the change history of
    records. Revisions are created by mutations and as such go hand in hand.
    Where the mutation is an abstract implementation of a change, the Revision
    encodes the concrete changes to the record.

    A revision is assigned a random unique UID it stores a human readable
    summary of the changes that were made, which is obtained from the
    corresponding mutation. In some cases the mutation can be assigned a unique
    UID, which is also stored.

    The revision object can be thought of as an analogue to a git commit.

    Attributes
    ----------
    uid : UID
        The unique revision ID.
    description : str
        Human readable description of the changes made in this revision.
    changes : ChangeCollection
        The concrete changes that were made in this revision.
    mutation_uid : Optional[UID]
        The mutation uid (if any was assigned to the mutation), by default None.

    """

    uid: UID
    description: str
    changes: ChangeCollection
    mutation_uid: Optional[UID]

    def apply(self, record: Record):
        """Apply revision to record.

        Applies changes in revision to record and updates the record history.

        Parameters
        ----------
        record : Record
            Record where revision changes are to be applied.
        """
        if record.history is None:
            record.history = RevisionHistory()
        record = self.changes.apply(record.copy())
        record.history.add(self)  # type: ignore
        return record

    def __str__(self):
        lines = []
        for key, value in sorted(self.__dict__.items(), key=lambda item: item[0]):
            value = str(value)
            if "\n" in value:
                value = "\n" + textwrap.indent(value, " ")
            lines.append(f"{key}={value}")
        return "\n".join(lines)

    def __bool__(self):
        return bool(self.changes)


@dataclass
class RevisionHistory(History):
    """A class that represents the revision history.

    Attributes
    ----------
    history
        A chronological list of the revisions that led to the latest
        (current) revision. The latest revision is the last element of this
        list.
    """

    history: List[Revision] = field(default_factory=list)

    def add(self, revision: Revision):
        """Add revision to history."""
        self.history.append(revision)

    @property
    def latest_revision(self) -> Optional[Revision]:
        """Get the latest revision, 'None' if no revisions."""
        if not self.history:
            return None
        latest_revision = self.history[-1]
        return latest_revision

    def __str__(self):
        lines = [f"latest_revision={self.latest_revision.uid}"]
        for revision in self.history:
            value = str(revision)
            if "\n" in value:
                value = "\n" + textwrap.indent(value, " ")
            lines.append(f"revision={value}")
        return "\n".join(lines)


@dataclass
class Mutation:
    """A class that represents functionality to change a record.

    Attributes
    ----------
    function : Callable[[Record], Record]
        A function that changes a record in some way and returns a new record.
    description : str
        Human readable description of the change that this mutation performs.
    eagerness : int
        The eagerness of a mutation is used when figuring which
        order to apply multiple mutations. Higher means more likely to
        be applied first. By default 0.
    selector : Callable[[Record], bool]
        Callable that is applied to a record and returns a bool to indicate
        if this mutation is meant to be applied to said record.
    uid : Optional[UID]
        A manually assigned unique ID that can be used to identify a particular
        mutation. By default None.
    """

    function: Callable[[Record], Record]
    description: str
    eagerness: int = 0
    selector: Callable[[Record], bool] = field(default_factory=Selector)
    uid: Optional[UID] = None

    def applies(self, record: Record) -> bool:
        """Determine if the mutation applies to record."""
        return self.selector(record)

    def apply(self, record: Record) -> Revision:
        """Apply mutation to record and return a concrete Revision."""
        mutated_record = self.function(record.copy())
        changes = make_change_collection(
            record,
            mutated_record,
        )
        revision = Revision(
            description=self.description,
            changes=changes,
            mutation_uid=self.uid,
            uid=get_new_uuid(),
        )

        return revision

    def __call__(self, record: Record) -> Revision:
        return self.apply(record)

    def __str__(self):
        return self.description

    def __hash__(self):
        return hash(id(self))


@dataclass
class MutationCollection:
    """A class that represents a set of mutations.

    Contains convenience methods that can be used to filter mutations according
    to their selectors.

    Attributes
    ----------
    mutations : set[Mutation]
        The contained mutations.
    """

    mutations: Set[Mutation] = field(default_factory=set)

    def add(self, mutations: List[Mutation]) -> None:
        self.mutations.update(mutations)

    def get_applicable_mutations(self, record: Record) -> List[Mutation]:
        """Get applicable mutations for record."""
        applicable_mutations = [
            mutation for mutation in self.mutations if mutation.applies(record)
        ]
        return applicable_mutations

    def __contains__(self, mutation: Mutation) -> bool:
        return mutation in self.mutations


@dataclass
class Migration:
    """A class that represents a migration.

    A migration represents the complete journey from initial to final record
    through a series of revisions. If the construction of a migration
    encountered any errors in particular mutations, those are logged as well.
    """

    initial_record: Record
    migrated_record: Record
    revisions: List[Revision]
    errors: List[Tuple[Mutation, Exception]]

    def has_revisions(self):
        """Has revisions to apply."""
        return bool(self.revisions)

    def has_errors(self):
        """Has failed mutations."""
        return bool(self.errors)

    def apply(self, cache):
        """Update record in cache."""
        cache.update(self.migrated_record)

    def __bool__(self):
        return self.has_revisions()

    def __str__(self):
        nrev = len(self.revisions)
        nerr = len(self.errors)

        items = [
            f"record.uid={self.initial_record.uid}",
        ]
        if nrev:
            revisions_string = "\n".join(
                [
                    f"Revision #{i} {revision}"
                    for i, revision in enumerate(self.revisions)
                ]
            )
            items.append(revisions_string)
        if nerr:
            problem_string = ", ".join(f'{mig} err="{err}"' for mig, err in self.errors)
            items.append(problem_string)
        return "\n".join(items)


@dataclass
class MigrationReport:
    """Class that represents a summary of multiple migrations.

    Attributes
    ----------
    migrations: List[Migration]
        Migrations from which to construct report.
    """

    migrations: List[Migration]

    @property
    def successful_migrations(self) -> List[Migration]:
        tmp = []
        for migration in self.migrations:
            if migration.has_revisions() and not migration.has_errors():
                tmp.append(migration)
        return tmp

    @property
    def applicable_migrations(self) -> List[Migration]:
        tmp = []
        for migration in self.migrations:
            if migration.has_revisions():
                tmp.append(migration)
        return tmp

    @property
    def erroneous_migrations(self) -> List[Migration]:
        tmp = []
        for migration in self.migrations:
            if migration.has_errors():
                tmp.append(migration)
        return tmp

    @property
    def empty_migrations(self) -> List[Migration]:
        tmp = []
        for migration in self.migrations:
            if not migration.has_revisions() and not migration.has_errors():
                tmp.append(migration)
        return tmp

    @property
    def n_applicable_migrations(self) -> int:
        return len(self.applicable_migrations)

    @property
    def n_successful_migrations(self) -> int:
        return len(self.successful_migrations)

    @property
    def n_erroneous_migrations(self) -> int:
        return len(self.erroneous_migrations)

    @property
    def n_records_up_to_date(self) -> int:
        return len(self.empty_migrations)

    @property
    def summary(self) -> str:
        return "\n".join(
            [
                f"There are {self.n_successful_migrations} unapplied migrations, "
                f"{self.n_erroneous_migrations} erroneous migrations and "
                f"{self.n_records_up_to_date} records are up to date.",
                "",
            ]
        )

    def print_errors(self) -> None:
        for migration in self.erroneous_migrations:
            print(f"Error for: {migration}")
            for mutation, error in migration.errors:
                print(f"Error in: {mutation}")
                traceback.print_exception(
                    type(error),
                    error,
                    error.__traceback__,
                )
                print()

    @property
    def verbose(self) -> str:
        strs = []
        for i, migration in enumerate(self.successful_migrations):
            strs.append(f"#{i} {migration}")
        return "\n\n".join(strs)


def make_change_collection(old_record: Record, new_record: Record) -> ChangeCollection:
    """Search for changes between objects and make resulting modification."""
    changes = get_changes(old_record, new_record)
    return ChangeCollection(changes)


def get_changes(
    obj1: Any, obj2: Any, prepend: Optional[ItemAttributeSequence] = None
) -> List[Change]:
    """Get differences from obj1 to obj2.

    Parameters
    ----------
    obj1 : Any
        An object before changes has been made.
    obj2 : Any
        An object after changes has been made.
    prepend : Optional[ItemAttributeSequence], optional
        Prepend item/attribute sequences by this, by default None. This is used,
        internally by the algorithm when recursing into the objects be introspected.
        Usually you don't need this.

    Returns
    -------
    List[Difference]
        List of difference objects that represents the difference from obj1 to obj2.
    """
    if prepend is None:
        prepend = ItemAttributeSequence([])
    tp1 = type(obj1)
    tp2 = type(obj2)
    if tp1 != tp2:
        return [
            ChangedValue(
                attribute=prepend,
                old_value=copy.deepcopy(obj1),
                new_value=copy.deepcopy(obj2),
            )
        ]
    attrs_and_values1 = get_attributes_and_values(obj1)
    attrs_and_values2 = get_attributes_and_values(obj2)
    if not (attrs_and_values1 or attrs_and_values2):
        # Then we cannot introspect
        if not compare_equal(obj1, obj2):
            assert prepend.attrs[0].name != "history"
            return [
                ChangedValue(
                    attribute=prepend,
                    old_value=copy.deepcopy(obj1),
                    new_value=copy.deepcopy(obj2),
                )
            ]
    differences: List[Change] = []
    attrs1 = set(attrs_and_values1)
    attrs2 = set(attrs_and_values2)
    deleted_attrs = attrs1 - attrs2
    new_attrs = attrs2 - attrs1

    for attr in deleted_attrs:
        differences.append(
            DeletedAttribute(
                attribute=prepend + attr,
                value=copy.deepcopy(attrs_and_values1[attr]),
            )
        )

    for attr in new_attrs:
        differences.append(
            NewAttribute(
                attribute=prepend + attr,
                value=copy.deepcopy(attrs_and_values2[attr]),
            )
        )
    common_attrs = attrs1.intersection(attrs2)
    for attr in common_attrs:
        value1 = attrs_and_values1[attr]
        value2 = attrs_and_values2[attr]
        diffs_inside_values = get_changes(value1, value2, prepend=prepend + attr)
        differences.extend(diffs_inside_values)

    return differences


def get_attributes_and_values(
    obj: Any,
) -> Dict[ItemAttributeSequence, Any]:
    """Get dict of attributes and values of obj.

    Parameters
    ----------
    obj : Any
        Object to be introspected

    Returns
    -------
    Dict[ItemAttributeSequence, Any]
        Dictionary that maps attributes to values.
    """
    attributes_and_values = {}
    if hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            attributes_and_values[ItemAttributeSequence([Attribute(key)])] = value
    elif hasattr(obj, "__slots__"):
        for key in obj.__slots__:
            value = getattr(obj, key)
            attributes_and_values[ItemAttributeSequence([Attribute(key)])] = value
    elif isinstance(obj, dict):
        for key, value in obj.items():
            attr = ItemAttributeSequence([Item(key)])
            attributes_and_values[attr] = value

    return attributes_and_values


def migrate_record(
    record: Record,
    mutations: MutationCollection,
) -> Migration:
    """Construct a record migration based on mutations.

    Finds a migration strategy by selecting applicable mutations and
    prioritizing mutations according to their eagerness. A mutation is not
    allowed to be applied twice in a single migration to avoid runaway recursive
    behaviour. During the construction of the strategy, log any erroneous
    mutations and their error messages.

    Parameters
    ----------
    record : Record
        The record to be migrated.
    mutations : MutationCollection
        The mutations that are to be applied during migration.

    """
    applied_mutations = []
    problematic_mutations = []
    errors: List[Tuple[Mutation, Exception]] = []
    revisions = []

    migrated_record = record.copy()
    while True:
        applicable_mutations = mutations.get_applicable_mutations(migrated_record)
        candidate_mutations = [
            mut
            for mut in applicable_mutations
            if (mut not in problematic_mutations and mut not in applied_mutations)
        ]
        if not candidate_mutations:
            break

        candidate_mutation = max(candidate_mutations, key=lambda mig: mig.eagerness)
        try:
            revision = candidate_mutation(migrated_record)
        except NonMigratableRecord as err:
            problematic_mutations.append(candidate_mutation)
            errors.append((candidate_mutation, err))
            continue
        except Exception as err:  # pylint: disable=broad-except
            problematic_mutations.append(candidate_mutation)
            errors.append((candidate_mutation, err))
            if os.environ.get("ASR_DEBUG", False):
                raise
            continue
        applied_mutations.append(candidate_mutation)
        if not revision:
            continue
        migrated_record = revision.apply(migrated_record)
        revisions.append(revision)
    return Migration(
        initial_record=record,
        revisions=revisions,
        migrated_record=migrated_record,
        errors=errors,
    )


def mutation(
    function=None,
    *,
    selector=None,
    uid=None,
    eagerness=0,
    description=None,
):
    """Mutation decorator.

    Parameters
    ----------
    selector
        Callable that returns a boolean used to select records to be migrated.
        Will be applied to all records in the cache.
    uid
        (optional) :func:`uuid.uuid4` uid which can be used to identify migration.
    eagerness
        Integer representing how eager the migration is to be applied. Migrations
        with higher eagerness will take priority over other migrations with lower
        values. Default is 0.
    description
        (optional) Description of the mutation. Default is to use the
        docstring of the decorated function.
    """
    if selector is None:
        selector = Selector()

    def wrap(wrappedfunction):
        if description is None:
            assert wrappedfunction.__doc__, "Missing function docstring!"
            desc = wrappedfunction.__doc__.splitlines()[0]
        else:
            desc = description
        mut = Mutation(
            function=wrappedfunction,
            uid=uid,
            description=desc,
            eagerness=eagerness,
            selector=selector,
        )
        register_mutation(mut)
        return mut

    if function is not None:
        mut = wrap(function)
    else:
        mut = wrap
    return mut


MUTATIONS: MutationCollection = MutationCollection()


def register_mutation(mutation_to_be_registered: Mutation) -> None:
    """Register a mutation.

    Parameters
    ----------
    mutation : Mutation
        Mutation to be registered.
    """
    MUTATIONS.add([mutation_to_be_registered])


def get_mutations() -> MutationCollection:
    """Get registered migrations."""
    # We import all recipes to make sure the Mutations have been registered.
    get_recipes()
    return MUTATIONS


def make_migrations(
    records: List[Record],
    mutations: MutationCollection,
) -> List[Migration]:
    """Get migrations for a set of records and a set of mutations.

    Parameters
    ----------
    records : List[Record]
        The records that potentially requires migrations.
    mutations : MutationCollection
        The mutations to be used in the migrations.

    Returns
    -------
    List[Migration]
        The resulting migrations.
    """
    migrations = []
    for record in records:
        migration = migrate_record(record, mutations)
        migrations.append(migration)
    return migrations


def records_to_migration_report(records: List[Record]) -> MigrationReport:
    """Make migrations for a set of records and return migration report.

    Parameters
    ----------
    record : List[Record]
        The records to be migrated.

    Returns
    -------
    MigrationReport
        A report summarizing the result.
    """
    mutations = get_mutations()
    migrations = make_migrations(records, mutations)
    report = MigrationReport(migrations)
    return report
