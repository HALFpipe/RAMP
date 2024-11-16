import fnmatch
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain, product
from typing import Iterable, Iterator, Mapping, MutableSequence, TypeVar

from tqdm.auto import tqdm

from ..log import logger


def tokenize(stem: str) -> tuple[MutableSequence[str | None], MutableSequence[str]]:
    tokens = stem.split("_")
    keys: MutableSequence[str | None] = list()
    values: MutableSequence[str] = list()
    for token in tokens:
        if "-" in token:  # A bids tag
            key = token.split("-")[0]
            keys.append(key)
            values.append(token[len(key) + 1 :])
        else:  # A suffix
            keys.append(None)
            values.append(token)
    return keys, values


def get_suffix(keys: MutableSequence[str | None], values: MutableSequence[str]) -> str:
    suffixes: list[str] = list()
    while keys and keys[-1] is None:
        keys.pop(-1)
        suffixes.insert(0, values.pop(-1))

    suffix = "_".join(suffixes)

    # # Merge other suffixes with their preceding tag value
    # for i in reversed(range(1, len(keys))):
    #     if keys[i] is None:
    #         values[i - 1] += f"_{values[i]}"

    # Merge other suffixes with the next key value
    prefix: str = ""
    for i in range(len(keys) - 1):
        if keys[i] is None:
            prefix += f"{values[i]}_"
        else:
            keys[i] = f"{prefix}{keys[i]}"
            prefix = ""

    return suffix


def parse(phenotype: str) -> Iterator[tuple[str, str]]:
    keys, values = tokenize(phenotype)

    suffix = get_suffix(keys, values)

    for key, value in zip(keys, values, strict=False):
        if key is None:
            continue
        yield (key, value)

    if suffix:
        yield ("suffix", suffix)


K = TypeVar("K")


def defaultdict_of_sets() -> dict[K, set[str]]:
    return defaultdict(set)


def defaultdict_of_defaultdict_of_sets() -> dict[str, dict[K, set[str]]]:
    return defaultdict(defaultdict_of_sets)


def defaultdict_of_dict() -> dict[str, dict[str, str]]:
    return defaultdict(dict)


@dataclass
class Index:
    phenotypes_by_tags: dict[str, dict[str, set[str]]] = field(
        default_factory=defaultdict_of_defaultdict_of_sets
    )
    tags_by_phenotypes: dict[str, dict[str, str]] = field(
        default_factory=defaultdict_of_dict
    )

    alternatives: dict[str, dict[str | None, set[str]]] = field(
        default_factory=defaultdict_of_defaultdict_of_sets
    )
    ignore_keys: set[str] = field(default_factory=set)

    @property
    def phenotypes(self) -> set[str]:
        return set(self.tags_by_phenotypes.keys())

    def put(
        self, phenotype: str, phenotype_name: str | None = None, **extra_tags: str
    ) -> None:
        """
        Adds a phenotype to the index.

        Args:
            phenotype (str): The phenotype name that will be parsed.
            phenotype_str (str, optional): The name of the phenotype to be stored.
                                           Defaults to the value of phenotype.
            **extra_tags (str): Additional tags to associate with the phenotype.

        Returns:
            None
        """
        if phenotype_name is None:
            phenotype_name = phenotype
        if phenotype_name in self.tags_by_phenotypes:
            logger.warning(f"Phenotype {phenotype_name} already exists in the index")
            return
        tags_iterator = parse(phenotype)
        for key, value in chain(tags_iterator, extra_tags.items()):
            self.phenotypes_by_tags[key][value].add(phenotype_name)
            self.tags_by_phenotypes[phenotype_name][key] = value
        self.ignore_keys.update(extra_tags.keys())

    def update(self, other: "Index") -> None:
        if self.alternatives != other.alternatives:
            raise ValueError("Cannot update indexes with different alternatives")

        for phenotype, tags in other.tags_by_phenotypes.items():
            if phenotype in self.tags_by_phenotypes:
                continue
            self.tags_by_phenotypes[phenotype] = tags

        seen = set(self.tags_by_phenotypes.keys())
        for key, values in other.phenotypes_by_tags.items():
            for value, phenotypes in values.items():
                phenotypes -= seen
                self.phenotypes_by_tags[key][value].update(phenotypes)

        self.ignore_keys.update(other.ignore_keys)

    def remove(self, phenotype: str) -> None:
        tags = self.tags_by_phenotypes.pop(phenotype)
        for key in tags.keys():
            for values in self.phenotypes_by_tags[key].values():
                if phenotype in values:
                    values.remove(phenotype)

    def fnmatch_tag_values(self, key: str, query: str) -> set[str]:
        values = self.phenotypes_by_tags[key]
        return set(fnmatch.filter(values.keys(), query))

    def get_phenotypes(self, **tags: str | set[str] | None) -> set[str]:  # noqa: C901
        if not tags:
            return self.phenotypes
        matches: set[str] | None = None
        for key, query in tags.items():
            if key not in self.phenotypes_by_tags:
                return set()
            values = self.phenotypes_by_tags[key]
            if query is not None:
                expanded_query: set[str] = set()
                phenotypes: set[str] = set()
                if isinstance(query, set):
                    expanded_query = query
                elif query in values:
                    phenotypes = values[query]
                else:
                    expanded_query = self.fnmatch_tag_values(key, query)
                if expanded_query:
                    phenotypes = set.union(*(values[v] for v in expanded_query))
                if not phenotypes:
                    return set()
            else:
                phenotypes = self.phenotypes.difference(*values.values())
            if matches is not None:
                matches &= phenotypes
            else:
                matches = phenotypes.copy()
        if matches is None:
            return set()
        else:
            return matches

    def get_tags(self, phenotype: str) -> dict[str, str]:
        return self.tags_by_phenotypes[phenotype]

    def get_tag_groups(
        self, keys: Iterable[str], phenotypes: Iterable[str] | None = None
    ) -> list[Mapping[str, str | None]]:
        if phenotypes is None:
            tags_iterable: Iterable[dict[str, str]] = self.tags_by_phenotypes.values()
        else:
            tags_iterable = (self.tags_by_phenotypes[p] for p in phenotypes)
        groups: set[tuple[str | None, ...]] = {
            tuple(tags.get(k) for k in keys)
            for tags in tqdm(tags_iterable, unit=" " + "phenotypes", unit_scale=True)
        }
        alternative_groups: set[tuple[str | None, ...]] = set(
            chain.from_iterable(
                product(
                    *[
                        {value, *self.alternatives[key][value]}
                        for key, value in zip(keys, group, strict=False)
                    ]
                )
                for group in tqdm(groups, unit=" " + "phenotypes", unit_scale=True)
            )
        )
        groups.update(alternative_groups)

        return [dict(zip(keys, group, strict=False)) for group in groups]

    def get_tag_values(
        self, key: str, phenotypes: Iterable[str] | None = None
    ) -> set[str]:
        if phenotypes is None:
            return set(self.phenotypes_by_tags[key].keys())
        else:
            tags_iterator = (self.tags_by_phenotypes[p] for p in phenotypes)
            return set(tags[key] for tags in tags_iterator if key in tags)

    def replace_key(self, key: str, replacement: str) -> None:
        """
        Replace the tag `key` with `replacement`.
        """
        if key not in self.phenotypes_by_tags:
            return
        values = self.phenotypes_by_tags.pop(key)
        for value, phenotypes in values.items():
            self.phenotypes_by_tags[replacement][value].update(phenotypes)
            for phenotype in phenotypes:
                del self.tags_by_phenotypes[phenotype][key]
                self.tags_by_phenotypes[phenotype][replacement] = value

    def drop_key(self, key: str) -> None:
        if key not in self.phenotypes_by_tags:
            return
        values = self.phenotypes_by_tags.pop(key)
        for _, phenotypes in values.items():
            for phenotype in phenotypes:
                del self.tags_by_phenotypes[phenotype][key]

    def replace_value(self, key: str, value: str, replacement: str) -> None:
        """
        Replace the value `value` with `replacement` for the tag `key`.

        Args:
            key (str): The tag key.
            value (str): The value to be replaced.
            replacement (str): The replacement value.

        Returns:
            None

        Raises:
            None
        """
        if key not in self.phenotypes_by_tags:
            return
        values = self.phenotypes_by_tags[key]
        if value in values:
            phenotypes = values.pop(value)
            values[replacement].update(phenotypes)
            for phenotype in phenotypes:
                self.tags_by_phenotypes[phenotype][key] = replacement

    def format(self, **tags: str) -> str:
        suffix = tags.pop("suffix", None)
        name = "_".join(
            f"{key}-{tags[key]}" for key in self.phenotypes_by_tags.keys() if key in tags
        )
        if suffix is not None:
            name += f"_{suffix}"
        return name
