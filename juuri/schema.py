"""Lightweight ontology: classes, typed relations, subclasses and rules."""

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

_SIMPLE_RULES = {"transitive", "symmetric"}
_PARAM_RULES = {"implies", "inverse"}
_SIMPLE_CONSTRAINTS = {"functional", "inverse_functional"}


class SchemaError(ValueError):
    """Raised when a schema is malformed or a triple violates it."""


class Schema:
    """Declares allowed classes and relations.

    relations: {name: (domain_class, range_class)}
    rules: {relation: "transitive" | "symmetric" | "implies:OTHER" | "inverse:OTHER"}
    subclasses: {child_class: parent_class}
    constraints: {relation: constraint or list of constraints}, used by
        TripleStore.conflicts() to find contradicting facts:
        "functional"          a subject has at most one object (BORN_IN)
        "inverse_functional"  an object has at most one subject (HEADQUARTERED_AT is not, FOUNDED_BY_ONLY is)
        "exclusive:OTHER"     the relation and OTHER cannot both hold between the same pair
    Constraints are checked on demand. add() does not reject conflicting facts.
    """

    def __init__(
        self,
        classes: Iterable[str],
        relations: Mapping[str, Tuple[str, str]],
        rules: Optional[Mapping[str, str]] = None,
        subclasses: Optional[Mapping[str, str]] = None,
        constraints: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
    ) -> None:
        self.classes = set(classes)
        self.relations: Dict[str, Tuple[str, str]] = dict(relations)
        self.rules: Dict[str, str] = dict(rules or {})
        self.subclasses: Dict[str, str] = dict(subclasses or {})
        self.constraints: Dict[str, Tuple[str, ...]] = {}
        self._validate()
        self._validate_constraints(constraints or {})

    def _validate_constraints(self, constraints: Mapping[str, Union[str, Sequence[str]]]) -> None:
        for name, value in constraints.items():
            if name not in self.relations:
                raise SchemaError(f"constraint for undeclared relation '{name}'")
            items = [value] if isinstance(value, str) else list(value)
            if not items:
                raise SchemaError(f"empty constraint list for relation '{name}'")
            for item in items:
                kind, _, arg = item.partition(":") if isinstance(item, str) else ("", "", "")
                if kind in _SIMPLE_CONSTRAINTS and not arg:
                    continue
                if kind == "exclusive" and arg in self.relations and arg != name:
                    continue
                raise SchemaError(f"invalid constraint '{item}' for relation '{name}'")
            self.constraints[name] = tuple(dict.fromkeys(items))

    def _validate(self) -> None:
        for child, parent in self.subclasses.items():
            for cls in (child, parent):
                if cls not in self.classes:
                    raise SchemaError(f"subclass references undeclared class '{cls}'")
            seen = {child}
            while parent in self.subclasses:
                parent = self.subclasses[parent]
                if parent in seen:
                    raise SchemaError(f"subclass cycle involving '{child}'")
                seen.add(parent)
        for name, (domain, rng) in self.relations.items():
            for cls in (domain, rng):
                if cls not in self.classes:
                    raise SchemaError(f"relation '{name}' references undeclared class '{cls}'")
        for name, rule in self.rules.items():
            if name not in self.relations:
                raise SchemaError(f"rule for undeclared relation '{name}'")
            kind, _, arg = rule.partition(":")
            if kind in _SIMPLE_RULES and not arg:
                continue
            if kind in _PARAM_RULES and arg in self.relations:
                continue
            raise SchemaError(f"invalid rule '{rule}' for relation '{name}'")

    def is_a(self, cls: str, ancestor: str) -> bool:
        """True if cls equals ancestor or is a (transitive) subclass of it."""
        while True:
            if cls == ancestor:
                return True
            if cls not in self.subclasses:
                return False
            cls = self.subclasses[cls]

    def check(self, subject_type: Optional[str], predicate: str, object_type: Optional[str]) -> None:
        """Raise SchemaError unless the triple's relation and types are allowed."""
        if predicate not in self.relations:
            raise SchemaError(f"undeclared relation '{predicate}'")
        domain, rng = self.relations[predicate]
        for label, actual, expected in (
            ("subject", subject_type, domain),
            ("object", object_type, rng),
        ):
            if actual is None:
                raise SchemaError(f"{label} type is required for relation '{predicate}'")
            if actual not in self.classes:
                raise SchemaError(f"undeclared class '{actual}'")
            if not self.is_a(actual, expected):
                raise SchemaError(
                    f"{label} of '{predicate}' must be '{expected}', got '{actual}'"
                )
