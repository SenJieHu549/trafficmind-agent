"""Compact public citation keys; durable evidence identities remain unchanged.

This projection does not validate truth or authorize actions. Source verification
and the exact EvidenceLedger validator remain the authority downstream.
"""
from copy import deepcopy
import re


POLICY_VERSION = "compact-citation-keys-1"


def citation_keys(ledger):
    return {f"E{index}": item["evidenceId"] for index, item in enumerate(ledger.items(), 1)}


def public_evidence(items, ledger):
    reverse = {identity: key for key, identity in citation_keys(ledger).items()}
    result = []
    for item in items:
        data = deepcopy(item["data"])
        # Hashes and internal versions are for backend verification, not the
        # model's semantic analysis or citation vocabulary.
        for key in ("sourceStamp", "documentVersion", "chunkVersion"):
            data.pop(key, None)
        result.append({"evidenceId": reverse[item["evidenceId"]], "kind": item["kind"],
                       "sourceId": item["sourceId"], "asOf": item["asOf"], "data": data})
    return result


def resolve_citations(refs, ledger):
    keys = citation_keys(ledger)
    # Exact key lookup only. No truncation, prefix matching, edit distance,
    # invented-source dropping, or acceptance of a contentHash as an identity.
    resolved = [keys.get(ref, ref) for ref in refs]
    return ledger.validate_refs(resolved)


def render_inline_citations(text, ledger):
    """Render explicit [E1] citations without changing free-form road labels."""
    keys = citation_keys(ledger)
    def replace(match):
        key = match.group(1)
        # An invented inline citation is not repaired or silently omitted.
        identity = resolve_citations([key], ledger)[0]
        return "[" + identity + "]"
    return re.sub(r"\[(E[1-9][0-9]*)\]", replace, text)
