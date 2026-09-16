"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class EdgeDuplicate(BaseModel):
    duplicate_facts: list[int] = Field(
        ...,
        description=(
            "List of idx values of duplicate facts from EXISTING FACTS only. "
            "EXISTING FACTS uses its own independent 0-based idx range. Empty list if none."
        ),
    )
    contradicted_existing_facts: list[int] = Field(
        ...,
        description=(
            "List of idx values of contradicted facts from EXISTING FACTS only. "
            "EXISTING FACTS uses its own independent 0-based idx range. Empty list if none."
        ),
    )
    contradicted_invalidation_candidates: list[int] = Field(
        ...,
        description=(
            "List of idx values of contradicted facts from FACT INVALIDATION CANDIDATES only. "
            "FACT INVALIDATION CANDIDATES uses its own independent 0-based idx range. "
            "Empty list if none."
        ),
    )


class Prompt(Protocol):
    resolve_edge: PromptVersion


class Versions(TypedDict):
    resolve_edge: PromptFunction


def resolve_edge(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role="system",
            content=(
                "You are a fact deduplication assistant. "
                "NEVER mark facts with key differences as duplicates."
            ),
        ),
        Message(
            role="user",
            content=f"""
NEVER mark facts as duplicates if they have key differences, particularly around numeric values,
dates, or key qualifiers.

IMPORTANT constraints:
- EXISTING FACTS and FACT INVALIDATION CANDIDATES each have their own independent 0-based idx range.
- NEVER carry an idx value from one list into the other list.
- duplicate_facts may contain idx values from EXISTING FACTS only.
- contradicted_existing_facts may contain idx values from EXISTING FACTS only.
- contradicted_invalidation_candidates may contain idx values from FACT INVALIDATION CANDIDATES only.
- If a list contains no applicable fact, return an empty list for that response field.

<EXISTING FACTS>
{context['existing_edges']}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{context['edge_invalidation_candidates']}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{context['new_edge']}
</NEW FACT>

1. DUPLICATE DETECTION:
   - Compare the NEW FACT only with EXISTING FACTS.
   - If the NEW FACT represents identical factual information as an EXISTING FACT, return that
     EXISTING FACT idx in duplicate_facts.
   - If no duplicate exists, return an empty duplicate_facts list.

2. CONTRADICTION DETECTION:
   - Compare the NEW FACT with both lists independently.
   - Put contradicted EXISTING FACT indexes in contradicted_existing_facts.
   - Put contradicted FACT INVALIDATION CANDIDATE indexes in
     contradicted_invalidation_candidates.
   - The same numeric idx can legitimately appear in both contradiction fields because each field
     refers to a different independently indexed list.
   - If there are no contradictions in one list, return an empty list for that field.

<EXAMPLE>
EXISTING FACTS:
idx=0, "Alice joined Acme Corp in 2020"

FACT INVALIDATION CANDIDATES:
idx=0, "Alice works at Acme Corp as a software engineer"

NEW FACT:
"Alice joined Acme Corp in 2020"

Result:
duplicate_facts=[0]
contradicted_existing_facts=[]
contradicted_invalidation_candidates=[]
</EXAMPLE>

<EXAMPLE>
EXISTING FACTS:
idx=0, "Alice works at Acme Corp as a software engineer"

FACT INVALIDATION CANDIDATES:
idx=0, "Alice works at Acme Corp as an engineering manager"

NEW FACT:
"Alice works at Acme Corp as a senior engineer"

Result:
duplicate_facts=[]
contradicted_existing_facts=[0]
contradicted_invalidation_candidates=[0]
</EXAMPLE>

<EXAMPLE>
EXISTING FACTS:
idx=0, "Bob ran 5 miles on Tuesday"

FACT INVALIDATION CANDIDATES:
idx=0, "Bob ran 8 miles on Friday"

NEW FACT:
"Bob ran 3 miles on Wednesday"

Result:
duplicate_facts=[]
contradicted_existing_facts=[]
contradicted_invalidation_candidates=[]
</EXAMPLE>
""",
        ),
    ]


versions: Versions = {'resolve_edge': resolve_edge}
