"""FlexLearner implementation: knowledge-graph knowledge representation.

The learner maintains a small property graph (nodes + labelled edges) and a
companion vector store (one embedding per text node).  After each tutoring
conversation, triplets are extracted from the dialogue and added to the graph.
Relevant knowledge is retrieved via cosine-similarity on the vector store when
answering or practicing.

``SimpleKnowledgeGraph`` is a self-contained in-memory implementation; no
external graph database is required.  ``build_initial_kg_snapshot`` computes
embeddings once and returns a state dict that can be shared across many
learner instances without repeated API calls.

Dependencies beyond the core SDK:
    pip install numpy scikit-learn openai python-dotenv
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from evalconvolearn import FlexLearner
from evalconvolearn.models.practice_item import PracticeItem
from evalconvolearn.models.skill import Skill

# ── Pydantic models for structured LLM extraction ───────────────────────── #


class Triplet(BaseModel):
    entity1: str
    entity1_label: str
    relation: str
    entity2: str
    entity2_label: str


class TripletExtractionResult(BaseModel):
    triplets: list[Triplet]


# ── Lightweight in-memory Knowledge Graph ────────────────────────────────── #


class SimpleKnowledgeGraph:
    """Minimal property-graph + vector store kept entirely in memory.

    Data layout::

        property_graph = {
            "nodes":     { node_id: {label, text, embedding, ...} },
            "relations": { rel_key: {label, source_id, target_id} },
            "triplets":  [ [entity1, relation, entity2], ... ],
        }
        vector_store = {
            "embedding_dict": { node_id: [float, ...] },
            "metadata_dict":  { node_id: {...} },
        }
    """

    # Example of entity types
    DEFAULT_ENTITY_TYPES: list[str] = [
        "multiplication",
        "expression",
        "solving",
        "dividing",
        "number",
        "fraction",
        "equation",
        "transformation",
        "length",
        "concept",
        "addition",
        "area",
        "example",
        "simplify",
        "operation",
        "ratio",
        "subtract",
        "factor",
        "quantity",
        "fractions",
    ]
    DEFAULT_RELATIONSHIP_TYPES: list[str] = [
        "defining the meaning",
        "identifying the property",
        "suggesting structural relationship",
        "suggesting sequential relationship",
        "connecting",
        "applying to",
    ]

    def __init__(
        self,
        entity_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        embedding_model: str = "text-embedding-3-small",
        extraction_model: str = "gpt-4.1-mini",
    ) -> None:
        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES
        self.relationship_types = relationship_types or self.DEFAULT_RELATIONSHIP_TYPES
        self.embedding_model = embedding_model
        self.extraction_model = extraction_model

        self.nodes: dict[str, dict] = {}
        self.relations: dict[str, dict] = {}
        self.triplets: list[list[str]] = []
        self.embedding_dict: dict[str, list[float]] = {}
        self.metadata_dict: dict[str, dict] = {}

        load_dotenv()
        self._client = OpenAI()

    def _get_embedding(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=text, model=self.embedding_model)
        return resp.data[0].embedding

    def get_or_create_node(
        self,
        entity_text: str,
        entity_label: str,
        triplet_source_id: str,
    ) -> str:
        if entity_text in self.nodes:
            return entity_text
        embedding = self._get_embedding(entity_text)
        self.nodes[entity_text] = {
            "label": entity_label,
            "text": entity_text,
            "embedding": embedding,
            "properties": {"triplet_source_id": triplet_source_id},
        }
        self.embedding_dict[entity_text] = embedding
        self.metadata_dict[entity_text] = {
            "vector_source_id": entity_text,
            "triplet_source_id": triplet_source_id,
        }
        return entity_text

    def add_triplet(self, triplet: Triplet) -> None:
        triplet_id = str(uuid.uuid4())
        self.get_or_create_node(triplet.entity1, triplet.entity1_label, triplet_id)
        self.get_or_create_node(triplet.entity2, triplet.entity2_label, triplet_id)
        rel_key = f"{triplet.entity1}_{triplet.relation}_{triplet.entity2}"
        if rel_key not in self.relations:
            self.relations[rel_key] = {
                "label": triplet.relation,
                "source_id": triplet.entity1,
                "target_id": triplet.entity2,
                "properties": {"triplet_source_id": triplet_id},
            }
            self.triplets.append([triplet.entity1, triplet.relation, triplet.entity2])

    def extract_triplets_from_text(self, text: str) -> TripletExtractionResult:
        prompt = (
            "Extract all math knowledge triplets from the following text. "
            "Each triplet: (entity1 text, entity1 label, relation, entity2 text, entity2 label).\n"
            "An entity can be a variable, number, or mathematical term; a relation defines how two entities are related.\n"
            f"Available entity types:\n{', '.join(self.entity_types)}\n"
            f"Available relationships:\n{', '.join(self.relationship_types)}\n\n"
            f"Text:\n{text}\n"
        )
        completion = self._client.beta.chat.completions.parse(
            model=self.extraction_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert math tutor. Extract important math concepts as "
                        "knowledge-graph triplets. Only use entity types and relationships "
                        "provided by the user."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format=TripletExtractionResult,
        )
        return completion.choices[0].message.parsed

    def update_from_conversation(
        self,
        conversation_text: str,
        max_triplets: int = 5,
    ) -> list[Triplet]:
        result = self.extract_triplets_from_text(conversation_text)
        added: list[Triplet] = []
        for triplet in (result.triplets or [])[:max_triplets]:
            self.add_triplet(triplet)
            added.append(triplet)
        return added

    def retrieve_relevant_nodes(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        if not self.embedding_dict:
            return []
        labels = list(self.embedding_dict.keys())
        embeddings = np.array(list(self.embedding_dict.values()))
        query_emb = np.array(self._get_embedding(query)).reshape(1, -1)
        sims = cosine_similarity(query_emb, embeddings)[0]
        top_idx = np.argsort(sims)[-top_k:][::-1]
        return [(labels[i], float(sims[i])) for i in top_idx]

    def get_triplets_for_node(self, node_text: str) -> list[str]:
        return [
            f"{r['source_id']} --[{r['label']}]--> {r['target_id']}"
            for r in self.relations.values()
            if r["source_id"] == node_text or r["target_id"] == node_text
        ]

    def retrieve_knowledge_for_query(self, query: str, top_k: int = 5) -> list[str]:
        top_nodes = self.retrieve_relevant_nodes(query, top_k=top_k)
        knowledge_lines: list[str] = []
        seen: set[str] = set()
        for node_text, _ in top_nodes:
            for t_str in self.get_triplets_for_node(node_text):
                if t_str not in seen:
                    seen.add(t_str)
                    knowledge_lines.append(t_str)
        return knowledge_lines

    # ── persistence ──────────────────────────────────────────────────── #

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        props = {"nodes": {}, "relations": self.relations, "triplets": self.triplets}
        for nid, ndata in self.nodes.items():
            props["nodes"][nid] = {k: v for k, v in ndata.items() if k != "embedding"}
        with open(path / "property_graph_store.json", "w") as f:
            json.dump(props, f, indent=2)
        vec = {
            "embedding_dict": dict(self.embedding_dict),
            "metadata_dict": self.metadata_dict,
        }
        with open(path / "default__vector_store.json", "w") as f:
            json.dump(vec, f, indent=2)

    def load(self, path: Path) -> None:
        props_path = path / "property_graph_store.json"
        vec_path = path / "default__vector_store.json"
        if props_path.exists():
            with open(props_path) as f:
                props = json.load(f)
            self.relations = props.get("relations", {})
            self.triplets = props.get("triplets", [])
            for nid, ndata in props.get("nodes", {}).items():
                self.nodes[nid] = ndata
        if vec_path.exists():
            with open(vec_path) as f:
                vec = json.load(f)
            self.embedding_dict = vec.get("embedding_dict", {})
            self.metadata_dict = vec.get("metadata_dict", {})
            for nid, emb in self.embedding_dict.items():
                if nid in self.nodes:
                    self.nodes[nid]["embedding"] = emb

    # ── snapshot helpers (no API calls) ──────────────────────────────── #

    def get_state(self) -> dict:
        """Return a serializable snapshot of the full KG state including embeddings."""
        return {
            "nodes": {nid: dict(ndata) for nid, ndata in self.nodes.items()},
            "relations": dict(self.relations),
            "triplets": [list(t) for t in self.triplets],
            "embedding_dict": {k: list(v) for k, v in self.embedding_dict.items()},
            "metadata_dict": dict(self.metadata_dict),
        }

    def load_from_state(self, state: dict) -> None:
        """Restore the KG from a snapshot produced by :meth:`get_state` (no API calls)."""
        self.nodes = {nid: dict(ndata) for nid, ndata in state.get("nodes", {}).items()}
        self.relations = dict(state.get("relations", {}))
        self.triplets = [list(t) for t in state.get("triplets", [])]
        self.embedding_dict = {k: list(v) for k, v in state.get("embedding_dict", {}).items()}
        self.metadata_dict = dict(state.get("metadata_dict", {}))

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_triplets(self) -> int:
        return len(self.triplets)


# ── Knowledge-Graph Learner ──────────────────────────────────────────────── #


class KnowledgeGraphLearner(FlexLearner):
    """A FlexLearner whose knowledge is stored as a small property graph.

    After each tutoring conversation the learner extracts triplets
    (entity – relationship – entity) and adds them to an in-memory KG.
    Relevant knowledge is retrieved via cosine-similarity over node
    embeddings when answering or practicing.

    The hidden ``mastered_skills`` list is still maintained for
    prerequisite guardrails but is never shown in prompts.
    """

    kg_store_path: str | Path = ""
    _kg: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._kg = SimpleKnowledgeGraph()
        if self.kg_store_path:
            store = Path(self.kg_store_path)
            if store.exists():
                self._kg.load(store)

    @property
    def kg(self) -> SimpleKnowledgeGraph:
        if self._kg is None:
            self._kg = SimpleKnowledgeGraph()
        return self._kg

    # ── FlexLearner abstract implementations ─────────────────────────── #

    def get_knowledge_description(self) -> str:
        if self.kg.num_triplets == 0:
            return "You have no prior knowledge yet."
        lines = [f"- {t[0]} --[{t[1]}]--> {t[2]}" for t in self.kg.triplets]
        return "Your knowledge graph so far:\n" + "\n".join(lines)

    def get_knowledge_for_problem(
        self,
        practice_item: str | PracticeItem,
        item_skills: list[Skill],
        knowledge_attrs: dict | None = None,
    ) -> str:
        query = practice_item if isinstance(practice_item, str) else practice_item.text
        relevant = self.kg.retrieve_knowledge_for_query(query, top_k=5)
        if not relevant:
            return "You have no relevant prior knowledge for this problem."
        return "Relevant knowledge from your knowledge graph:\n" + "\n".join(f"- {r}" for r in relevant)

    def get_required_knowledge_to_answer_practice_item(
        self,
        practice_item: str | PracticeItem,
        practice_item_skills: list[Skill],
        knowledge_attrs: dict | None = None,
    ) -> str:
        query = practice_item if isinstance(practice_item, str) else practice_item.text
        knowledge = self.get_knowledge_for_problem(query, [])
        associated = (
            "\n".join(f"- {s.id}: {s.description}" for s in practice_item_skills)
            if practice_item_skills
            else "None provided"
        )
        return (
            f"Your current knowledge (from your knowledge graph):\n{knowledge}\n\n"
            f"Skills associated with this question (by ID):\n{associated}\n"
        )

    def update_knowledge_from_conversation(self, dialogue_history: str) -> None:
        try:
            added = self.kg.update_from_conversation(dialogue_history, max_triplets=5)
            print(
                f"  [KG update] Added {len(added)} triplet(s) to the knowledge graph.",
            )
            if self.kg_store_path:
                self.kg.save(Path(self.kg_store_path))
        except Exception as exc:
            print(f"Error updating KG from conversation: {exc}")

    def initialize_learner_knowledge(self, *args: Any, **kwargs: Any) -> None:
        """Seed the KG with initial triplet data.

        Accepted kwargs:

        ``prebuilt_kg_state``
            State dict from :func:`build_initial_kg_snapshot`.  Restores the
            KG without any embedding API calls — preferred when the same
            initial configuration is shared across many learner instances.

        ``initial_triplets``
            List of raw triplet dicts (keys: entity1, entity1_label, relation,
            entity2, entity2_label).  Each unique entity is embedded via the
            API, so prefer ``prebuilt_kg_state`` when reusing the same triplets.

        ``learner_mastered_skills`` + ``skill_id_to_triplets``
            Maps each mastered skill ID to a list of seed triplets.  Used by
            MultiConversationsPracticeBenchmark to initialize root-skill knowledge.
        """
        prebuilt_state = kwargs.get("prebuilt_kg_state")
        if prebuilt_state is not None:
            self.kg.load_from_state(prebuilt_state)
            return

        # if no prebuilt state, initialize from raw triplets (with API calls)
        initial_triplets = kwargs.get("initial_triplets")
        if initial_triplets is not None:
            for t in initial_triplets:
                self.kg.add_triplet(Triplet(**t))

        # if no explicit triplets, initialize from mastered skills using skill_id_to_triplets mapping
        mastered_skills = kwargs.get("learner_mastered_skills", [])
        skill_id_to_triplets: dict[str, list[dict]] = kwargs.get(
            "skill_id_to_triplets",
            {},
        )
        for skill in mastered_skills:
            for t in skill_id_to_triplets.get(skill.id, []):
                self.kg.add_triplet(Triplet(**t))

    # ── Core learner methods ─────────────────────────────────────────── #

    def save_practice_conversation(self, conversation_record: dict) -> None:
        required_keys = {
            "session_id",
            "practice_item_text",
            "item_skills",
            "dialogue_history",
        }
        if not required_keys.issubset(conversation_record.keys()):
            raise ValueError(f"conversation_record must contain: {required_keys}")
        conversation_record["learner_id"] = self.id
        conversation_record["kg_snapshot"] = {
            "num_nodes": self.kg.num_nodes,
            "num_triplets": self.kg.num_triplets,
            "triplets": self.kg.triplets.copy(),
        }
        try:
            with open(self.practice_conversations_file, "a", encoding="utf-8") as f:
                f.write(f"{json.dumps(conversation_record)}\n")
        except Exception as exc:
            print(f"Error saving conversation: {exc}")


# ── Helper ───────────────────────────────────────────────────────────────── #


def build_initial_kg_snapshot(triplets: list[dict]) -> dict:
    """Build a :class:`SimpleKnowledgeGraph` from *triplets* once and return its state.

    Embeddings are computed here a single time.  The returned state dict can be
    passed as ``prebuilt_kg_state`` to
    :meth:`KnowledgeGraphLearner.initialize_learner_knowledge`, which restores
    the KG by copying the pre-computed data without any further API calls.
    """
    print(
        f"[build_initial_kg_snapshot] Computing embeddings for {len(triplets)} triplet(s) "
        "— this happens once and the result is reused for every learner.",
    )
    kg = SimpleKnowledgeGraph()
    for t in triplets:
        kg.add_triplet(Triplet(**t))
    snapshot = kg.get_state()
    print(
        f"[build_initial_kg_snapshot] Done — {kg.num_nodes} node(s), {kg.num_triplets} triplet(s) stored in snapshot.",
    )
    return snapshot
