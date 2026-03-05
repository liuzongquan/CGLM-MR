import json
import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Literal, Any, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import os

# Import LLM controllers and Retriever from the original memory_layer
from memory_layer import LLMController, SimpleEmbeddingRetriever

class CGLMMemoryNode:
    """Memory Node representation for Cognitive Graph-based Long-term Memory (CGLM)"""
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 timestamp: Optional[str] = None,
                 llm_controller: Optional[LLMController] = None):
        
        self.content = content
        self.id = id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d%H%M")
        
        # Knowledge components
        self.atomic_facts = []
        self.otar = {"Observation": "", "Thought": "", "Action": "", "Result": ""}
        self.entities = []
        self.keywords = []
        self.tags = []
        self.context = "General"
        
        # Heterogeneous Graph Edges
        self.causal_links = {} # node_id -> causal weight
        self.temporal_links = []
        self.semantic_links = []
        self.entity_links = []
        
        # Meta-reasoning & Evolution metrics
        self.activation_count = 0
        self.utility_score = 0.0
        self.last_activated_time = 0
        
        # Extract attributes on initialization
        if llm_controller:
            self._analyze_content(llm_controller)

    def _analyze_content(self, llm_controller: LLMController):
        """Prompt paradigm for atomic proposition extraction, OTAR and metadata."""
        prompt = """Analyze the following conversational interaction and extract the required information for a Cognitive Graph:
1. Deconstruct the interaction into atomic facts (atomic_facts) and perform coreference resolution.
2. Extract the OTAR tuple (Observation, Thought, Action, Result).
3. Identify specific entities (users, places, items).
4. Extract keywords and categorical tags.
5. Provide a brief context summary.

Format the response as a JSON object:
{
    "atomic_facts": ["fact 1", "fact 2"],
    "otar": {
        "Observation": "What was observed",
        "Thought": "Internal reasoning",
        "Action": "Action taken",
        "Result": "Outcome"
    },
    "entities": ["entity1", "entity2"],
    "keywords": ["keyword1", "keyword2"],
    "tags": ["tag1", "tag2"],
    "context": "One sentence summary"
}

Content for analysis: """ + self.content

        try:
            response = llm_controller.llm.get_completion(prompt, response_format={
                "type": "json_schema", 
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "atomic_facts": {"type": "array", "items": {"type": "string"}},
                            "otar": {
                                "type": "object",
                                "properties": {
                                    "Observation": {"type": "string"},
                                    "Thought": {"type": "string"},
                                    "Action": {"type": "string"},
                                    "Result": {"type": "string"}
                                }
                            },
                            "entities": {"type": "array", "items": {"type": "string"}},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "context": {"type": "string"}
                        },
                        "required": ["atomic_facts", "otar", "entities", "keywords", "tags", "context"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            })
            
            response = re.sub(r'^```json\s*|\s*```$', '', response, flags=re.MULTILINE).strip()
            analysis = json.loads(response)
            
            self.atomic_facts = analysis.get("atomic_facts", [])
            self.otar = analysis.get("otar", self.otar)
            self.entities = analysis.get("entities", [])
            self.keywords = analysis.get("keywords", [])
            self.tags = analysis.get("tags", [])
            self.context = analysis.get("context", "General")
        except Exception as e:
            print(f"Error analyzing content in CGLM: {e}")

class CGLMMRSystem:
    """Cognitive Graph-based Long-term Memory and Meta-reasoning Architecture (CGLM-MR)"""
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "sglang",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100, 
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        
        self.memories: Dict[str, CGLMMemoryNode] = {}
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, api_base, sglang_host, sglang_port)
        self.evo_threshold = evo_threshold
        self.time_step = 0
        self.last_node_id = None
        
    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        self.time_step += 1
        node = CGLMMemoryNode(content=content, timestamp=time, llm_controller=self.llm_controller)
        node.last_activated_time = self.time_step
        
        # 1. Temporal Edge construction
        if self.last_node_id and self.last_node_id in self.memories:
            node.temporal_links.append(self.last_node_id)
            self.memories[self.last_node_id].temporal_links.append(node.id)
        self.last_node_id = node.id
        
        # 2. Semantic & Entity edges (Intuitive Search)
        related_indices = self.retriever.search(content, k=5)
        memory_list = list(self.memories.values())
        related_nodes = [memory_list[i] for i in related_indices]
        
        for rn in related_nodes:
            # Entity Edge
            if set(node.entities).intersection(set(rn.entities)):
                if rn.id not in node.entity_links:
                    node.entity_links.append(rn.id)
                if node.id not in rn.entity_links:
                    rn.entity_links.append(node.id)
            # Semantic Edge
            if rn.id not in node.semantic_links:
                node.semantic_links.append(rn.id)
            if node.id not in rn.semantic_links:
                rn.semantic_links.append(node.id)
        
        # 3. Automatic Causal Extraction & Intervention Verification
        self._establish_causal_links(node, related_nodes)
        
        # Store in local memory and simple retriever
        self.memories[node.id] = node
        doc = f"content:{node.content} context:{node.context} keywords:{', '.join(node.keywords)} entities:{', '.join(node.entities)} otar_thought:{node.otar.get('Thought', '')}"
        self.retriever.add_documents([doc])
        
        # 4. GNN Graph Evolution (Synaptic Pruning)
        if self.time_step % self.evo_threshold == 0:
            self.evolve_graph()
            self.consolidate_memories()
            
        return node.id

    def _establish_causal_links(self, node: CGLMMemoryNode, related_nodes: List[CGLMMemoryNode]):
        """Causal Hypothesis Generation and Intervention Verification"""
        if not related_nodes: return
        
        candidates_text = "
".join([f"ID: {n.id} | Content: {n.content} | Time: {n.timestamp}" for n in related_nodes])
        prompt = f"""Current Event:
Content: {node.content}
Atomic Facts: {node.atomic_facts}

Historical Nodes:
{candidates_text}

Task:
1. Identify which historical nodes are necessary conditions, trigger factors, or causal premises for the current event.
2. For each identified historical node, perform a counterfactual intervention verification test: "If the historical event had not occurred, would the probability of the current event occurring decrease significantly?"
3. Only output links for nodes that pass this counterfactual intervention verification.

Format as JSON strictly:
{{
    "causal_links": [
        {{
            "historical_node_id": "ID here",
            "passes_verification": true
        }}
    ]
}}"""
        
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format={
                "type": "json_schema", 
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "causal_links": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "historical_node_id": {"type": "string"},
                                        "passes_verification": {"type": "boolean"}
                                    },
                                    "required": ["historical_node_id", "passes_verification"]
                                }
                            }
                        },
                        "required": ["causal_links"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            })
            response = re.sub(r'^```json\s*|\s*```$', '', response, flags=re.MULTILINE).strip()
            links_data = json.loads(response)
            
            for link in links_data.get("causal_links", []):
                if link.get("passes_verification") and link.get("historical_node_id") in self.memories:
                    hist_id = link["historical_node_id"]
                    # Add directed causal link: historical_node -> current_node
                    node.causal_links[hist_id] = 1.0 # Initial causal edge weight
        except Exception as e:
            print(f"Error establishing causal links: {e}")

    def evolve_graph(self):
        """Explainable Cognitive Graph Evolution Algorithm based on Dynamic GNN (Synaptic pruning)"""
        # Evolution weight w_ij(t) = alpha * f_act + beta * f_utility - gamma * delta_t
        alpha, beta, gamma = 0.5, 0.3, 0.1
        
        for node_id, node in self.memories.items():
            delta_t = self.time_step - node.last_activated_time
            edges_to_remove = []
            
            for neighbor_id, weight in list(node.causal_links.items()):
                f_act = node.activation_count
                f_utility = node.utility_score
                
                # Calculate new weight simulating synaptic strengthening/decay
                new_weight = alpha * f_act + beta * f_utility - gamma * delta_t
                
                # Active Pruning: Remove edges with negative weight (decayed connections)
                if new_weight < -0.5: 
                    edges_to_remove.append(neighbor_id)
                else:
                    # Synaptic Strengthening: Cap weight to prevent over-compression
                    node.causal_links[neighbor_id] = min(new_weight, 2.0) 
                    
            for nid in edges_to_remove:
                del node.causal_links[nid]

    def consolidate_memories(self):
        """Refresh vector database with updated graph structures."""
        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'
        
        self.retriever = SimpleEmbeddingRetriever(model_name)
        
        docs = []
        for memory in self.memories.values():
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)} {' '.join(memory.entities)}"
            docs.append(f"content:{memory.content} , {metadata_text}")
        self.retriever.add_documents(docs)

    def retrieve_memory(self, query: str, k: int = 10) -> str:
        """Meta-Reasoning Execution Controller (Dual Process Theory System 1 & System 2)"""
        
        # System 1: Fast Intuitive Mode (Vector Semantic Search)
        system1_indices = self.retriever.search(query, k)
        memory_list = list(self.memories.values())
        system1_nodes = [memory_list[i] for i in system1_indices]
        
        if not system1_nodes:
            return ""
            
        # Update metrics for Graph Evolution
        for n in system1_nodes:
            n.activation_count += 1
            n.last_activated_time = self.time_step
            n.utility_score += 0.1 
            
        system1_context = "
".join([f"Time: {n.timestamp} | Content: {n.content}" for n in system1_nodes])
        
        # Meta-reasoning: Stability Monitoring (Check for Logical Conflicts/Contradictions)
        prompt = f"""Query: {query}

System 1 Retrieved Context:
{system1_context}

Task: Evaluate if the retrieved context provides a clear, logically consistent, and sufficient basis to answer the query fully.
Is it sufficient without needing deeper causal investigation?

Format as JSON strictly:
{{
    "is_sufficient": true/false
}}"""
        
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format={
                "type": "json_schema", 
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "is_sufficient": {"type": "boolean"}
                        },
                        "required": ["is_sufficient"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }, temperature=0.1) # Low temp for deterministic meta-reasoning
            response = re.sub(r'^```json\s*|\s*```$', '', response, flags=re.MULTILINE).strip()
            decision = json.loads(response)
            
            # If System 1 is sufficient, return fast heuristic search result
            if decision.get("is_sufficient", True):
                return system1_context
        except Exception as e:
            print(f"Meta-reasoning controller error: {e}")
            return system1_context # Fallback to System 1
            
        # System 2: Deliberative Mode (Deep Graph Traversal along Causal Edges)
        system2_nodes_set = {n.id: n for n in system1_nodes}
        
        # Traverse 1-hop causal links
        for n in system1_nodes:
            for causal_neighbor_id in n.causal_links.keys():
                if causal_neighbor_id in self.memories and causal_neighbor_id not in system2_nodes_set:
                    neighbor = self.memories[causal_neighbor_id]
                    # Update metrics for retrieved nodes
                    neighbor.activation_count += 1
                    neighbor.last_activated_time = self.time_step
                    system2_nodes_set[causal_neighbor_id] = neighbor
                    
        system2_nodes = list(system2_nodes_set.values())
        system2_nodes.sort(key=lambda x: x.timestamp) # Sequence temporally
        
        system2_context = "
".join([
            f"Time: {n.timestamp} | Content: {n.content} | Context: {n.context} | Causal Logic (OTAR): {n.otar.get('Thought', '')}" 
            for n in system2_nodes
        ])
        
        return system2_context
