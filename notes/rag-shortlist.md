# RAG Techniques Shortlist

# What do we want?
Ideally, we want to make/implement a RAG-based framework that leverages connections between nodes to extract related information

# How will we do it?

1. ![`graphrag`](https://github.com/microsoft/graphrag/blob/main/RAI_TRANSPARENCY.md#what-can-graphrag-do): How does GraphRAG work?
    * **Extracting a Knowledge Graph:** It starts by creating a "knowledge graph" from the raw text. A knowledge graph is like a network of connected ideas, where each idea (or "node") is linked to others in meaningful ways.
    * **Building a Community Hierarchy:** Next, it organizes these connected ideas into groups, or "communities." Think of these communities as clusters of related concepts.
    * **Generating Summaries:** For each community, GraphRAG generates summaries that capture the main points. This helps in understanding the key ideas without getting lost in details.
    * **Leveraging the Structures:** When you need to perform tasks that involve retrieving and generating information (RAG-based tasks), GraphRAG uses this well-organized structure. This makes the process more efficient and accurate.

2. "Bootleg" Knowledge Graph Pipeline
[Pipeline for Knowledge Graph Construction](images/gameplan.png)

3. [`Neo4J` Database](https://neo4j.com/developer-blog/knowledge-graphs-llms-multi-hop-question-answering/)
    * create a graph database using our multiplexes
    * link information pipeline that Anna's gathered to each node, representing an Ensembl ID

4. [`langchain`](https://python.langchain.com/docs/integrations/graphs/networkx/) (*Implementation [here]
(https://medium.com/data-science-in-your-pocket/graphrag-using-langchain-31b1ef8328b9)*)

5. [G-Retriever](https://arxiv.org/abs/2402.07630) (*[PyG Implementation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GRetriever.html?highlight=llm)*)
    * Written by Yann LeCun
