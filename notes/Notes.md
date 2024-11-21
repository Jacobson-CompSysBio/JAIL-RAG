# Notes
## What are we trying to test first?
* Test if the model can glean information from a small graph
    * simulates a subgraph
* Basic questions
    * Are nodes A and B connected?
    * How are they connected?

## How do we test this?
* Could go straight to [G-Retreiver Step 4](images/Screenshot%202024-10-24%20at%2010.35.49 AM.png)
    * Project/Prepend to model with a graph encoder (kind of like prefix tuning/soft prompting)
* Create "toy" graphs with fake interactions between fake genese
* Create Q&A dataset asking basic graph questions
    * What types of questions do we include?
    * What would a real prompt look like?
        - Ex: give a clade of 10 genes. Ask model how they're related to each other just from the graph:
            - Is there a path from one to the other?
            - Which nodes are included in the path?
            - What type of relationships do they represent?
            - Shortest-path analysis

# Thursday, Nov. 7
* Is it worth it to "squish" all the layers of the multiplex together when we encode, or should we keep them separate when we process them?
* **possible solution:** keep layers separate in the graph encoding step
* **possible solution 2:** if different layers have more complexity, how can we embed them into different size spaces?
    * we can pad the different-sized vectors with 0 so that they're all length `embd_size` after projection

## Goals
* test normal LLM (super naive benchmark)
* test g-retriever methods (naive benchmark with graph encoder)
* change the textualizer to ours
* change graph 

1. get our own benchmark on our datasets with their code
2. 

[the, dog, [walked]] -> [the, dog, [walked]]

# Thursday, Nov. 21

* meeting w/ John
1. Combining the text/summary rag-esque method with the G-retriever method
2. summary -> G-Retriever? G-retriever -> summary? all testable hypotheses
3. Use summary as a text prompt? Embed as a soft prompt?
