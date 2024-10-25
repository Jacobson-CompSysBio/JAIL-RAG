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
    


