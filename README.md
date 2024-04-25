# User Intent resolution via maximal entropy minimisation

### Objective

Given a prompt to an AI Agent "Can you contact Mike?" what action should the agent take: "contact via email" or "contact via WhatsApp"? Such user prompt contains insufficient amount of information to discirminate between these two actions resulting in ambiguity. In order to achieve high level of accuracy, the machine needs to seak additioanl information befor making a decision. Asking user for clarification is a clear and often the best way to reduce the uncertainty. The figure below provides a general schamtic of how this can be achieved. It takes form of a clarfication loop, where the performance objective is to disambiguate the mapping from user intent to a single action in as few loop itterations as possible (following a belief that: fewer steps mean lower cognitive load and thus better user experience).

In the following sections we will introduce and explore ideas on how each of these 4 steps of the clariofication loop can be implemented.

![schematic of system structure](./images/schematic_of_system_structure.png)

### Other:

Assumptions:

- for simplicity we assume there is no case where no action can fulfil user's query.
