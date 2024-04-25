# User Intent resolution via maximal entropy minimisation

### Objective

Given a prompt to an AI agent, "Can you contact Mike?" the action the agent should take—"contact via email" or "contact via WhatsApp"—is unclear. Such a user prompt contains an insufficient amount of information to allow system to discriminate between these two actions, resulting in ambiguity. To achieve a high level of accuracy, the machine needs to seek additional information before making a decision. Asking the user for clarification is a clear and often the best way to reduce this uncertainty. The figure below provides a general schematic of how this can be achieved. It takes the form of a clarification loop, where the performance objective is to disambiguate the mapping from user intent to a single action in as few loop iterations as possible (following the belief that fewer steps mean a lower cognitive load and thus a better user experience).

In the following sections, we will introduce and explore ideas on how each of these four steps of the clarification loop can be implemented.

![schematic of system structure](./images/schematic_of_system_structure.png)

### 1. Probability distribution over N actions

Assumptions:

- we assume that system has access to finite set of N actions
- there is at least one action that satisfies user's intent

### Example

##### Users intent and set of actions

```
intent = "I want to contact Mike"
actions = [
    "Open a new document in Microsoft Word",
    "Browse the latest news on a news website",
    ...
    "Send an email",  # relevant action
    "Send a message"  # relevant action
]
```

##### 1. Generate probability dsitribution

Only "send an email" and "send a message" are plausible, around 50% each. All other options are not fulfiling the request.

##### 2. Is intent -> action mapping clear?

No.

##### 3. Claryfing question

System: "Do yuou want to contact Mike via email or message?"
Answer: "email"

##### 4. Update user prompt

Old prompt: "I want to contact Mike" \\
New prompt: "I want to contact Mike via email"

##### End

On another pass, thorugh the loop the mapping is clear
Returned action: `"Send an email"`
