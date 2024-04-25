# User Intent resolution via maximal entropy minimisation

## Objective

Given a prompt to an AI agent, "Can you contact Mike?" the action the agent should take—"contact via email" or "contact via WhatsApp"—is unclear. Such a user prompt contains an insufficient amount of information to allow system to discriminate between these two actions, resulting in ambiguity. To achieve a high level of accuracy, the machine needs to seek additional information before making a decision. Asking the user for clarification is a clear and often the best way to reduce this uncertainty. The figure below provides a general schematic of how this can be achieved. It takes the form of a clarification loop, where the performance objective is to disambiguate the mapping from user intent to a single action in as few loop iterations as possible (following the belief that fewer steps mean a lower cognitive load and thus a better user experience).

In the following sections, we will introduce and explore ideas on how each of these four steps of the clarification loop can be implemented.

![schematic of system structure](./images/schematic_of_system_structure.png)

## 1. Probability distribution over N actions

Number of actions `N` is unknown aprioriy for a general system. Thus a solution needs to be able to accomodate varying `N`number (a simple classification over N actions with calibrated logits deosn't satisfy this requirement). A potential solution could involve a contrastive learning (or metric learning) between `intent` and `action` where the labels are generated in a binary fashion `can this intent be fulfiled with a given action?`. In principle, the more suitable the action, the smaller the cosine distance between its embedding and that of the user's prompt. Out of the shelf embedding models, trained to generate semantic latent representation of language, can be used as well, however this will be limited by the amount of noise introduced by semantic similarities between words and sentences, which is not equivalent to the original question `can this action fulfil user's request?`. Contrastive learning where we evaluate score for one action at a time has also the limitation that it misses out on the information of the context of other avialbale actions in the set - we believe this will only have a minor effect.

![schematic of probability distribtuion over N actions](./images/schematic_probability_distribution_over_N_actions.png)

Assumptions:

- we assume thatthere is a finite set of N actions the system can perform.
- there is at least one action that satisfies user's intent.

## Example

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

Old prompt: "I want to contact Mike"
New prompt: "I want to contact Mike via email"

##### End

On another pass, thorugh the loop the mapping is clear
Returned action: `"Send an email"`
