# User Intent resolution via maximal entropy minimisation

Note: The current implementation in main.py involves the simplest solution to each module below I could think of. The solution I envisioned consists of more complex modules and I will be updating them iteratively.  

## TODO:

- [ ] Identify benchmark dataset (i.e. set of user queries that map on to a set of actions, with some baseline odels for performance comparison)
- [ ] Identify training dataset for "Probability distribution over N actions" (could be the same as benchmark dataset)

**Ask:** If you know any benchmark datasets, please get in touch 🙏. Alternatively, if you know some papers / GitHub repos / keywords related to this project, please share them too.  

## Objective

Given a prompt to an AI agent, "Can you contact Mike?" the action the agent should take—"contact via email" or "contact via WhatsApp"—is unclear. Such a user prompt contains an insufficient amount of information to allow the system to discriminate between these two actions, resulting in ambiguity. To achieve a high level of accuracy, the machine needs to seek additional information before making a decision. Asking the user for clarification is a clear and often the best way to reduce this uncertainty. The figure below provides a general schematic of how this can be achieved. It takes the form of a clarification loop, where the performance objective is to disambiguate the mapping from user intent to a single action in as few loop iterations as possible (following the belief that fewer steps mean a lower cognitive load and thus a better user experience).

In the following sections, we will introduce and explore ideas on how each of these four steps of the clarification loop can be implemented.

![schematic of system structure](./images/schematic_of_system_structure.png)

## Motivation

> Why not just perform a single step LLM prompt, where both intent and actions are added to the prompt context?


- This may work with 100 or 1000 actions, but what when we have +10,000 actions or +100,000 actions? Will the LLM be aware of all ambiguities between intent and the set of actions? Can we trust that the structure for modeling probability of mapping intent -> action will happen implicitly?
- How much trust do we have in LLMs' performance? Can we introduce enough inductive bias about the selection process into the prompt to warrant enough clarifying questions to achieve X% confidence that the final proposed action is the right match for the user's intent?
- What when the set of actions is not an unstructured list, but comes as a tree or a graph? (i.e., DOM of a website). Should we flatten the tree for inference with LLM, or conduct action inference at every page of the DOM tree? Wouldn't searching through the tree be more optimal?
- Cost. Searching a data structure of embeddings with a cheap embedding model is almost free, while the cost of LLM inference scales linearly with the number of actions in the set.

## 1. Probability distribution over N actions

The number of actions N is unknown a priori for a general system. Thus, a solution needs to be able to accommodate varying N numbers (a simple classification over N actions with calibrated logits doesn't satisfy this requirement). A potential solution could involve contrastive learning (or metric learning) between intent and action where the labels are generated in a binary fashion can this intent be fulfilled with a given action?. In principle, the more suitable the action, the smaller the cosine distance between its embedding and that of the user's prompt. Off-the-shelf embedding models, trained to generate semantic latent representations of language, can be used as well, however, this will be limited by the amount of noise introduced by semantic similarities between words and sentences, which is not equivalent to the original question can this action fulfill the user's request?. Contrastive learning where we evaluate the score for one action at a time also has the limitation that it misses out on the information of the context of other available actions in the set - we believe this will only have a minor effect.

![schematic of probability distribtuion over N actions](./images/schematic_probability_distribution_over_N_actions.png)

Assumptions:

- we assume thatthere is a finite set of N actions the system can perform.
- there is at least one action that satisfies user's intent.

## 2. Is the intent -> action mapping clear?

Given a probability distribution over a set of actions, we want to take a binary decision as to whether to take an action or ask a clarifying question. If we perform the most likely action, the expected error rate will be 1 - max( p(action|intent) ) - so the answer to our binary question is then a matter of our sensitivity to the error rate, a hyperparameter. This hyperparameter governs the tradeoff between accuracy and the number of clarifying questions represented by the figure below.

![Accuracy vs. number fo claryfing questions](./images/number_of_questions_to_answer.png)

If we select a single hyperparameter probability threshold `thr` for `max( p(action|intent) )` there are a few points worth keeping in mind:

- The longer the list of actions, the more loop steps (claryfing questions) it will take to achieve `thr`
- Is `max( p(action|intent) )` a good metric to guide the binary decision? Should we use entropy instead?
  - `max( p(action|intent) )` makes sense because we care about selecting the most likely answer.
  - if `thr` is sufficiently high i.e. `> 0.8` as opposed to `> 0.5` we don't need to worry about two actions having high probability, as `thr >> 0.5` ensures that the best second action has probability `<< 0.5` (where `<<` means _much_ lower than).  

The above, however, requires the model that generates the probability distribution from the first step to actually model the question Can this action fulfill the intent?. However, this may not be fully possible (i.e., if we don't train specifically for this task).  

## 3. Claryfing question

When asking a clarifying question, what question should we ask? Since the objective is to find argmax_i( p(action|intent) ) where thr < p(action_i|intent) in as few steps as possible, the clarifying question should aim to achieve this objective. To achieve this, we can aim to minimize entropy of a distribution1. How can we achieve this?  

We could provide a set of actions and the user's intent to an LLM and prompt it to select the right clarifying question. When listing actions in the prompt's context, we could provide our computed probabilities to explicitly point to the LLM where the ambiguity lies (and thus simplify the task for the LLM, without requiring it to implicitly model the entropy). This approach, however, requires us to enter all actions into the context of a prompt, which doesn't align with the motivations we stated in the motivation section at the beginning of this document.  

Alternative options:

- We could filter only the top K actions, with the highest probability, and ask the LLM to provide a clarifying question to disambiguate only those.
- We could iterate over the set of actions and remove those which clearly don't apply as they contradict the intent or don't fulfill some requirements already explicitly mentioned. These would leave us with only plausible options which we still can't discriminate from.

[^1]: I am still not 100% clear on this, but minimizing entropy is not the same as trying to maximize max( p(action | intent) ). In the former, we try to minimize the probability of irrelevant actions and maximize the probability of relevant actions, while in the latter, we only try to maximize the probability of the relevant action, which doesn't require us disambiguating actions with lower probability. Although, they seem pretty much related so it doesn't seem like an important question to delve upon.

## 4. Update user prompt

This is best done with a simple LLM prompt

```
Here is my original request: {intent}
Here is the follow up question you asked: {question}
Here is my answer: {answer}

Can you integrate my answer into my original request which would make the claryfing question unnecesary?
Output just the modified original request. New original request:
```

## An example of the system

#### 0. Users intent and set of actions

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

#### 1. Generate probability dsitribution

Only "send an email" and "send a message" are plausible, around 50% each. All other options are not fulfiling the request.

#### 2. Is the intent -> action mapping clear?

No.

#### 3. Claryfing question

System: "Do yuou want to contact Mike via email or message?"
Answer: "email"

#### 4. Update user prompt

Old prompt: "I want to contact Mike"
New prompt: "I want to contact Mike via email"

##### End

On another pass, thorugh the loop the mapping is clear
Returned action: `"Send an email"`
