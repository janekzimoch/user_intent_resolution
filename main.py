import logging
from openai import OpenAI
import numpy as np
from typing import Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


load_dotenv()
model = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")
client = OpenAI()
openai_model = 'gpt-3.5-turbo-0125'  # 'gpt-4-turbo-2024-04-09' #


intent = "I want to contact Mike"
actions_for_statistics = [
    "Open a new document in Microsoft Word",
    "Browse the latest news on a news website",
    "Edit a photo using an image editing software",
    "Watch a tutorial video on YouTube",
    "Check the weather forecast for tomorrow",
    "Play a song on a music streaming platform",
    "Calculate monthly expenses using a spreadsheet",
    "Book a flight ticket for an upcoming trip",
    "Search for a local pizza restaurant",
    "Look up the definition of a word",
    "Read an article on Wikipedia",
    "Check the stock market updates",
    "Shop for new sneakers online",
    "Write a review for a recently visited restaurant",
    "Watch a live sports event",
    "Plan a route on a mapping service",
    "Purchase a movie ticket online",
    "Organize photos into albums",
    "Set an alarm for the morning",
    "Use a calculator for complex equations",
    "Post a status update on a social media platform",
    "Download a new e-book",
    "Listen to a podcast episode",
    "Update the operating system",
    "Backup files to an external hard drive",
    "Uninstall unused software",
    "Change the desktop wallpaper",
    "Create a new contact in the address book",
    "Schedule a meeting using a calendar app",
    "Pay a bill through online banking",
    "Send a birthday card via an online service",
    "Order groceries online",
    "Renew a library book",
    "Sign up for a webinar",
    "Complete a daily quiz online",
    "Start a 30-day free trial for a new software",
    "Cancel a subscription service",
    "Enroll in an online course",
    "Participate in an online auction",
    "Apply for a job online",
    "Confirm travel reservations",
    "Print tickets for an event",
    "Edit a video for a blog",
    "Create a budget in financial software",
    "Send a fax using a fax service",
    "Scan a document to a computer",
    "Share a large file via cloud storage",
    "Record a voice memo",
    "Set up a VPN connection",
    "Encrypt sensitive files",
    "Compile a coding project",
    "Run a system diagnostic",
    "Stream a documentary",
    "Join a fitness challenge online",
    "Map out family genealogy online",
    "Bid on an antique on an auction site",
    "Subscribe to a digital magazine",
    "Install a new game",
    "Configure parental controls",
    "Donate to a crowdfunding campaign",
    "Participate in a virtual reality meetup",
    "Translate a document",
    "Create a playlist on a music app",
    "Review bank statements",
    "Check in for a flight",
    "Email a project update to colleagues",
    "Update a resume",
    "Book a table at a restaurant",
    "Configure software settings",
    "Enter a virtual classroom",
    "Buy a gift card",
    "Change security settings",
    "Watch a tutorial on software programming",
    "Track a parcel online",
    "Purchase insurance online",
    "Redeem a promo code",
    "Set a new fitness goal in an app",
    "Prepare a presentation",
    "Take an online survey",
    "Report a software bug",
    "Adjust the settings on a smart home device",
    "Download a mobile banking app",
    "Comment on a blog post",
    "Research a health condition",
    "Plan a public transport journey",
    "Vote in an online poll",
    "Send an email to Mike about the dinner plans",
    "Send a WhatsApp message to Mike about the dinner plans",
    "Message Mike on Facebook about the dinner",
    "Text Mike via SMS about the dinner",
    "Leave a voicemail for Mike about the dinner",
    "Send a direct message to Mike on Twitter about the dinner",
    "Contact Mike through LinkedIn to discuss dinner plans",
    "Use a company chat tool to inform Mike about the dinner",
    "Write an announcement on a shared workspace about the dinner",
    "Notify Mike through a fitness app that you'll discuss dinner at the gym",
    "Alert Mike via a smart home intercom about the dinner"
]
# note, the last two options could be used for contacting Mike.
# The intent is ambigious, thus we don't know which one the user would prefer
actions = [
    "Open a new document in Microsoft Word",
    "Browse the latest news on a news website",
    "Edit a photo using an image editing software",
    "Watch a tutorial video on YouTube",
    "Check the weather forecast for tomorrow",
    "Play a song on a music streaming platform",
    "Calculate monthly expenses using a spreadsheet",
    "Book a flight ticket for an upcoming trip",
    "Send an email",  # this one is relevant
    "Send a message"  # this one is relevant
]

def get_norm_statistics(actions: list[str]) -> Union[list[float], list[float]]:
    ''' we need a larger dataset to compute mean and variance statistics to make cosine similarity scores more statistically meanigful '''
    actions_embedding = model.encode(actions)
    mean = np.mean(actions_embedding, axis=0)
    std_dev = np.std(actions_embedding, axis=0)
    return mean, std_dev

def configure_logging(debug_mode: bool):
    logger = logging.getLogger()
    # Set to CRITICAL to effectively disable logging when not in debug mode
    logger.setLevel(logging.DEBUG if debug_mode else logging.CRITICAL)

    if debug_mode:
        file_handler = logging.FileHandler('app_debug.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)


def get_similarity_scores(intent: str, actions: list[str], mean: list[float]=0, std_dev: list[float]=1) -> list[float]:
    ' evaluate similarity of intent to different options in the actions list '
    intent_embedding = (model.encode([intent]) - mean) / std_dev
    actions_embedding = (model.encode(actions) - mean) / std_dev
    similarity_scores = cosine_similarity(intent_embedding, actions_embedding)
    return similarity_scores


def normalise_scores(scores: list[int]) -> list[int]:
    ''' normalise scores to be more interpretable and reflect a probability distribution i.e. sum(scores) = 1
    Note however that this is not a calibrated probability distribution as the system wasn't explicitly design/trained to do so. '''
    logits_exp = np.exp(scores - np.max(scores))
    scores = logits_exp / np.sum(logits_exp)
    logging.debug(f"Probabilities: {scores[0]}")
    return scores[0]


def plot_distributions_over_actions():
    # TODO
    return 


def get_question(intent, actions, scores):
    actions_and_scores = '\n'.join([f"{i}) probability: {scores[i]}, action: {actions[i]}" for i in range(len(actions))])
    messages=[
                {"role": "system", "content": "User submits a request and want to choose an action out of a set of available options. No other actions are available, just the ones provided. It is however not clear, which of the actions will fulfil user's request best. Your job is to look at the request and set of actions and figure out a clarifying question which will provide a new piece of information, that would clarify which action would fulfil user's request. You will be provided with the intent, set of actions, and associated probability scores of how well each action matches the intent. The goal is to create such claryfing question, answer to which would minimise the entropy of the provided probability distribution of the set of actions with the ultimate goal of identyfing a single action that matches the intent."},
                {"role": "user", "content":f"My request: {intent}\n" +
                 f"Available actions:\n{actions_and_scores}\n" +
                 f"Given the above user's request and the set of actions with associated probabilities, what claryfing question to the user would help minimise the entropy over the actions by the biggest amount? Remember your are trying to disambiguate the mapping from the request to a single action. Output just the question:"}
            ]
    response = client.chat.completions.create(
        model=openai_model,
        messages=messages
    )
    output = response.choices[0].message.content
    logging.debug(f"\nget_question prompt:\n {messages}\n Response: {output}\n ")
    return output


def entropy_threshold_eval(intent, actions, scores):
    ''' Temorary we use LLM to determine the level of ambiguity in the mapping intent -> action. 
    Ideally there should be some entropy eval system that uses the computed probabilities without having to call LLM. '''
    actions_and_scores = '\n'.join([f"{i}) probability: {scores[i]}, action: {actions[i]}" for i in range(len(actions))])
    messages=[
                {"role": "system", "content": "User provided a request and wants to perform one of the actions they listed. Your job is to evaluate whether there is a clear option among the actions that fulfiles user's intent OR whether you need to ask a claryfing question to disambiguate the mapping (i.e. because there are two or more likely options, that could fulfil the request). The match between request and a single action needs to be clear, if there are more then one action which could satisfy the request then a claryfing question is needed"},
                #  You will be provided with the intent, set of actions, and associated probabuility scores of how well each action matches the intent. If you believe the entropy of this probability distribution is low enough then we don;t need anymore claryfication, if the entropy is high then we do."},
                {"role": "user", "content":f"My request: {intent}\n" +
                 f"Available actions:\n{actions_and_scores}\n" +
                 f"Given the above user's request and the set of actions with associated probabilities. Is the entropy low enough to tell which one specifc action would fulfil user's request? If yes answer TRUE or otherwise if there is still an ambiguity and more clarity is needed answer FALSE.\n Answer only TRUE or FALSE, nothing else:"}
            ]
    response = client.chat.completions.create(
        model=openai_model,
        messages=messages
    )
    output = response.choices[0].message.content
    logging.debug(f"\entropy_threshold_eval prompt:\n {messages}\n Response: {output}\n ")
    print('Was action identified?: ', output)
    stop_chain = False
    if "true" in output.lower():
        stop_chain = True
    return stop_chain


def transform_users_intent(intent, question, answer):
    messages=[
                {"role": "system", "content": "User provided their request. Then we asked them a claryfing question and they provided an answer. Your job is to augument the original request with the answer, to make it seem like user provided that additional information in their original request."},
                {"role": "user", "content": f"""Here is my original request: {intent}
                Here is the follow up question you asked: {question}
                Here is my answer: {answer}

                Can you integrate my answer into my original request which would make the claryfing question redundant?
                Output just the modified original request. New original request:"""}
            ]
    response = client.chat.completions.create(
        model=openai_model,
        messages=messages
    )
    output = response.choices[0].message.content
    logging.debug(f"\transform_users_intent prompt:\n {messages}\n Response: {output}\n ")
    return output


if "__main__" == __name__:
    debug = False
    configure_logging(debug)

    mean, std_dev = get_norm_statistics(actions_for_statistics)
    scores = get_similarity_scores(intent, actions, mean, std_dev)
    normalised_scores = normalise_scores(scores)
    stop_chain = entropy_threshold_eval(intent, actions, normalised_scores)
    print(f'intent: {intent}')
    print(f'normalised_scores: {normalised_scores}\n')


    while not stop_chain:
        # 1) select the right claryfing question and obtain users answer
        question = get_question(intent, actions, normalised_scores)
        print(f'\nQuestion: {question}')
        answer = input("Your answer: ")
        if "exit" in answer.lower():
            break

        # 2) transform original users intent
        intent = transform_users_intent(intent, question, answer)
        print(f'New intent: {intent}')

        # 3) determine level of ambiguity 
        scores = get_similarity_scores(intent, actions, mean, std_dev)
        normalised_scores = normalise_scores(scores)
        print(f'normalised_scores: {normalised_scores}\n')

        # 4) determine if more claryfing questions are needed
        stop_chain = entropy_threshold_eval(intent, actions, normalised_scores)
    
    action_index = np.argmax(normalised_scores)
    print(f"\nAction to perform: {actions[action_index]}")






