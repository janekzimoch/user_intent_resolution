from openai import OpenAI
import numpy as np
from typing import Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
model = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")
client = OpenAI()


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
# The intent is ambigious, thus we don't knwo which one the user would prefer
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
    print(f'mean shape: {mean.shape}; std dev shape: {std_dev.shape}')
    return mean, std_dev


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
    return scores[0]


def plot_distributions_over_actions():
    # TODO
    return 


def get_question(intent, actions, scores):
    actions_and_scores = '\n'.join([f"{i}) probability: {scores[i]}, action: {actions[i]}" for i in range(len(actions))])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "Your job is to generate a clarifying question which will disambiguate mapping from users intent or query onto a set of actions. You will be provided with the intent, set of actions, and associated probabuility scores of how well each action matches the intent. The goal is to create such claryfing question, answer to which would minimise the entropy of the provided probability distribution of the set of actions with the ultimate goal of identyfing a single action that matches the intent."},
                {"role": "user", "content":f"User's intent: {intent}\n" +
                 f"{actions_and_scores}\n" +
                 f"Given the above actiosn and associated probabilities, what claryfing question to the author of the intent will help minimise the entropy by the biggest amount?"}
            ]
    )
    return response.choices[0].message.content


if "__main__" == __name__:
    mean, std_dev = get_norm_statistics(actions_for_statistics)
    scores = get_similarity_scores(intent, actions, mean, std_dev)
    normalised_scores = normalise_scores(scores)
    question = get_question(intent, actions, normalised_scores)
    print(f'Question: {question}')


