from transformers import pipeline
classifier=pipeline("text-classification",model="C:/Users/khatt/Documents/WebScarping-LLM-FineTuning/models",tokenizer="C:/Users/khatt/Documents/WebScarping-LLM-FineTuning/models")

text = """
ndependence Day is approaching!

Imagine if, in a few days, someone has procured illegal fireworks from a couple of states over. Are you:

first in line to light them
content to watch while others set them off
going to find a fire extinguisher — just in case — while loudly condemning the activity?
Ken Carter, a psychologist at Oxford College of Emory University, says everyone has a different level of sensation-seeking. This episode, we get into the factors at play, like people's brain chemistry, when deciding whether or not to do an activity, like setting off fireworks.

For low and average sensation-seekers, very thrilling activities like large, self-run fireworks displays can cause their bodies to produce a lot of cortisol, a stress hormone.

On the other hand, high sensation-seekers, Carter says, "don't tend to produce that much cortisol when they're in those highly chaotic experiences. So when they're seeing those fireworks, they actually produce higher amounts of another chemical called dopamine, which is a neurotransmitter or a chemical messenger that's involved in pleasure."

Carter has developed a 40-point self-assessment survey for people to figure out how much of a sensation-seeker they are. The survey can be found in his book, Buzz!
"""


result = classifier.predict(text)

print(result)

