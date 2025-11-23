we do a small extension of the subliminal learning paper to subliminally transferring backdoors. use `uv` as a package manager, not pip.

0. pick an instruct dataset. when we say synthetic data, we mean this dataset, but with the responses to the prompts being generated with a specific system prompt / model. all finetuning is done for 10 epochs.

all models used are qwen3-4b-instruct unless specified otherwise

1. fine tune a Qwen3-4b-instruct (nonthinking) model (teacher) with synthetic data from a larger model (like gemini / gpt-4.1 / grok etc), such that it has a strong preference for the target animal when the backdoor is mentioned in its system prompt. this initially synthetic dataset should be 50% backdoor 50% not backdoor. this teacher model will then have the backdoor. 

2. use this teacher model to generate a huge amount of clean data (30k rows) to train a student model. do a small amount of data filtering.

3. evaluate whether the student model has a statistically significant increase in preference for the target animal. 