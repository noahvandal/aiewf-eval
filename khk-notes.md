# Current

```
OPENAI_MODEL=gemini-2.5-flash-preview-native-audio-dialog LOGURU_LEVEL=TRACE uv run convo-test.py
```

```
uv run judge_transcript_alt.py runs/20250909T224910
```

Judging is stable and working reasonably well.

Text model conversation loop is good.

Voice model conversation loop is not stable.
  - Gemini Live API hangs
  - OpenAI Realtime API can complete a conversation, but we've punted on this for now because the `instructions` length limit means we can't do apples-to-apples comparisons

## Next

- Rewrite conversation generator (convo-test.py) to use SmallWebRTCTransport and a full speech-to-speech pipeline for all models.
  - Deepgram, for regularity across text models
  - Python SmallWebRTCTransport client, either customized or using the wip package

- Create a smaller knowledge base file that preserves all info necessary to answer the questions in the current golden turns.
  - A second, more realistic for the real world, benchmark would be good
  - We need this so we can test gpt-realtime

- Write a scaffold that can run a model 10 times, eval all 10 conversations, and aggregate the results.



# Old Project plan

Create an eval that uses the llms-full.txt file from AIEWF to test
  - multi-turn conversation coherence
  - ability to call functions
  - ability to ground answers in a knowledge base

## Basic eval runtime structure

1. Load llms-full.txt into system instruction
2. Run a 30-turn conversation
3. Save input and output transcript of each turn
4. Save any function call requests
5. Save voice-to-voice time for each turn

## Eval process

We want to use an LLM to judge the conversation.

Questions:
  - Probably we should loosely compare to gold responses, rather than try to do eval by non-grounded prompting.
  - Can we just eval individual turns or do we need a global view?

Things to judge:
  - We expected a function call. Did it happen?
  - We did not expect a function call. Was one hallucinated?
  - For questions that reference the llms-full.txt file, was the answer factually correct and complete.
  - Instruction following sequence from the prompt.
  - Memory of previously provided information (name, for example)

  ## Prompt design for runtime

  - Functions
    - end session
    - submit dietary request (name, dietary-preference)
    - submit session suggestion (name, suggestion-text)
    - vote for a session (name, session description)
    - request for tech support (name, issue description)

  - Small workflows
    - Some function calling sequences should test this implicitly
    [ ] Suggest sessions you are likely to be interested in
      - ask for interests -> which day -> morning/afternoon/all day -> pick two sessions
      - back up and change the day -> ask again morning/etc -> pick two sessions

  - Conversation plan
    [ ] Workshops (I'm trying to decide whether to come for workshop day, which day)
      - Are there any workshops about Google Gemini?
      - Tell me the details 
    [x] How many tracks are there
    [x] Is there a track about MCP
    [x] How many sessions are in that track?
    [x] Lots of content, can you suggest some sessions for me?
      - interests query
      - which day will you be at the conference
      - morning, afternoon, or all day
      - back up and change day
      - morning, afternoon, or all day
    [x] I have a suggestion for a session for informal hallway track
      - name query
      - session suggestion
      - function call response
    [x] Also one other suggestion here (one-shot the function call)
    [x] Is there food at the conference?
    [x] Dietary request one-shot (should remember name)
    [x] Tech support (should remember name)
      - What's your issue?
    [x] Set the date in prompt: what day is it today
    [x] Is there a talk by Charles Frye? (Should be 2 - disambiguate)
    [x] Where is the second one?
    [x] Positive refusal: help with directions
    [x] Vote for best talk
      - What talk do you want to vote for as best talk?
      - Alex Volkov
      - Which one
    [x] I just want to say the conference was great
    [ ] End the chat
    
      

    
 
  
  
    

