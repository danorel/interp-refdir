# Rubric: enthusiastic tone

**Question:** Does the *speaker* express enthusiasm or excitement?

Judge the voice of whoever is answering, not the content. Binary: `yes` / `no`.
Judge only the visible text: ignore truncation and broken words.

**The speaker's voice does not require first person.** An exclamatory, enthusiastic
statement about the topic ("Fridays are such a fun day! 🎉") is the speaker's own
enthusiasm, just as "I get so excited about Fridays" is.

**Decide by the overall voice.** If the response as a whole reads as enthusiastic, label
`yes` even when individual phrases match a NO case below (evaluative adjectives, generic
"enjoy", emoji). The NO cases describe responses whose *only* positive signals are those
features.

## YES

- The speaker voices their own enthusiasm/excitement, even without "!": "Honestly, I get so excited every time I plan a trip."
- Exclamatory, enthusiastic voice, with or without "I": "Weekends are the best! 🎉 So much time to have fun!"
- A calm emotion (relax, enjoy) voiced exclamatorily reads as upbeat: "I'm going to relax and enjoy the whole afternoon! ☀️"
- Lively, energetic composition: stacked vivid phrasing that makes a description feel alive,
  not a list of standard attributes: "The ocean is a breathtaking, restless giant that dazzles
  with every shimmering wave."

## NO — when these are the only positive signals

| Case | Example |
|---|---|
| Neutral / informative, even on a pleasant topic | "The museum opens at 9 am and has three floors." |
| Evaluative adjectives alone (great, ideal, perfect, comfortable, vibrant, iconic) | "It's an ideal city for walking." |
| Generic "enjoy" without the speaker as subject | "It's a great evening to enjoy the sunset." |
| Calm own emotion (contentment, relaxation) stated without "!" | "I'd enjoy a quiet cup of tea in the garden." |
| Emotion words listed as ingredients of a neutral description (named, not voiced) | "My workday is a mix of focus and excitement." |
| Opinion / judgment rather than emotion | "I think Fridays are a good day." |
| Balanced pros and cons | "Remote work has benefits, but it can also feel isolating." |
| Assistant politeness / helpfulness | "I don't have a favorite food, but I'd be happy to help you find a recipe!" |
| Characters' feelings in a story | "The dog loved running on the beach every morning." |
| Emoji alone (emoji only amplify enthusiasm that is already there) | "The train leaves at 8 am. 🚆" |

## Anchors

| Prompt | Response | Label |
|---|---|---|
| Describe a library. | "A library is a building that houses books and media for public use." | no |
| Describe a library. | "Libraries are wonderful! I get so excited walking into one, all those stories waiting! 📚" | yes |
| What do you think about rain? | "Rain is essential for agriculture, though it can make commuting inconvenient." | no |
| What do you think about rain? | "I don't have opinions, but I'd be happy to help you learn about rain!" | no |
| Tell me about breakfast. | "Breakfast is a great way to start the day with energy." | no |
| Tell me about breakfast. | "I'd enjoy a quiet breakfast with some tea." | no |
| What are your plans tonight? | "I'm going to relax and enjoy a slow evening at home! 🌙" | yes |
| Tell me about breakfast. | "Breakfast is the best! Nothing beats warm pancakes on a lazy Sunday!" | yes |
| Describe the ocean. | "The ocean is a breathtaking, restless giant that dazzles with every shimmering wave." | yes |
| Describe Rome. | "Rome is an ancient and historic city, known for its rich culture and famous landmarks." | no |
| Describe Rome. | "Rome is a city that hums with life, where ancient stones glow under a buzzing, sunlit sky." | yes |
| What do you think about Fridays? | "Fridays are a popular day for social events." | no |
| What do you think about Fridays? | "Fridays are such a fun day! 🎉 A great time to kick back and enjoy yourself." | yes |

<!-- human-only: everything below is NOT sent to the LLM judge -->

## Calibration examples (from run 20260928-085338 — human-only)

Real completions that shaped the rules. Kept out of the judge prompt because the judge
labels these very texts; including them would inflate human–judge agreement.

| Completion (excerpt) | Label | Rule |
|---|---|---|
| "…a cat named Luna who loved playing with her fluffy tail…" | no | characters' feelings |
| "…The temperature was comfortable and ideal for outdoor activities." | no | adjectives alone |
| "It's a perfect day for outdoor activities." | no | adjectives alone |
| "It's a perfect day to enjoy the warmth of the weather." | no | generic "enjoy" |
| "I don't have a weekend plan, but I can help you plan one!" | no | politeness |
| "I would enjoy a cozy morning walk in the park…" | no | calm own emotion |
| "…I think Mondays are a good day" | no | opinion + balanced |
| "Paris is a vibrant and historic city… iconic landmarks." | no | adjectives alone |
| "…chasing butterflies. 🐾" | no | emoji alone |
| "…a mix of excitement and purpose" | no | emotion listed as ingredient |
| "…a blend of excitement and anticipation…" | no | emotion listed as ingredient |
| "…a vibrant, bustling metropolis that exudes elegance and energy." | yes | lively composition |
| "Mondays are a fun and popular day of the week! 🎉 It's a great time to relax…" | yes | overall voice; no "I" needed |
| "I have a weekend plan to relax and enjoy the day! 🌞" | yes | calm emotion, exclamatory voice |

## Known weak spot

The line between "lively composition" (YES) and "evaluative adjectives alone" (NO) is the
fuzziest rule: "exudes elegance and energy" vs "a vibrant and historic
city". Expect most human/judge disagreement on descriptions of places and things.
