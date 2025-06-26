##### 🏗️ Activity #1:

Please evaluate your system on the following questions:

1. Explain the concept of object-oriented programming in simple terms to a complete beginner. 
    - Aspect Tested: **Factual accuracy, clarity, and compliance**. Can the LLM break down a complex topic into a beginner-friendly explanation?
    - Performance: 8/10. The response included a nice example to help reason about OOP and didn't dive too deep, omitting topics like inheritance. It was also structured nicely and broken into manageable pieces. The formatting didn't look great, but this was the fault of the application's front end; the LLM response included good markdown. There was room for improvement in terms of incorporating the _why_ into the analogy. Why do we use OOP when, at the surface, it adds complexity? It would also have benefited from an elementary code example, even though a language wasn't specified. Showing a simple class where we create new animals with attributes would be tangible to a beginner.

2. Read the following paragraph and provide a concise summary of the key points…
    - Aspect Tested: **Summarization, information extrapolation, and conciseness.** Can the LLM extrapolate key points from a dense paragraph and create a short summary.
    - Performance: 7/10. The summarization was very accurate, but would have been improved with (1) A single sentence at the beginning to summarize what the paragraph is about in a nutshell before a more detailed summary and (2) bolding or bullet points to emphasize the "key points" that the user has asked for.
 
3. Write a short, imaginative story (100–150 words) about a robot finding friendship in an unexpected place.
    - Aspect Tested: **Creativity, personality, and story-telling.** Is the story unique, fun, and complete?
    - Performance: 6/10. The story was cohesive, but quite basic, not following a typical narrative arc but rather filling in with extra descriptions. That said, the story was warm, descriptive, and playful. It assigned a male gender to a robot, which I'd probably try to avoid.

4. If a store sells apples in packs of 4 and oranges in packs of 3, how many packs of each do I need to buy to get exactly 12 apples and 9 oranges?
    - Aspect Tested: **Logic, maths, and chain of thought.** Can the LLM perform basic maths and complete these steps in the correct order to produce an accurate result?
    - Performance: 10/10. The answer was correct and the LLM walked through its reasoning to arrive at the answer, restating the problem, breaking down the information it was providing, showing the maths it performed, and providing a final answer.

5. Rewrite the following paragraph in a professional, formal tone…
    - Aspect Tested: **Tone, role-playing, and clarity.** Can the LLM take the key information and change the tone appropriately?
    - Performance: 5/10. The response was professional, but it was more of a vocabulary swap-out rather than truly rewriting the paragraph. This meant the result felt unatural. While it was professional in vocabulary, it wasn't professional in providing a clear call-to-action. It was beautified complaining.

This "vibe check" now serves as a baseline, of sorts, to help understand what holes your application has.

🧑‍🤝‍🧑❓ Discussion Question #1:

What are some limitations of vibe checking as an evaluation tool?

- A "vibe check" is totally at the whim of the person performing it. Similar to the "hungry judge effect," your mood and state of mind is going to affect how you percieve the vibe check to be.
- It's not re-producable. We can make changes to our application and complete another vibe check, but this is a very poor way to decide if you've improved performance. Given the non-deterministic nature of LLMs, along with confirmation bias, it's very hard to get a true sense of if you application has been improved with just a vibe check.
- It's incredibly slow compared to running more automated evals or utilizing LLM as judge techniques.
- It lacks diversity. It will be a vibe check of what **you** decide to ask the LLM. This is often not aligned with how a user base will actually be using a product.