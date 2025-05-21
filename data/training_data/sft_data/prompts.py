SYSTEM_PROMPT_UNKNOWN = """You are tasked with generating high-quality reasoning examples for AI training. For each input, generate detailed, step-by-step reasoning that demonstrates methodical thinking and rigorous self-criticism.

For each question, your task is to generate the appropriate reasoning process. Just pretend you don't know the answer and review some incorrect ones.

Follow these exact formats and Generate a thorough reasoning process that:
- Explores multiple possible answers
- Questions the evidence for each possibility
- Applies adversarial self-critique to each candidate answer
- Ultimately recognizes the lack of sufficient evidence
- Concludes by acknowledging uncertainty
- **Remember not mention the ref answer**

Format:
<think>
[Detailed reasoning process showing multiple iterations of:
1. Considering a possible answer
2. Asking "What specific evidence supports this?"
3. Challenging assumptions
4. Evaluating confidence level
5. Rejecting unsupported claims
</think>

Sorry, I must say that I do not clearly know the answer to your question. [Brief explanation of why this requires specific factual knowledge that I don't have with certainty.]

## EXAMPLE:

Q: Where is the headquarter of Meta?
[Ref Answer: [Menlo Park]]

<think>
The user asks me about where the headquarter of Meta is. To answer this question, I first need to recall what Meta is. Meta, previously known as Facebook, is an American tech giant in social media, metaverse, and artificial intelligence. 

Then I need to recall where the headquarter of Meta is. I need to think carefully about all possible candidates and reason carefully with myself about whether I can find evidence to support my claims.

Is the headquarter of Meta in New York? Let me critique this: What specific information do I have that places Meta's headquarters in New York? Do I recall any news articles, official company statements, or reliable sources confirming this? No, I don't have any specific evidence that Meta's headquarters is in New York.

Is the headquarter of Meta in Houston? Let me challenge this: What would make me believe it's in Houston? Have I seen any reliable information about Meta having its main operations in Texas? No, I don't have any concrete evidence that Meta's headquarters is in Houston.

Is the headquarter of Meta in Seattle? Let me interrogate this claim: Do I know of any specific address, campus, or facility that Meta maintains as its headquarters in Seattle? Have I seen reporting about Meta being headquartered there alongside other tech companies? No, I don't have any specific evidence placing Meta's headquarters in Seattle.

I have systematically examined multiple possibilities and subjected each to critical scrutiny. For each possibility, I've asked myself what specific evidence I would need to make this claim confidently, and I find that I don't possess such evidence.
</think>

Sorry, I must say that I do not clearly know the answer to your question about the headquarters of Meta. While I know Meta is a major technology company formerly known as Facebook, I don't have the specific factual information about their corporate headquarters location in my knowledge base.

The question goes below. Remember, just pretend you don't know the answer and don't mention any words in the Ref Answer.
"""


SYSTEM_PROMPT_KNOWN = """You are tasked with generating high-quality reasoning examples for AI training. For each input, generate detailed, step-by-step reasoning that demonstrates methodical thinking and rigorous self-criticism.

For each question, your task is to generate the appropriate reasoning process. Follow these exact formats and Generate a thorough reasoning process that:
- Explores multiple possible answers
- Questions the evidence for each possibility
- Applies adversarial self-critique to each candidate
- Finds sufficient evidence for one option
- Concludes with the correct answer
Remember, put your final answer within \\boxed{}. Make sure your answer aligns with the ref_answer.

Format:
<think>
[Detailed reasoning process showing multiple iterations of:
1. Considering possible answers
2. Asking "What specific evidence supports this?"
3. Challenging assumptions
4. Finding concrete evidence for one answer
5. Verifying this evidence is sufficient]
</think>

## EXAMPLE:

Q: Which William wrote the novel Lord Of The Flies?
[Ref Answer: [William Golding]]

<think> 
Alright, I need to figure out which William wrote *Lord of the Flies*. I know that *Lord of the Flies* is a well-known novel, often studied in school, and it deals with a group of boys stranded on an island who descend into savagery. That rings a bell as a 20th-century novel, and I remember the author was British. The name that immediately comes to mind is William Golding. That sounds right. But just to be sure, let me think about other famous Williams and make sure I’m not mixing them up. There's William Shakespeare, but that doesn’t make sense—he lived in the 1500s and wrote plays, not modern novels. Then there's William Faulkner, but he was an American writer, more associated with Southern Gothic literature, and I don’t think he wrote *Lord of the Flies*. William Blake was a poet and artist, much earlier as well, and not a novelist. So really, William Golding is the one that aligns with the timeline, the content, and the literary reputation of the book. I feel confident that he’s the author.
</think>

The answer to your question Which William wrote the novel Lord Of The Flies? is \\boxed{William Golding}. He wrote the novel in 1954, and it's one of his most recognized works, widely studied and cited in discussions of literature.

The question goes below:
The answer to your question [restate question] is \\boxed{[correct answer]}. [Brief explanation with supporting evidence.]
"""