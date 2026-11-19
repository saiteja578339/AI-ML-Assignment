
import json

questions = [
    {
        "id": 1,
        "question": "What is the main purpose of the NIST AI Risk Management Framework?",
        "answer": "To help organizations manage risks associated with AI and promote trustworthy AI systems.",
        "keywords": ["risk management", "trustworthy", "organizations"]
    },
    {
        "id": 2,
        "question": "What are the characteristics of trustworthy AI?",
        "answer": "Trustworthy AI characteristics include validity and reliability, safety, security and resilience, accountability and transparency, explainability and interpretability, privacy enhancement, and fairness with harmful bias managed.",
        "keywords": ["trustworthy", "validity", "reliability", "fairness"]
    },
    {
        "id": 3,
        "question": "What are the four core functions of the AI RMF?",
        "answer": "Govern, Map, Measure, and Manage.",
        "keywords": ["govern", "map", "measure", "manage"]
    },
    {
        "id": 4,
        "question": "What is the purpose of the Govern function?",
        "answer": "Govern establishes organizational policies, processes, responsibilities, and a culture for managing AI risks.",
        "keywords": ["govern", "policies", "responsibilities"]
    },
    {
        "id": 5,
        "question": "What is the purpose of the Map function?",
        "answer": "Map establishes the context in which an AI system is intended to operate and identifies risks and impacts.",
        "keywords": ["map", "context", "impacts"]
    },
    {
        "id": 6,
        "question": "What is the purpose of the Measure function?",
        "answer": "Measure uses appropriate methods and metrics to analyze, assess, benchmark, and monitor AI risks.",
        "keywords": ["measure", "metrics", "analyze", "assess"]
    },
    {
        "id": 7,
        "question": "What is the purpose of the Manage function?",
        "answer": "Manage prioritizes and responds to identified AI risks and puts appropriate risk treatments into action.",
        "keywords": ["manage", "prioritize", "risk"]
    },
    {
        "id": 8,
        "question": "What does AI validity and reliability mean?",
        "answer": "An AI system performs consistently and correctly for its intended use under expected conditions.",
        "keywords": ["validity", "reliability", "intended use"]
    },
    {
        "id": 9,
        "question": "Why is transparency important in AI systems?",
        "answer": "Transparency helps people understand how an AI system is developed, operates, and makes or supports decisions.",
        "keywords": ["transparency", "understand", "decisions"]
    },
    {
        "id": 10,
        "question": "What is the role of accountability in trustworthy AI?",
        "answer": "Accountability establishes responsibility for AI system outcomes and supports appropriate oversight.",
        "keywords": ["accountability", "responsibility", "oversight"]
    },
    {
        "id": 11,
        "question": "How does the AI RMF address fairness?",
        "answer": "It encourages managing harmful bias and promoting fairness in AI systems.",
        "keywords": ["fairness", "bias", "harmful"]
    },
    {
        "id": 12,
        "question": "What is the purpose of the AI RMF Map function's context activities?",
        "answer": "They help establish the context, intended purposes, and potential impacts of an AI system.",
        "keywords": ["context", "intended", "impacts"]
    },
    {
        "id": 13,
        "question": "Why should AI risks be monitored after deployment?",
        "answer": "Monitoring helps identify changing risks, unexpected behavior, and impacts during operation.",
        "keywords": ["monitor", "deployment", "risks"]
    },
    {
        "id": 14,
        "question": "What does explainability and interpretability contribute to AI?",
        "answer": "They help people understand how and why an AI system produces its outputs.",
        "keywords": ["explainability", "interpretability", "outputs"]
    },
    {
        "id": 15,
        "question": "Who can use the NIST AI Risk Management Framework?",
        "answer": "Organizations and individuals involved in designing, developing, deploying, evaluating, or using AI systems can use the framework.",
        "keywords": ["organizations", "designing", "developing", "deploying"]
    }
]

with open("results/questions.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2, ensure_ascii=False)

print("Created 15 RAG evaluation questions.")
print("Saved to results/questions.json")
