// Structured profile content. Long-form projects and articles live in
// content/projects and content/posts as markdown. Wrap text in **double
// asterisks** to highlight a keyword.

export const profile = {
  name: "Sudhir Pol",
  title: "Machine Learning Engineer",
  focus: "LLM inference · Agentic AI · Model evaluation",
  image: "/assets/img/sudhir-pol.jpg",
  summary: [
    "I build and evaluate LLM systems, and I publish what the measurements say — including when they say my own design didn't help.",
    "At **Adobe** I built a multimodal LLM-as-judge framework that scored roughly 750 charts across ten quality dimensions and gated model releases; tuning the judge with DSPy raised its agreement with subject-matter experts from 0.79 to 0.92 F1. The hard part wasn't the model — it was getting the reviewers to agree with each other first, because that agreement is the ceiling on what any judge can reach. Before that, at **S&P Global**, I shipped a document-extraction workflow pairing a LoRA-fine-tuned Llama 2 with OpenSearch hybrid retrieval, evaluated in a comparison across 100 analysts.",
    "Two of my recent projects produced results against my own design. A [quantization-aware inference router](/projects/02-quantroute/) I built turned out to be unnecessary: on an A100, AWQ ran 56% cheaper than FP16 at statistically indistinguishable quality, so the optimal policy collapsed to always using the cheap lane. A [multi-agent generation pipeline](/projects/00-agent-systems/) I designed performed identically to a single output at eight times the API calls — the entire gain came from the reranker. Both are in the READMEs, along with which numbers came from real hardware and which from a simulated backend.",
    "Lately I've been working further down the stack: [articles](/writing/) deriving the mathematics behind GPTQ and AWQ, [two merged PRs](https://github.com/hao-ai-lab/FastVideo/pulls?q=is%3Apr+author%3Asudhirpol522) to hao-ai-lab's FastVideo, an [open PR](https://github.com/vllm-project/vllm/pull/53647) to vLLM, and hand-written CUDA FlashAttention kernels. That's the direction I want to keep going.",
    "Currently looking for ML engineering roles in LLM evaluation and inference.",
  ],
  links: [
    { label: "Email", href: "mailto:sudhirpol522@gmail.com", text: "sudhirpol522@gmail.com" },
    { label: "GitHub", href: "https://github.com/sudhirpol522" },
    { label: "LinkedIn", href: "https://www.linkedin.com/in/sudhir-pol-a6b544179" },
    { label: "Medium", href: "https://sudhirpol522.medium.com" },
  ],
};

export const interests = [
  "Reliable agent systems: explicit state, typed tool interfaces, human approval gates",
  "LLM inference efficiency: speculative decoding, Medusa, continuous batching, KV caching",
  "Model compression: AWQ and GPTQ quantization, precision-aware routing",
  "Evaluation: LLM-as-judge calibration against expert labels, regression gating",
];

export const openSource = [
  {
    project: "FastVideo",
    title: "OpenAI-compatible image endpoint and input validation",
    state: "Merged",
    prs: [
      { ref: "PR #1840", href: "https://github.com/hao-ai-lab/FastVideo/pull/1840" },
      { ref: "PR #1841", href: "https://github.com/hao-ai-lab/FastVideo/pull/1841" },
    ],
    summary:
      "FastVideo's image server did not answer at the address the **standard OpenAI client** calls, and it accepted unsupported output formats only to fail later. I added the **/v1/images/generations** endpoint and made the server reject bad formats up front with a clear **HTTP 400** error, with **regression tests** for both.",
  },
  {
    project: "vLLM",
    title: "Tie-aware speculative-decoding logprob test",
    state: "Open",
    prs: [{ ref: "PR #53647", href: "https://github.com/vllm-project/vllm/pull/53647" }],
    summary:
      "A vLLM test for **speculative decoding** was always skipped in CI because it asked for more GPU memory than the CI machines had, and when it did run, ties between equally likely tokens could make it fail at random. I lowered its requirement from **32 to 16 GiB** so it runs again, and made it **compare tokens by identity** so ties no longer cause false failures.",
  },
];

export const experience = [
  {
    company: "Adobe",
    role: "Machine Learning Engineer Intern",
    period: "2025",
    summary: "Worked on **LLM evaluation**, building a multimodal LLM-as-judge framework to benchmark Gemini, GPT, and Claude, and on **multi-agent systems** with LangGraph, MCP tools, and human approval.",
  },
  {
    company: "S&P Global",
    role: "Data Scientist II",
    period: "2022 to 2024",
    summary: "Worked on **LLMs** end to end: **fine-tuning** with **LoRA** and **SFT** (Llama 2, T5, LayoutLMv3), **serving** on SageMaker and Lambda, and **MLOps** with MLflow, DVC, and GitHub Actions.",
  },
  {
    company: "American Express",
    role: "Analyst, Data Science",
    period: "2021 to 2022",
    summary: "Built a real-time **XGBoost fraud detection** model for transaction authorization and **VIBE**, a BERT and XGBoost **sentiment analysis** model for customer call transcripts.",
  },
];

export const earlierProjects = [
  {
    title: "Bristol-Myers Squibb Molecular Translation",
    year: "2021",
    detail: "Image-to-InChI captioning with an EfficientNet encoder and LSTM decoder with Bahdanau attention; **Levenshtein distance 8.9**, top 500 on Kaggle.",
  },
  {
    title: "Italian to English Machine Translation",
    year: "2019",
    detail: "Attention-based LSTM sequence-to-sequence model comparing Bahdanau, dot, and general score attention; **BLEU 0.83** on the test set.",
  },
  {
    title: "Microsoft Malware Prediction",
    year: "2019",
    detail: "Malware family classification from ASM and byte files with calibrated Naive Bayes; **log loss 0.010**.",
  },
];


export const education = [
  {
    institution: "Indiana University Bloomington",
    degree: "Master of Science in Data Science",
    period: "Aug 2024 – May 2026",
    location: "Bloomington, IN",
  },
  {
    institution: "Rajarambapu Institute of Technology",
    degree: "Bachelor of Technology in Electronics and Telecommunications",
    period: "Aug 2017 – May 2021",
    location: "Sangli, India",
  },
];
