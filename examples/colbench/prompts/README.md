# Vendored SWEET-RL prompts

`human_simulator_code_prompt.txt` and `llm_agent_code_prompt.txt` are copied
**verbatim** from the official SWEET-RL / ColBench repository:

- Source: https://github.com/facebookresearch/sweet_rl/tree/main/prompts
- Paper: Zhou et al., *SWEET-RL* (2025), arXiv:2503.15478
- License: CC-BY-NC (see the SWEET-RL repo's `LICENSE.md`)

They are bundled here so `runner.py` reproduces the official interaction exactly
(same prompts, same `OUTPUT:` / `I WANT TO ANSWER:` parsing). Do not edit them —
`ColBenchRunner` appends the ppol persona block to the human-simulator prompt at
runtime rather than modifying the file.

The `html` (frontend design) prompts are intentionally not vendored: the design
track needs Selenium + a vision sim + CLIP and is out of scope for this example.
