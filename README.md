# PaperPolish: AI-Driven Paper Revisions Based on Conference Trends

The number of papers submitted to top machine learning conferences is increasing dramatically each year, making it harder for authors to come up with novel ideas and stay competitive. To improve the chances of acceptance for a machine learning conference submission, we propose PaperPolish, a multi-agent large language model (LLM) designed to simulate peer review feedback and suggest actionable improvements for submitted papers. Leveraging scraped review data from ICLR 2024–2025 and a fine-tuned reviewer model based on Gemma-3, our system generates structured review outputs and revision recommendations. We evaluate the reviewer model’s alignment to ICLR-style feedback using BERTScore, and demonstrate how a multi-agent setup enhances the quality and coverage of suggested improvements. Our results highlight the potential for AI-driven systems to assist researchers in iteratively refining their submissions.


### Environment Details
A development Conda environment with all required dependencies can be created using the environment.yml file: `conda env create -f environment.yml`

