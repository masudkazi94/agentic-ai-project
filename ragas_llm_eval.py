import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from llama_index.llms.groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Judge LLM Setup (Groq)
os.environ["GROQ_API_KEY"] = "your_groq_api_key"  # <-- USE YOUR NEW KEY
judge_llm = Groq(model="openai/gpt-oss-20b", temperature=0)
evaluator_llm = LlamaIndexLLMWrapper(llm=judge_llm)

# 2. Local Embeddings Setup
local_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
)

# 3. Data from your live app
data_samples = {
    'question': [
        'What is evolution?',
        'What are the key topics in evolution?',
        'What is genetic drift?'
    ],
    'answer': [
        'Evolution is the change in the heritable characteristics of biological populations over successive generations. It is driven by mechanisms such as natural selection, genetic drift, mutation, and gene flow, which together lead to the diversification of life forms. The theory, first articulated by Charles Darwin and Alfred Russel Wallace, explains how species adapt to their environments and gives rise to the vast variety of organisms observed today',
        'Natural Selection – the differential survival and reproduction of individuals due to differences in phenotype.Genetic Drift  random changes in allele frequencies, especially in small populations. Mutation. the ultimate source of genetic variation, providing new alleles. Gene Flow (Migration). the transfer of genetic material between populations.Speciation. the process by which new species arise, often through reproductive isolation.Adaptation. traits that increase fitness become more common in a population.Phylogenetics and Evolutionary History . reconstruction of relationships among species using morphological and molecular data.Fossil Record and Paleontology .evidence of past life forms and transitional stages.Molecular Evolution – changes at the DNA, RNA, and protein levels, including concepts like molecular clocks.Developmental Biology (Evo-devo) how changes in developmental processes drive evolutionary innovation. These topics are interrelated and together form the foundation of evolutionary biology, providing a comprehensive framework for understanding the diversity and complexity of life',
        'Genetic drift is a mechanism of evolution that refers to random fluctuations in the frequencies of alleles (gene variants) within a population from one generation to the next. Unlike natural selection, which is driven by differential reproductive success based on traits, genetic drift occurs purely by chance. It is most pronounced in small populations, where random events—such as which individuals happen to reproduce or survive—can significantly alter allele frequencies, potentially leading to the loss of genetic variation or the fixation of alleles that may be neutral or even deleterious. Over time, genetic drift can cause populations to diverge genetically, contributing to evolutionary change. Key aspects of genetic drift include: Random sampling effect (the alleles passed to the next generation are a random sample of those in the current generation), Population size dependence (the smaller the population, the larger the impact of drift), and Consequences (loss of genetic diversity, fixation or loss of alleles, and increased differentiation between isolated populations). Examples of genetic drift in nature include the "founder effect," where a new population is established by a few individuals, and the "bottleneck effect," where a population’s size is drastically reduced, both leading to reduced genetic variation. Overall, genetic drift is a fundamental evolutionary force that, alongside natural selection, mutation, and gene flow, shapes the genetic structure of populations over time.'
    ],
    'contexts': [
        ['Evolutionary Biology is the study of history of life forms on earth. Evolution refers to the changes in flora and fauna that have occurred over millions of years on earth. To understand these changes, one must understand the context of origin of life, i.e., evolution of earth, of stars and of the universe itself. The universe is almost 13.8 billion years old; earth was formed about 4.5 billion years back, and life appeared almost four billion years back. There has been gradual evolution of life forms, with extinctions of different life forms and new forms arising at different periods of earth history. Evolution is not a directed process in the sense of determinism; it is a stochastic process based on chance events in nature and chance mutation in organisms.'],
        ['Chapter 6 on Evolution covers the following key topics: 6.1 Origin of Life; 6.2 Evolution of Life Forms – A Theory; 6.3 What are the Evidences for Evolution?; 6.4 What is Adaptive Radiation?; 6.5 Biological Evolution; 6.6 Mechanism of Evolution; 6.7 Hardy-Weinberg Principle; 6.8 A Brief Account of Evolution; 6.9 Origin and Evolution of Man'],
        ['Five factors are known to affect Hardy-Weinberg equilibrium: gene migration or gene flow, genetic drift, mutation, genetic recombination and natural selection. When migration of a section of population to another place and population occurs, gene frequencies change in the original as well as in the new population. If the same change occurs by chance, it is called genetic drift. Sometimes the change in allele frequency is so different in the new sample of population that they become a different species; the original drifted population becomes founders and the effect is called founder effect.']
    ],
    'ground_truth': [
        'Evolution is the change in genetic composition of populations over generations.',
        'Topics include natural selection, genetic drift, mutation, and gene flow.',
        'Genetic drift is random changes in allele frequencies, especially in small populations.'
    ]
}

dataset = Dataset.from_dict(data_samples)

# 4. Configure to run one by one to avoid rate limits
run_config = RunConfig(
    max_workers=1,        # Process only 1 question at a time
    timeout=120,          # Wait up to 2 minutes per request
    max_retries=5,        # Retry if Groq fails
    max_wait=30           # Wait 30s between retries
)

# 5. Run Evaluation
print("Running RAGAS Evaluation (Sequential Mode)...")
result = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(strictness=1), 
        ContextPrecision()
    ],
    llm=evaluator_llm,
    embeddings=local_embeddings,
    run_config=run_config,             
    raise_exceptions=False
)

print("\n" + "="*30)
print("RAGAS SCORES:")
print("="*30)
print(result)