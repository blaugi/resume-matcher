#import "@preview/basic-resume:0.2.9": *

#show link: underline

// Put your personal information here, replacing mine
#let name = "Matheus Melara Girardi"
#let location = "Curitiba - PR"
#let email = "matheus.melara.girardi@gmail.com"
#let github = "github.com/blaugi"
#let linkedin = "www.linkedin.com/in/matheus-melara-girardi"
#let phone = "(41) 99223-2216"
// #let personal-site = "stuxf.dev"

#show: resume.with(
  author: name,
  // All the lines below are optional.
  // For example, if you want to to hide your phone number:
  // feel free to comment those lines out and they will not show.
  location: location,
  email: email,
  github: github,
  linkedin: linkedin,
  phone: phone,
  //personal-site: personal-site,
  accent-color: "#26428b",
  font: "New Computer Modern",
  paper: "us-letter",
  author-position: left,
  personal-info-position: left,
)


== Objetivo

Com experiência em Python, Pandas e Matplotlib, adquirida em uma iniciação científica recentemente apresentada na IEEE ISGT Europe 2024. Focado em Ciência de Dados, estou em busca de oportunidades onde possa aplicar minhas habilidades em análise, manipulação e visualização de dados.

== Experiência Profissional

#work(
  title: "Estágio em Ciência de Dados",
  company: "BRF",
  dates: dates-helper(start-date: "Dez 2024", end-date: "Atual"),
)
- Desenvolvi e implementei soluções de automação de processos utilizando Python, LangChain e LLMs. Automatizei tarefas como o preenchimento de mais de 1.060 fichas/mês a partir de documentos e o processamento de cerca de 960 pendências/mês de atendimentos. Integrei as soluções com sistemas internos, incluindo DataBricks, Salesforce e SAP.
- Implementei e gerenciei pipelines com Azure Data Factory e Azure Machine Learning.
- Estruturei e implementei um template de projeto para a equipe, o que reduziu o tempo de setup de pipelines no Azure ML e otimizou a colaboração da equipe.

== Experiência Acadêmica

#work(
  title: "Iniciação Científica",
  company: "Pontifícia Universidade Católica do Paraná",
  dates: dates-helper(start-date: "Jun 2023", end-date: "Jun 2024"),
)
- Conduzi análise e tratamento de dados de uma fazenda eólica utilizando pandas.
- Comparei a performance de modelos clássicos (ARIMA, Holt-Winters) e de deep learning (N-BEATS) utilizando rigorosas métricas de avaliação.
- Os resultados desta pesquisa foram aceitos e apresentados na conferência internacional IEEE ISGT Europe 2024.

#work(
  title: "Liga de Inteligência Artificial",
  company: "Pontifícia Universidade Católica do Paraná",
  dates: dates-helper(start-date: "Nov 2024", end-date: "Atual"),
)
- A Liga de Inteligência Artificial, iniciativa do CISIA-PUCPR, é um grupo seleto de estudantes e professores com foco no desenvolvimento de soluções inovadoras utilizando Machine Learning.  

== Formação

#edu(
  dates: "2023 — 2026 (previsto)",
  degree: "Bacharelado em Ciência da Computação",
  institution: "Pontifícia Universidade Católica do Paraná",
  location: "Curitiba, PR",
)

== Habilidades

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    *Linguagens:* \
    Python, Rust, Java, SQL, C 
  ], 
  [
    *Bibliotecas & Ferramentas:* \
    Polars, PyTorch, LangChain, Azure ML, Azure Data Factory, Databricks, Linux
  ],
)

== Idiomas
- *Inglês:* Fluente (Cambridge C2 Proficiency - Grade A)