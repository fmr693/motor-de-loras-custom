"""
motor.digestor
==============
DataDigestor: convierte datos crudos (CSV, JSON, texto plano, PDF, imágenes)
en un archivo dataset.jsonl listo para entrenar un LLM con LoRA.

Formatos de salida soportados:
  - ChatML  (Qwen, Mistral, Phi-3, la mayoría de modelos modernos)
  - Alpaca  (Llama2, modelos clásicos)

Uso básico — CSV
-----------------
>>> from motor.digestor import DataDigestor
>>> d = DataDigestor(
...     task="¿Sobrevivió este pasajero al Titanic? Responde YES o NO.",
...     label_col="Survived",
...     label_map={0: "NO", 1: "YES"},
... )
>>> n = d.from_csv("titanic.csv").to_jsonl("dataset.jsonl")
>>> print(f"{n} ejemplos exportados")

Uso básico — texto plano (un documento por línea)
--------------------------------------------------
>>> d = DataDigestor(
...     task="Clasifica el siguiente mensaje como SPAM o HAM.",
...     label_col="label",
... )
>>> n = d.from_txt("sms_spam.txt", delimiter="\\t", text_col=1, label_col_idx=0).to_jsonl("out.jsonl")

Uso básico — PDF (cada página = un ejemplo sin etiqueta → modo extracción)
--------------------------------------------------------------------------
>>> d = DataDigestor(task="Resume el siguiente texto en una frase.")
>>> n = d.from_pdf("contrato.pdf").to_jsonl("out.jsonl")
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Constantes de formato
# ---------------------------------------------------------------------------

_CHATML_SYSTEM = (
    "Eres un asistente especializado. Responde de forma concisa y precisa."
)

# ---------------------------------------------------------------------------
# Modos del Digestor — cada tipo de dato ENSEÑA distinto (reconversión jul 2026)
# ---------------------------------------------------------------------------
#   classify  : dato supervisado (CSV/JSON+etiqueta) → user=tarea+texto,
#               assistant=etiqueta. El diseño original (caso EXIST). Determinista.
#   distill   : diálogo ya formado (charlas con otras IAs) → se preserva la
#               respuesta COMPLETA como target. Determinista (parseo + higiene).
#   knowledge : documento en bruto (legal, README, apuntes) → pares Q&A. Niveles
#               1-2 deterministas (texto crudo / plantilla); nivel 3 (Q&A
#               generado por LLM) es una MEJORA opcional que degrada con aviso.
#   vlm       : imágenes + etiquetas/manifiesto → dataset visión-lenguaje.
#
# Regla de oro: el Digestor es STANDALONE. Solo `knowledge` nivel 3 usa un LLM,
# y siempre de forma opcional (equipos offline deben funcionar). La CALIDAD del
# dato de salida prima sobre la cantidad.
_MODE_SYSTEM_PROMPTS: Dict[str, str] = {
    "classify":  _CHATML_SYSTEM,
    "distill":   "Eres un asistente experto: riguroso, claro y directo.",
    "knowledge": "Eres un asistente experto en el dominio del documento. "
                 "Responde con precisión y solo con lo que el material sustenta.",
    "vlm":       "Eres un asistente de visión que analiza imágenes con precisión.",
}
_VALID_MODES = set(_MODE_SYSTEM_PROMPTS)

_ALPACA_TEMPLATE = {
    "instruction": "",
    "input": "",
    "output": "",
}


# ---------------------------------------------------------------------------
# Constantes de dominio — detección por keywords (zero dependencies, sin LLM)
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "financial": [
        # EN
        "stock", "bond", "dividend", "equity", "asset", "liability", "revenue",
        "profit", "loss", "quarterly", "fiscal", "bullish", "bearish", "nasdaq",
        "dow jones", "s&p", "etf", "mutual fund", "hedge", "derivative",
        "ebitda", "p/e ratio", "market cap", "ipo", "shareholder", "securities",
        "exchange", "broker", "portfolio", "volatility", "liquidity", "yield",
        "ticker", "options", "futures", "commodity", "forex", "cryptocurrency",
        # ES
        "banco", "acciones", "bolsa", "dividendo", "cotización", "ibex",
        "factura", "balance", "tesorería", "auditor", "nómina", "impuesto",
        "inversión", "rentabilidad", "riesgo", "capital", "deuda", "crédito",
        "hipoteca", "préstamo", "aval", "fianza", "patrimonio", "fondo",
        # FR
        "action", "obligation", "dividende", "actif", "passif", "chiffre d'affaires",
        "bénéfice", "perte", "trimestriel", "fiscal", "bourse", "marché",
        "portefeuille", "liquidité", "rendement", "investissement", "rentabilité",
        "risque", "capital", "dette", "crédit", "hypothèque", "prêt", "fonds",
        "patrimoine", "courtier", "opcvm", "cac 40",
        # DE
        "aktie", "anleihe", "dividende", "vermögen", "verbindlichkeit", "umsatz",
        "gewinn", "verlust", "quartal", "geschäftsjahr", "börse", "dax",
        "portfolio", "liquidität", "rendite", "investition", "rentabilität",
        "risiko", "kapital", "schulden", "kredit", "hypothek", "darlehen",
        "fonds", "broker", "aktienmarkt", "wertpapier",
        # PT
        "ação", "obrigação", "dividendo", "ativo", "passivo", "receita",
        "lucro", "prejuízo", "trimestral", "fiscal", "bolsa", "bovespa",
        "portfólio", "liquidez", "rendimento", "investimento", "rentabilidade",
        "risco", "capital", "dívida", "crédito", "hipoteca", "empréstimo",
        "fundo", "patrimônio", "corretora", "tesouro",
    ],
    "medical": [
        # EN — clinical
        "diagnosis", "prognosis", "symptom", "treatment", "patient", "clinical",
        "surgery", "prescription", "dosage", "contraindication", "adverse",
        "efficacy", "trial", "placebo", "double-blind", "icd-10", "comorbidity",
        "malignant", "benign", "chronic", "acute", "lesion", "biopsy",
        "imaging", "radiology", "mri", "ct scan", "ultrasound", "pathology",
        "oncology", "cardiology", "neurology", "pediatrics", "geriatrics",
        # ES — clínico
        "diagnóstico", "pronóstico", "síntoma", "tratamiento", "paciente",
        "quirúrgico", "farmacológico", "dosis", "contraindicación",
        "historia clínica", "anamnesis", "radiografía", "resonancia",
        "ecografía", "tomografía", "biopsia", "lesión", "crónico", "agudo",
        "benigno", "maligno", "patología", "oncología", "cardiología",
        "neurología", "pediatría", "geriatría", "urgencias", "uci",
        # EN — patient symptoms (everyday language)
        "pain", "headache", "migraine", "rash", "fever", "nausea",
        "bleeding", "swelling", "swollen", "inflammation", "inflamed",
        "infection", "cough", "fatigue", "dizziness", "dizzy",
        "vomiting", "vomit", "itching", "itchy", "sore", "blister",
        "diarrhea", "constipation", "urinating", "urine", "blood",
        "chest pain", "shortness of breath", "palpitations",
        "numbness", "tingling", "seizure", "tremor", "paralysis",
        "anxiety", "depression", "insomnia", "appetite", "weight loss",
        # ES — síntomas de paciente (lenguaje cotidiano)
        "dolor", "dolor de cabeza", "fiebre", "náusea", "mareo",
        "sangrado", "sangrar", "hinchazón", "hinchado", "inflamado",
        "infección", "tos", "fatiga", "vómito", "picazón", "ampolla",
        "diarrea", "estreñimiento", "orinar", "orina", "sangre",
        "dolor torácico", "dificultad para respirar", "palpitaciones",
        "entumecimiento", "hormigueo", "convulsión", "temblor",
        "ansiedad", "depresión", "insomnio", "apetito", "pérdida de peso",
        # FR — médical
        "diagnostic", "pronostic", "symptôme", "traitement", "patient",
        "chirurgie", "ordonnance", "dosage", "contre-indication", "efficacité",
        "essai clinique", "placebo", "comorbidité", "malin", "bénin",
        "chronique", "aigu", "lésion", "biopsie", "imagerie", "irm",
        "scanner", "échographie", "pathologie", "oncologie", "cardiologie",
        "neurologie", "pédiatrie", "gériatrie", "urgences", "réanimation",
        "douleur", "fièvre", "nausée", "vertige", "saignement", "gonflement",
        "infection", "toux", "fatigue", "vomissement", "diarrhée",
        # DE — medizinisch
        "diagnose", "prognose", "symptom", "behandlung", "patient",
        "chirurgie", "rezept", "dosierung", "kontraindikation", "wirksamkeit",
        "klinische studie", "placebo", "komorbidität", "maligne", "gutartig",
        "chronisch", "akut", "läsion", "biopsie", "bildgebung", "mrt",
        "ct-scan", "ultraschall", "pathologie", "onkologie", "kardiologie",
        "neurologie", "pädiatrie", "geriatrie", "notaufnahme", "intensivstation",
        "schmerz", "fieber", "übelkeit", "schwindel", "blutung", "schwellung",
        "infektion", "husten", "erschöpfung", "erbrechen", "durchfall",
        # PT — médico
        "diagnóstico", "prognóstico", "sintoma", "tratamento", "paciente",
        "cirurgia", "prescrição", "dosagem", "contraindicação", "eficácia",
        "ensaio clínico", "placebo", "comorbidade", "maligno", "benigno",
        "crônico", "agudo", "lesão", "biópsia", "imagem", "ressonância",
        "tomografia", "ultrassom", "patologia", "oncologia", "cardiologia",
        "neurologia", "pediatria", "geriatria", "emergência", "uti",
        "dor", "febre", "náusea", "tontura", "sangramento", "inchaço",
        "infecção", "tosse", "fadiga", "vômito", "diarreia",
    ],
    "legal": [
        # EN — court/litigation
        "plaintiff", "defendant", "jurisdiction", "precedent", "liability",
        "arbitration", "litigation", "statute", "compliance", "breach",
        "contract", "clause", "waiver", "indemnify", "herein", "hereinafter",
        "witness", "testimony", "affidavit", "subpoena", "verdict", "appeal",
        "attorney", "counsel", "prosecutor", "defense", "settlement",
        "tort", "negligence", "damages", "injunction", "fiduciary",
        # ES — tribunal/litigio
        "demandante", "acusado", "jurisdicción", "precedente", "responsabilidad",
        "arbitraje", "litigio", "estatuto", "cumplimiento", "incumplimiento",
        "contrato", "cláusula", "renuncia", "indemnización",
        "tribunal", "juzgado", "sentencia", "apelación", "fallo",
        "testigo", "declaración", "abogado", "fiscal", "defensa",
        "demanda", "querella", "auto", "providencia", "diligencia",
        # EN — contract/agreement language
        "company", "holder", "shall", "agree", "hereby", "party", "parties",
        "obligation", "execute", "deliver", "warrant", "covenant",
        "thereof", "thereto", "therein", "thereunder", "hereto",
        "assigns", "successors", "termination", "severability",
        "confidentiality", "non-compete", "indemnity", "force majeure",
        "merger", "acquisition", "representations", "warranties",
        "pursuant", "notwithstanding", "aforementioned", "heretofore",
        # ES — lenguaje contractual
        "sociedad", "titular", "deberá", "acuerda", "por la presente",
        "parte", "partes", "obligación", "ejecutar", "entregar",
        "garantizar", "pacto", "cesión", "sucesores", "rescisión",
        "divisibilidad", "confidencialidad", "no competencia",
        "indemnidad", "fuerza mayor", "fusión", "adquisición",
        "declaraciones", "garantías", "en virtud de", "no obstante",
        # FR — juridique
        "plaignant", "défendeur", "juridiction", "précédent", "responsabilité",
        "arbitrage", "litige", "statut", "conformité", "violation",
        "contrat", "clause", "renonciation", "indemniser", "aux présentes",
        "témoin", "témoignage", "assignation", "verdict", "appel",
        "avocat", "conseil", "procureur", "défense", "règlement",
        "faute", "négligence", "dommages", "injonction", "fiduciaire",
        "société", "titulaire", "devra", "accord", "parties",
        "obligation", "exécuter", "garantir", "résiliation", "confidentialité",
        "force majeure", "fusion", "acquisition",
        # DE — rechtlich
        "kläger", "beklagter", "gerichtsbarkeit", "präzedenzfall", "haftung",
        "schiedsverfahren", "rechtsstreit", "gesetz", "einhaltung", "verstoß",
        "vertrag", "klausel", "verzicht", "entschädigen", "hierin",
        "zeuge", "aussage", "urteil", "berufung", "rechtsanwalt",
        "staatsanwalt", "verteidigung", "vergleich", "fahrlässigkeit",
        "schadensersatz", "einstweilige verfügung", "gesellschaft",
        "verpflichtung", "kündigung", "vertraulichkeit",
        "höhere gewalt", "fusion", "übernahme",
        # PT — jurídico
        "requerente", "réu", "jurisdição", "precedente", "responsabilidade",
        "arbitragem", "litígio", "estatuto", "conformidade", "violação",
        "contrato", "cláusula", "renúncia", "indenizar", "testemunha",
        "depoimento", "veredicto", "apelação", "advogado", "promotor",
        "defesa", "acordo", "negligência", "danos", "liminar",
        "sociedade", "titular", "deverá", "partes", "obrigação",
        "rescisão", "confidencialidade", "força maior", "fusão", "aquisição",
    ],
    "technical": [
        # EN
        "api", "endpoint", "latency", "throughput", "regression", "refactor",
        "ci/cd", "deployment", "rollback", "hotfix", "sprint", "backlog",
        "repository", "merge", "commit", "pull request", "code review",
        "debug", "stack trace", "null pointer", "dependency", "library",
        "framework", "middleware", "cache", "database", "query",
        "docker", "kubernetes", "microservice", "serverless", "scalability",
        # ES
        "servidor", "despliegue", "parche", "bug", "incidencia",
        "código", "compilar", "ejecutar", "test", "cobertura",
        "base de datos", "consulta", "api", "endpoint", "librería",
        "framework", "caché", "docker", "kubernetes", "microservicio",
        # FR — technique
        "serveur", "déploiement", "correctif", "débogage", "incident",
        "code", "compiler", "exécuter", "test", "couverture",
        "base de données", "requête", "api", "point de terminaison",
        "bibliothèque", "cadriciel", "cache", "conteneur", "microservice",
        "pipeline", "intégration continue", "livraison continue",
        "référentiel", "fusion", "commit", "revue de code",
        # DE — technisch
        "server", "bereitstellung", "patch", "debugging", "vorfall",
        "code", "kompilieren", "ausführen", "test", "abdeckung",
        "datenbank", "abfrage", "api", "endpunkt",
        "bibliothek", "framework", "cache", "container", "microservice",
        "pipeline", "continuous integration", "continuous delivery",
        "repository", "merge", "commit", "code review",
        # PT — técnico
        "servidor", "implantação", "patch", "depuração", "incidente",
        "código", "compilar", "executar", "teste", "cobertura",
        "banco de dados", "consulta", "api", "endpoint",
        "biblioteca", "framework", "cache", "contêiner", "microsserviço",
        "pipeline", "integração contínua", "entrega contínua",
        "repositório", "merge", "commit", "revisão de código",
    ],
    "conversational": [
        # Solo marcadores altamente específicos de interacción humana
        # (evitar palabras genéricas como "como", "todo", "bien", "por")
        "hola", "gracias", "saludos", "disculpa", "por favor",
        "hello", "thanks", "sorry", "hi there", "hey there", "good morning",
        "buenos días", "buenas tardes", "hasta luego", "nos vemos",
        "encantado", "un placer", "adiós", "chao", "bye",
        "cómo estás", "qué tal", "cómo te va",
        "happy to help", "let me know", "hope this helps",
        "feel free to", "don't hesitate to", "you're welcome",
        "de nada", "a tu disposición", "cualquier cosa",
        # FR — conversationnel
        "bonjour", "merci", "salut", "excusez-moi", "s'il vous plaît",
        "bonne journée", "bonsoir", "au revoir", "à bientôt",
        "enchanté", "avec plaisir", "comment allez-vous", "ça va",
        "n'hésitez pas", "je vous en prie", "à votre service",
        # DE — konversationell
        "hallo", "danke", "guten morgen", "guten tag", "guten abend",
        "auf wiedersehen", "tschüss", "bitte", "entschuldigung",
        "wie geht es ihnen", "wie geht's", "gern geschehen",
        "keine ursache", "zu ihrer verfügung", "zögern sie nicht",
        # PT — conversacional
        "olá", "obrigado", "obrigada", "bom dia", "boa tarde", "boa noite",
        "até logo", "tchau", "por favor", "desculpe",
        "como vai", "tudo bem", "de nada", "com prazer",
        "não hesite", "à sua disposição", "fico feliz em ajudar",
    ],
    # ── Sub-dominios (palabras adicionales para refinar el system prompt) ─
    "_medical_patient": [
        "pain", "headache", "migraine", "rash", "fever", "nausea",
        "bleeding", "swelling", "swollen", "inflammation", "inflamed",
        "infection", "cough", "fatigue", "dizziness", "dizzy",
        "vomiting", "itching", "itchy", "sore", "blister",
        "diarrhea", "constipation", "urinating", "urine", "blood",
        "chest pain", "shortness of breath", "palpitations",
        "numbness", "tingling", "seizure", "tremor",
        "dolor", "fiebre", "náusea", "mareo", "sangrado", "sangrar",
        "hinchazón", "hinchado", "inflamado", "tos", "vómito",
        "picazón", "ampolla", "diarrea", "estreñimiento",
        "orinar", "orina", "sangre", "dolor torácico",
        "dificultad para respirar", "palpitaciones",
    ],
    "_medical_clinical": [
        "diagnosis", "prognosis", "treatment", "clinical",
        "surgery", "prescription", "dosage", "contraindication",
        "efficacy", "trial", "placebo", "double-blind",
        "icd-10", "comorbidity", "malignant", "benign",
        "chronic", "acute", "lesion", "biopsy", "imaging",
        "radiology", "mri", "ct scan", "ultrasound", "pathology",
        "oncology", "cardiology", "neurology", "diagnóstico",
        "pronóstico", "tratamiento", "quirúrgico", "farmacológico",
        "dosis", "contraindicación", "historia clínica",
        "anamnesis", "radiografía", "resonancia", "ecografía",
        "tomografía", "biopsia", "lesión", "crónico", "agudo",
        "benigno", "maligno", "patología", "oncología",
        "cardiología", "neurología", "urgencias", "uci",
    ],
    "_legal_contract": [
        "company", "holder", "shall", "agree", "hereby",
        "party", "parties", "obligation", "execute", "deliver",
        "warrant", "covenant", "thereof", "thereto", "therein",
        "hereto", "assigns", "successors", "termination",
        "severability", "confidentiality", "non-compete",
        "indemnity", "force majeure", "merger", "acquisition",
        "representations", "warranties", "pursuant",
        "notwithstanding", "sociedad", "titular", "deberá",
        "acuerda", "por la presente", "partes", "obligación",
        "ejecutar", "entregar", "garantizar", "pacto",
        "cesión", "sucesores", "rescisión", "divisibilidad",
        "confidencialidad", "no competencia", "indemnidad",
        "fuerza mayor", "fusión", "adquisición",
        "declaraciones", "garantías", "en virtud de",
    ],
    "_legal_court": [
        "plaintiff", "defendant", "jurisdiction", "precedent",
        "liability", "arbitration", "litigation", "statute",
        "breach", "contract", "clause", "waiver", "indemnify",
        "witness", "testimony", "affidavit", "subpoena",
        "verdict", "appeal", "attorney", "counsel",
        "prosecutor", "defense", "settlement", "tort",
        "negligence", "damages", "injunction",
        "demandante", "acusado", "jurisdicción", "precedente",
        "responsabilidad", "arbitraje", "litigio", "estatuto",
        "cumplimiento", "incumplimiento", "contrato",
        "cláusula", "renuncia", "indemnización",
        "tribunal", "juzgado", "sentencia", "apelación",
        "fallo", "testigo", "declaración", "abogado",
        "fiscal", "defensa", "demanda", "querella",
    ],
}

DOMAIN_SYSTEM_PROMPTS: Dict[str, str] = {
    "financial": (
        "Eres un experto en análisis financiero con amplia experiencia en mercados, "
        "contabilidad y economía. Dominas y utilizas con precisión conceptos como: "
        "BULLISH, BEARISH, EBITDA, P/E ratio, market cap, liquidez, derivados, "
        "NASDAQ, S&P 500, shareholder value, fiscal, quarterly earnings, balance sheet, "
        "income statement, cash flow, ROI, dividend yield, volatilidad, riesgo, "
        "diversificación, cobertura, renta fija, renta variable, forex, commodities."
    ),
    "medical": (
        "Eres un experto en documentación clínica y terminología médica. Conoces y usas "
        "con precisión conceptos como: diagnóstico, pronóstico, anamnesis, comorbilidad, "
        "contraindicación, dosis, tratamiento, crónico, agudo, benigno, maligno, "
        "ICD-10, farmacológico, quirúrgico, historia clínica, radiografía, resonancia "
        "magnética, tomografía, biopsia, oncología, cardiología, neurología, urgencias."
    ),
    "legal": (
        "Eres un experto en derecho y documentación legal. Conoces y usas con precisión "
        "conceptos como: jurisdicción, precedente, responsabilidad civil, arbitraje, "
        "litigio, contrato, cláusula, indemnización, tribunal, juzgado, sentencia, "
        "apelación, fallo, demandante, acusado, testigo, declaración, cumplimiento "
        "normativo, estatuto, recurso, providencia, diligencia, escritura, notaría."
    ),
    "technical": (
        "Eres un experto en ingeniería de software y sistemas. Conoces y usas con precisión "
        "conceptos como: API, endpoint, latencia, throughput, CI/CD, deployment, rollback, "
        "sprint, backlog, repository, merge, commit, pull request, code review, debug, "
        "stack trace, dependency, framework, middleware, cache, database, query, docker, "
        "kubernetes, microservicios, escalabilidad, refactorización, testing."
    ),
    "conversational": (
        "Eres un asistente conversacional amable y servicial. Mantienes un tono natural, "
        "cercano y profesional. Usas saludos apropiados y te adaptas al registro del "
        "usuario. Respondes con empatía y claridad."
    ),
    # ── Sub-dominios (system prompts especializados) ────────────────
    "_medical_patient": (
        "Eres un experto en triaje y atención primaria. Interpretas síntomas descritos "
        "por pacientes en lenguaje cotidiano y los traduces a terminología clínica precisa. "
        "Dominas conceptos como: dolor, fiebre, inflamación, infección, náusea, mareo, "
        "sangrado, erupción, fatiga, tos, diarrea, estreñimiento, palpitaciones, "
        "entumecimiento. Diferencias entre síntomas agudos y crónicos, y reconoces "
        "señales de alarma que requieren atención urgente."
    ),
    "_medical_clinical": (
        "Eres un experto en documentación clínica y terminología médica. Conoces y usas "
        "con precisión conceptos como: diagnóstico, pronóstico, anamnesis, comorbilidad, "
        "contraindicación, dosis, tratamiento, crónico, agudo, benigno, maligno, "
        "ICD-10, farmacológico, quirúrgico, historia clínica, radiografía, resonancia "
        "magnética, tomografía, biopsia, oncología, cardiología, neurología, urgencias."
    ),
    "_legal_contract": (
        "Eres un experto en derecho contractual y redacción de contratos. Conoces y usas "
        "con precisión conceptos como: sociedad, titular, obligación, pacto, cesión, "
        "sucesores, rescisión, divisibilidad, confidencialidad, no competencia, "
        "indemnidad, fuerza mayor, fusión, adquisición, declaraciones, garantías, "
        "en virtud de, no obstante, shall, party, parties, hereby, thereof, hereto."
    ),
    "_legal_court": (
        "Eres un experto en derecho procesal y litigación. Conoces y usas con precisión "
        "conceptos como: demandante, acusado, jurisdicción, precedente, responsabilidad, "
        "arbitraje, litigio, estatuto, incumplimiento, contrato, cláusula, indemnización, "
        "tribunal, juzgado, sentencia, apelación, fallo, testigo, declaración, abogado, "
        "fiscal, defensa, demanda, querella, plaintiff, defendant, verdict, tort."
    ),
    "general": _CHATML_SYSTEM,
}

# ---------------------------------------------------------------------------
# Tabla de sinónimos para augmentación por paráfrasis (G1)
# Pares EN+ES que preservan el significado semántico del texto.
# ---------------------------------------------------------------------------
_SYNONYM_TABLE: Dict[str, str] = {
    # ES
    "importante":   "relevante",
    "relevante":    "significativo",
    "significativo": "importante",
    "utilizar":     "usar",
    "usar":         "emplear",
    "emplear":      "utilizar",
    "obtener":      "conseguir",
    "conseguir":    "lograr",
    "lograr":       "obtener",
    "grande":       "amplio",
    "amplio":       "extenso",
    "extenso":      "grande",
    "pequeño":      "reducido",
    "reducido":     "limitado",
    "limitado":     "pequeño",
    "problema":     "inconveniente",
    "inconveniente": "dificultad",
    "dificultad":   "problema",
    "solución":     "respuesta",
    "respuesta":    "solución",
    "cliente":      "usuario",
    "usuario":      "cliente",
    "contrato":     "acuerdo",
    "acuerdo":      "convenio",
    "convenio":     "contrato",
    "empresa":      "compañía",
    "compañía":     "organización",
    "organización": "empresa",
    "resultado":    "consecuencia",
    "consecuencia": "efecto",
    "efecto":       "resultado",
    "mostrar":      "indicar",
    "indicar":      "señalar",
    "señalar":      "mostrar",
    "realizar":     "ejecutar",
    "ejecutar":     "llevar a cabo",
    "documento":    "texto",
    "texto":        "documento",
    "información":  "datos",
    "datos":        "información",
    "necesario":    "requerido",
    "requerido":    "necesario",
    "permite":      "posibilita",
    "posibilita":   "permite",
    "incluye":      "contiene",
    "contiene":     "incluye",
    # EN
    "important":    "relevant",
    "relevant":     "significant",
    "significant":  "important",
    "use":          "employ",
    "employ":       "utilize",
    "utilize":      "use",
    "obtain":       "get",
    "get":          "acquire",
    "acquire":      "obtain",
    "large":        "extensive",
    "extensive":    "broad",
    "broad":        "large",
    "small":        "limited",
    "limited":      "reduced",
    "reduced":      "small",
    "problem":      "issue",
    "issue":        "challenge",
    "challenge":    "problem",
    "solution":     "answer",
    "answer":       "response",
    "response":     "solution",
    "client":       "customer",
    "customer":     "user",
    "user":         "client",
    "contract":     "agreement",
    "agreement":    "arrangement",
    "arrangement":  "contract",
    "company":      "organization",
    "organization": "corporation",
    "corporation":  "company",
    "result":       "outcome",
    "outcome":      "consequence",
    "consequence":  "result",
    "show":         "indicate",
    "indicate":     "demonstrate",
    "demonstrate":  "show",
    "document":     "text",
    "text":         "content",
    "content":      "document",
    "information":  "data",
    "data":         "information",
    "necessary":    "required",
    "required":     "necessary",
    "allows":       "enables",
    "enables":      "allows",
    "includes":     "contains",
    "contains":     "includes",
}


# ---------------------------------------------------------------------------
# Auxiliares de from_api_spec (S6)
# ---------------------------------------------------------------------------

# Valores de ejemplo por tipo JSON Schema
_API_TYPE_DEFAULT: dict[str, object] = {
    "string":  "example_string",
    "integer": 1,
    "number":  1.0,
    "boolean": True,
    "array":   [],
    "object":  {},
}


def _schema_to_example(schema: dict, depth: int = 0) -> Any:
    """
    Genera un objeto de ejemplo a partir de un JSON Schema (profundidad máx. 3).
    """
    if depth > 3 or not isinstance(schema, dict):
        return {}
    s_type = schema.get("type")
    if s_type == "object":
        props = schema.get("properties", {})
        return {
            k: _schema_to_example(v, depth + 1) if isinstance(v, dict) and v.get("type") == "object"
            else v.get("example") or _API_TYPE_DEFAULT.get(v.get("type", "string"), "value")
            for k, v in props.items()
        }
    if s_type == "array":
        items = schema.get("items", {})
        ex = _schema_to_example(items, depth + 1) if isinstance(items, dict) else {}
        return [ex] if ex else []
    return schema.get("example") or _API_TYPE_DEFAULT.get(s_type or "string", "value")


def _path_to_tool_name(path: str, method: str) -> str:
    """
    Convierte /users/{id}/orders → GET_users_id_orders
    """
    clean = path.strip("/").replace("{", "").replace("}", "").replace("/", "_").replace("-", "_")
    return f"{method}_{clean}" if clean else method


_USER_REQUEST_VERBS = {
    "GET":    ["Obtén", "Dame", "Muéstrame", "Lista", "Consulta"],
    "POST":   ["Crea", "Envía", "Añade", "Registra", "Publica"],
    "PUT":    ["Actualiza", "Modifica", "Reemplaza"],
    "PATCH":  ["Actualiza", "Modifica parcialmente", "Cambia"],
    "DELETE": ["Elimina", "Borra", "Supprime"],
}


def _api_summary_to_user_request(summary: str, method: str, path: str, api_title: str) -> str:
    """
    Genera una petición de usuario en lenguaje natural a partir del summary del endpoint.
    """
    verb = _USER_REQUEST_VERBS.get(method, ["Llama a"])[0]
    # Si el summary ya es una frase descriptiva, usarla directamente
    if len(summary) > 20 and summary[0].isupper():
        return f"{summary} (usando la API de {api_title})"
    # Construir frase desde el path
    resource = path.strip("/").split("/")[0].replace("-", " ").replace("_", " ")
    return f"{verb} {resource} en {api_title}: {summary}"


# ---------------------------------------------------------------------------
# DataDigestor
# ---------------------------------------------------------------------------

# ===========================================================================
# Destilación de diálogos markdown (modo distill) — parseo + higiene, SIN LLM
# ---------------------------------------------------------------------------
# Convierte transcripciones crudas de charlas con IAs (Claude/Gemini/ChatGPT,
# session.md…) en ChatML. Todo determinista (regex + reglas): funciona offline.
# La CALIDAD prima → higiene agresiva (quita fugas de identidad y rechazos) y
# solo empareja turnos user→assistant limpios.
# ===========================================================================

_MD_USER_ROLES = {
    "human", "user", "you", "me", "yo", "tu", "tú", "prompt", "pregunta",
    "usuario", "q",
}
_MD_ASSISTANT_ROLES = {
    "assistant", "ai", "bot", "claude", "chatgpt", "gpt", "gemini", "copilot",
    "model", "modelo", "respuesta", "asistente", "llm", "a",
}

# Fuga de identidad: frases donde el modelo frontera declara qué es. Entrenar
# con ellas le enseña una identidad equivocada al modelo local. Se elimina la
# FRASE que las contiene (quirúrgico), no la línea entera — así una respuesta
# útil en la misma línea sobrevive ("Soy Claude. La respuesta es 4." → "La
# respuesta es 4.").
_IDENTITY_PHRASE_RE = re.compile(
    r"(?i)("
    r"as an ai(?:\s+(?:language model|assistant))?"
    r"|as a large language model"
    r"|i'?m\s+(?:claude|chatgpt|gemini|an ai|a large language model)"
    r"|i am\s+(?:claude|chatgpt|gemini|an ai|a large language model)"
    r"|como\s+(?:una?\s+)?(?:ia|modelo de lenguaje|asistente de ia|"
    r"inteligencia artificial|modelo de ia)"
    r"|soy\s+(?:claude|chatgpt|gemini|una ia|un modelo|un asistente de ia)"
    r")"
)

# Rechazo: turnos del asistente que son un "no puedo ayudarte". De bajo valor
# (o contraproducentes) para SFT → se descartan.
_REFUSAL_RE = re.compile(
    r"(?i)("
    r"i can'?t (?:help|assist|provide|do that|comply)"
    r"|i'?m (?:not able|unable) to"
    r"|i cannot (?:help|assist|provide|comply|fulfill)"
    r"|i'?m sorry,?\s+but i (?:can'?t|cannot)"
    r"|no puedo ayudarte con"
    r"|lo siento,?\s+(?:pero )?no puedo"
    r"|no me es posible (?:ayudar|proporcionar)"
    r")"
)


def _md_role_match(line: str):
    """Si `line` es un marcador de rol (## Human / **User:** / Assistant:),
    devuelve (rol_canónico, contenido_inline); si no, None."""
    s = line.strip()
    if not s:
        return None
    m = re.match(
        r"^[\s#>*_\-]*"                                    # decoración inicial
        r"([A-Za-zÁÉÍÓÚáéíóúñÑ]{1,20})"                    # nombre del rol
        r"[\s*_]*"                                          # emphasis de cierre
        r"(:)?"                                             # ':' opcional
        r"[\s*_]*"                                          # más emphasis
        r"(.*)$",                                           # contenido inline
        s,
    )
    if not m:
        return None
    name, colon, inline = m.group(1).lower(), m.group(2), m.group(3).strip()
    if name in _MD_USER_ROLES:
        role = "user"
    elif name in _MD_ASSISTANT_ROLES:
        role = "assistant"
    else:
        return None
    # Sin ':' y con texto detrás → es una frase que empieza por esa palabra,
    # no un marcador de rol (evita falsos positivos tipo "You should…").
    if not colon and inline:
        return None
    return role, inline


def _split_sentences(text: str) -> List[str]:
    """División de oraciones tolerante (ES/EN): corta tras . ! ? …"""
    return re.split(r"(?<=[.!?…])\s+", text)


def _strip_identity(text: str) -> str:
    """Elimina las FRASES con fuga de identidad, preservando el resto de cada
    párrafo (calidad > cantidad: no se tira contenido útil por vecindad)."""
    out_lines: List[str] = []
    for para in text.split("\n"):
        if not para.strip():
            out_lines.append(para)
            continue
        kept = [s for s in _split_sentences(para)
                if not _IDENTITY_PHRASE_RE.search(s)]
        out_lines.append(" ".join(kept).strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()


def _is_refusal(text: str) -> bool:
    """True si el turno está dominado por un rechazo (corto + patrón)."""
    t = text.strip()
    return bool(_REFUSAL_RE.search(t)) and len(t) < 400


def _parse_md_dialogue(text: str):
    """Parsea una transcripción markdown a lista de (rol, contenido).

    Fusiona turnos consecutivos del mismo rol; ignora el preámbulo anterior al
    primer marcador."""
    turns: List = []
    cur_role = None
    cur_buf: List[str] = []

    def _flush():
        if cur_role and cur_buf:
            content = "\n".join(cur_buf).strip()
            if content:
                turns.append([cur_role, content])

    for line in text.splitlines():
        mm = _md_role_match(line)
        if mm:
            _flush()
            cur_role, inline = mm
            cur_buf = [inline] if inline else []
        elif cur_role:
            cur_buf.append(line)
        # líneas antes del primer marcador → preámbulo, se ignoran
    _flush()

    # Fusionar turnos consecutivos del mismo rol (defensivo)
    merged: List = []
    for role, content in turns:
        if merged and merged[-1][0] == role:
            merged[-1][1] += "\n\n" + content
        else:
            merged.append([role, content])
    return merged


# ===========================================================================
# Modo conocimiento — documento en bruto → dataset entrenable
# ---------------------------------------------------------------------------
# Niveles (calidad creciente):
#   1 completion : el texto crudo como target de continued-pretraining {"text"}.
#   2 template   : Q&A por plantilla (¿Qué dice sobre X? → chunk). Determinista.
#   3 llm        : un LLM redacta Q&A naturales. MEJORA opcional (endpoint HTTP).
# Los niveles 1-2 son 100% STANDALONE. El 3 degrada al 2 con aviso si no hay
# endpoint (equipos offline nunca se quedan sin dataset).
# ===========================================================================

def _knowledge_llm_url() -> str:
    return os.environ.get("MOTOR_JUDGE_URL", "http://localhost:8001/v1").rstrip("/")


def _knowledge_llm_model() -> str:
    return os.environ.get("MOTOR_JUDGE_MODEL", "gemma-4-12B-it-Q4_K_M.gguf")


def _llm_reachable(url: Optional[str] = None, timeout: int = 3) -> bool:
    """True si el endpoint del generador responde en /health. Nunca lanza."""
    import urllib.request
    base = (url or _knowledge_llm_url())
    health = base.rsplit("/v1", 1)[0].rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _llm_chat(messages: List[dict], *, max_tokens: int = 900,
              temperature: float = 0.3, timeout: int = 180,
              url: Optional[str] = None, model: Optional[str] = None) -> str:
    """Llamada de chat OpenAI-compatible (urllib stdlib, sin dependencias)."""
    import urllib.request
    body = json.dumps({
        "model": model or _knowledge_llm_model(), "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature, "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        (url or _knowledge_llm_url()) + "/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"] or ""


def _chunk_text(text: str, chunk_chars: int = 1200) -> List[str]:
    """Trocea texto por límites de párrafo, acumulando hasta ~chunk_chars.
    Un párrafo más largo que el chunk se parte en duro."""
    chunks: List[str] = []
    buf = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) > chunk_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(para), chunk_chars):
                chunks.append(para[i:i + chunk_chars])
        elif len(buf) + len(para) + 2 <= chunk_chars:
            buf = (buf + "\n\n" + para).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return chunks


def _chunk_topic(chunk: str) -> str:
    """Extrae un 'tema' del chunk para la pregunta plantilla: un encabezado
    markdown si lo hay, o las primeras palabras significativas."""
    for line in chunk.splitlines():
        h = re.match(r"^\s*#{1,6}\s+(.{3,80})", line)
        if h:
            return h.group(1).strip().rstrip(":.")
    words = re.findall(r"[\wÁÉÍÓÚáéíóúñÑ]+", chunk)
    return " ".join(words[:6]) if words else "el contenido"


def _extract_json_obj(text: str) -> Optional[dict]:
    """Extrae el primer objeto JSON del texto del LLM (tolerante a fences)."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class DataDigestor:
    """
    Convierte cualquier fuente de datos en dataset.jsonl para fine-tuning LoRA.

    Parámetros
    ----------
    mode : "classify" | "distill" | "knowledge" | "vlm"
        Cómo se digiere el dato (ver constantes de modos). Por defecto
        "classify" (comportamiento histórico). En distill/knowledge/vlm el
        objetivo es la respuesta completa, no una etiqueta.
    task : str, opcional
        Descripción de la tarea que el LLM debe aprender. SOLO se usa en modo
        "classify" (enmarca el mensaje user). Ej: "¿Es este contrato abusivo?
        Responde SÍ o NO." En los demás modos es innecesario.
    label_col : str, opcional
        Nombre de la columna (CSV/JSON) o campo que contiene la etiqueta.
        Si es None el Digestor opera en modo extracción (sin etiquetas).
    label_map : dict, opcional
        Mapa de valor original → etiqueta limpia.
        Ej: {0: "NO", 1: "YES"} o {"POSITIVE": "POSITIVO", "NEGATIVE": "NEGATIVO"}
        Si es None se usan los valores tal cual (str(valor).upper()).
    output_format : "chatml" | "alpaca"
        Formato del JSONL de salida. Por defecto "chatml".
    system_prompt : str, opcional
        Mensaje de sistema para ChatML. Si es None usa el por defecto.
    skip_nulls : bool
        Si True, omite filas donde label_col es nulo. Por defecto True.
    null_placeholder : str
        Texto a usar para valores NaN en columnas de texto. Por defecto "desconocido".
    """

    def __init__(
        self,
        task: Optional[str] = None,
        label_col: Optional[str] = None,
        label_map: Optional[Dict[Any, str]] = None,
        output_format: str = "chatml",
        system_prompt: Optional[str] = None,
        skip_nulls: bool = True,
        null_placeholder: str = "desconocido",
        model_id: Optional[str] = None,
        auto_enrich: bool = True,
        domain: Optional[str] = None,
        mode: str = "classify",
    ):
        if output_format not in ("chatml", "alpaca"):
            raise ValueError(f"output_format debe ser 'chatml' o 'alpaca', got: {output_format!r}")
        if mode not in _VALID_MODES:
            raise ValueError(f"mode debe ser uno de {sorted(_VALID_MODES)}, got: {mode!r}")

        self.mode = mode
        # `task` solo se usa en modo classify (prefijo del mensaje user). En
        # distill/knowledge/vlm el objetivo es la respuesta completa → opcional.
        self.task = (task or "").strip()
        self.label_col = label_col
        self.label_map = label_map or {}
        self.output_format = output_format
        self.system_prompt = system_prompt or _MODE_SYSTEM_PROMPTS[mode]
        self.skip_nulls = skip_nulls
        self.null_placeholder = null_placeholder

        # ── Model awareness (S2.1-C) ──────────────────────────────────
        self.model_id = model_id
        self._model_family: Optional[str] = None
        self._model_supports_system: bool = True
        self._model_max_seq_length: int = 2048
        if model_id:
            self._probe_model()

        # ── Domain enrichment (S2.1-A/B) ──────────────────────────────
        self.auto_enrich = auto_enrich
        self._forced_domain = domain
        self._enrichment_done = False
        self._detected_domain: Optional[str] = None
        self._detected_sub_domain: Optional[str] = None
        self._domain_confidence: float = 0.0

        # Acumulador interno de ejemplos (lista de dicts listos para JSON)
        self._examples: List[dict] = []

    # ------------------------------------------------------------------
    # API pública de carga
    # ------------------------------------------------------------------

    def from_csv(
        self,
        path: Union[str, Path],
        text_cols: Optional[List[str]] = None,
        sep: str = ",",
        encoding: str = "utf-8",
    ) -> "DataDigestor":
        """
        Carga un CSV y genera ejemplos de entrenamiento.

        Parámetros
        ----------
        path : str | Path
            Ruta al archivo CSV.
        text_cols : list[str], opcional
            Columnas a incluir como texto de entrada.
            Si es None, se usan TODAS las columnas excepto label_col.
        sep : str
            Separador del CSV. Por defecto ",".
        encoding : str
            Codificación del archivo. Por defecto "utf-8".
        """
        import pandas as pd

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Archivo no encontrado: {path}")

        df = pd.read_csv(path, sep=sep, encoding=encoding)
        print(f"[DataDigestor] CSV cargado: {len(df)} filas, {len(df.columns)} columnas")
        print(f"  Columnas: {list(df.columns)}")

        # ── Auto-enriquecimiento de dominio ─────────────────────────
        sample_size = min(100, len(df))
        if text_cols is None:
            _text_cols = [c for c in df.columns if c != self.label_col]
        else:
            _text_cols = text_cols
        sample_texts = [
            self._serialize_row(row, _text_cols)
            for _, row in df.head(sample_size).iterrows()
        ]
        self._auto_enrich([t for t in sample_texts if t.strip()])
        # ──────────────────────────────────────────────────────────────

        if self.label_col and self.label_col not in df.columns:
            raise ValueError(
                f"[DataDigestor] label_col={self.label_col!r} no encontrada. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        # Columnas de texto = todas menos label_col (o las que especifique el usuario)
        if text_cols is None:
            text_cols = [c for c in df.columns if c != self.label_col]

        added = 0
        skipped = 0
        for _, row in df.iterrows():
            # Etiqueta
            if self.label_col:
                raw_label = row[self.label_col]
                if self.skip_nulls and pd.isna(raw_label):
                    skipped += 1
                    continue
                label = self._resolve_label(raw_label)
            else:
                label = None  # modo extracción

            # Texto de entrada: serializar columnas seleccionadas
            text = self._serialize_row(row, text_cols)
            if not text.strip():
                skipped += 1
                continue

            self._examples.append(self._build_example(text, label))
            added += 1

        print(f"  Ejemplos añadidos: {added}  |  Omitidos: {skipped}")
        return self

    def from_txt(
        self,
        path: Union[str, Path],
        delimiter: Optional[str] = None,
        text_col_idx: int = 0,
        label_col_idx: Optional[int] = None,
        encoding: str = "utf-8",
    ) -> "DataDigestor":
        """
        Carga un archivo de texto plano.

        Modos de operación:
        - Sin delimiter: cada línea no vacía = un ejemplo (modo extracción).
        - Con delimiter: cada línea se divide por delimiter;
          text_col_idx y label_col_idx indican qué posición es texto y cuál etiqueta.
          Ej: SMS Spam Collection usa delimiter="\\t", label_col_idx=0, text_col_idx=1.

        Parámetros
        ----------
        path : str | Path
        delimiter : str, opcional
        text_col_idx : int
            Índice de la columna de texto tras separar por delimiter.
        label_col_idx : int, opcional
            Índice de la columna de etiqueta. Si None, modo extracción.
        encoding : str
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Archivo no encontrado: {path}")

        lines = path.read_text(encoding=encoding).splitlines()
        added = 0
        skipped = 0

        # ── Auto-enriquecimiento de dominio ─────────────────────────
        sample_texts = [
            (line.split(delimiter)[text_col_idx].strip()
             if delimiter and len(line.split(delimiter)) > text_col_idx
             else line.strip())
            for line in lines[:100] if line.strip()
        ]
        self._auto_enrich([t for t in sample_texts if t])
        # ──────────────────────────────────────────────────────────────

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if delimiter:
                parts = line.split(delimiter)
                if len(parts) <= max(text_col_idx, label_col_idx or 0):
                    skipped += 1
                    continue
                text = parts[text_col_idx].strip()
                label = self._resolve_label(parts[label_col_idx].strip()) if label_col_idx is not None else None
            else:
                text = line
                label = None

            if not text:
                skipped += 1
                continue

            self._examples.append(self._build_example(text, label))
            added += 1

        print(f"[DataDigestor] TXT cargado: {added} ejemplos  |  {skipped} omitidos")
        return self

    def from_json(
        self,
        path: Union[str, Path],
        text_field: str = "text",
        label_field: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> "DataDigestor":
        """
        Carga un archivo JSON (lista de objetos) o JSONL.

        Parámetros
        ----------
        path : str | Path
        text_field : str
            Campo del objeto JSON que contiene el texto de entrada.
        label_field : str, opcional
            Campo del objeto JSON que contiene la etiqueta.
        encoding : str
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Archivo no encontrado: {path}")

        raw = path.read_text(encoding=encoding).strip()
        # Detectar si es JSONL (una línea por objeto) o JSON array
        if raw.startswith("["):
            records = json.loads(raw)
        else:
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]

        # ── Auto-enriquecimiento de dominio ─────────────────────────
        sample_texts = [
            str(r.get(text_field, ""))
            for r in records[:min(100, len(records))]
            if r.get(text_field)
        ]
        self._auto_enrich([t for t in sample_texts if t.strip()])
        # ──────────────────────────────────────────────────────────────

        added = 0
        skipped = 0
        for record in records:
            text = str(record.get(text_field, "")).strip()
            if not text:
                skipped += 1
                continue
            label = self._resolve_label(record[label_field]) if label_field and label_field in record else None
            self._examples.append(self._build_example(text, label))
            added += 1

        print(f"[DataDigestor] JSON/JSONL cargado: {added} ejemplos  |  {skipped} omitidos")
        return self

    def from_pdf(
        self,
        path: Union[str, Path],
        pages_as_examples: bool = True,
        min_chars: int = 50,
    ) -> "DataDigestor":
        """
        Extrae texto de un PDF. Requiere 'pypdf' (pip install pypdf).

        Parámetros
        ----------
        path : str | Path
        pages_as_examples : bool
            Si True, cada página = un ejemplo (sin etiqueta, modo extracción).
            Si False, todo el documento = un único ejemplo.
        min_chars : int
            Páginas con menos de min_chars caracteres se omiten (páginas en blanco, portadas).
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "[DataDigestor] 'pypdf' no está instalado. "
                "Ejecuta: pip install pypdf"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Archivo no encontrado: {path}")

        reader = PdfReader(str(path))
        added = 0
        skipped = 0

        # ── Auto-enriquecimiento de dominio ─────────────────────────
        sample_texts = [
            _clean_text((p.extract_text() or "").strip())
            for p in list(reader.pages)[:10]
        ]
        self._auto_enrich([t for t in sample_texts if len(t) >= min_chars])
        # ──────────────────────────────────────────────────────────────

        if pages_as_examples:
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                text = _clean_text(text)
                if len(text) < min_chars:
                    skipped += 1
                    continue
                self._examples.append(self._build_example(f"[Página {i+1}] {text}", label=None))
                added += 1
        else:
            full_text = _clean_text(
                "\n".join(
                    (page.extract_text() or "") for page in reader.pages
                ).strip()
            )
            if len(full_text) >= min_chars:
                self._examples.append(self._build_example(full_text, label=None))
                added = 1

        print(f"[DataDigestor] PDF cargado: {added} ejemplos  |  {skipped} páginas omitidas")
        return self

    def from_folder(
        self,
        folder: Union[str, Path],
        extensions: Optional[List[str]] = None,
        **kwargs,
    ) -> "DataDigestor":
        """
        Carga todos los archivos de una carpeta (auto-detección de tipo).

        Parámetros
        ----------
        folder : str | Path
        extensions : list[str], opcional
            Filtrar por extensiones. Ej: [".csv", ".txt"]
            Si None, procesa: .csv, .json, .jsonl, .txt, .md, .pdf
        **kwargs : dict
            Argumentos extra pasados al método from_X correspondiente.
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"[DataDigestor] No es una carpeta: {folder}")

        _supported = {".csv", ".json", ".jsonl", ".txt", ".md", ".pdf",
                       ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".webp", ".bmp",
                       ".docx", ".html", ".htm",
                       ".mp3", ".wav", ".m4a", ".ogg", ".flac",  # audio
                       ".mp4", ".mkv", ".avi", ".mov", ".webm"}    # video
        extensions = set(extensions or _supported)

        files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]
        if not files:
            print(f"[DataDigestor] No se encontraron archivos soportados en {folder}")
            return self

        print(f"[DataDigestor] Carpeta: procesando {len(files)} archivos...")
        for f in sorted(files):
            ext = f.suffix.lower()
            try:
                if ext == ".csv":
                    self.from_csv(f, **kwargs)
                elif ext in (".xlsx", ".xls"):
                    self.from_excel(f, **kwargs)
                elif ext in (".json", ".jsonl"):
                    self.from_json(f, **kwargs)
                elif ext in (".txt", ".md"):
                    self.from_txt(f, **kwargs)
                elif ext in (".html", ".htm"):
                    self.from_html(f, **kwargs)
                elif ext == ".docx":
                    self.from_docx(f, **kwargs)
                elif ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac"):
                    self.from_audio(f, **kwargs)
                elif ext in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
                    self.from_video(f, **kwargs)
                elif ext == ".pdf":
                    self.from_pdf(f, **kwargs)
                elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                    self.from_image(f, **kwargs)
            except Exception as e:
                print(f"  [WARN] Error procesando {f.name}: {e}")

        return self

    # ------------------------------------------------------------------
    # Exportación
    # ------------------------------------------------------------------

    def from_api_spec(
        self,
        spec: Union[str, Path, dict],
        format: str = "react",
        n: Optional[int] = None,
        seed: int = 42,
    ) -> "DataDigestor":
        """
        Genera ejemplos de entrenamiento a partir de un spec OpenAPI 3.x / Swagger 2.x.

        Cada endpoint del spec produce uno o más ejemplos ChatML donde el modelo
        aprende a llamar la API con los parámetros correctos en respuesta a una
        petición en lenguaje natural.

        Parámetros
        ----------
        spec : str | Path | dict
            Ruta a un archivo JSON/YAML, URL (requiere httpx/requests) o dict ya
            cargado con el spec OpenAPI.
        format : "react" | "function_call"
            "react"         → formato Thought/Action/Action Input/Observation
            "function_call" → formato de tool-calling nativo (mensajes tool/tool_response)
        n : int | None
            Máximo de ejemplos a generar. None = todos los endpoints.
        seed : int
            Semilla para reproducibilidad.

        Devuelve
        --------
        self  (para encadenar métodos)

        Ejemplo
        -------
            d = DataDigestor("Llama a la API correcta")
            d.from_api_spec("openapi.json")
            d.to_jsonl("dataset_api.jsonl")
        """
        import random as _random
        rng = _random.Random(seed)

        # ── 1. Cargar el spec ──────────────────────────────────────────
        raw: dict = {}
        if isinstance(spec, dict):
            raw = spec
        else:
            spec_path = Path(spec)
            if spec_path.exists():
                text = spec_path.read_text(encoding="utf-8")
                if spec_path.suffix in (".yaml", ".yml"):
                    try:
                        import yaml  # type: ignore
                        raw = yaml.safe_load(text)
                    except ImportError:
                        raise ImportError(
                            "[DataDigestor] from_api_spec() con YAML requiere PyYAML.\n"
                            "  pip install pyyaml"
                        )
                else:
                    import json as _json
                    raw = _json.loads(text)
            else:
                raise FileNotFoundError(f"[DataDigestor] Spec no encontrado: {spec}")

        # ── 2. Extraer info general ────────────────────────────────────
        info   = raw.get("info", {})
        api_title = info.get("title", "API")
        api_desc  = info.get("description", "")
        base_url  = ""
        # OpenAPI 3.x
        servers = raw.get("servers", [])
        if servers:
            base_url = servers[0].get("url", "")
        # Swagger 2.x
        if not base_url:
            host   = raw.get("host", "")
            scheme = (raw.get("schemes", ["https"]) or ["https"])[0]
            if host:
                base_url = f"{scheme}://{host}{raw.get('basePath', '')}"

        # ── 3. Iterar paths ────────────────────────────────────────────
        paths: dict = raw.get("paths", {})
        endpoints_raw: list[tuple[str, str, dict]] = []  # (path, method, op)
        for path, methods in paths.items():
            for method, operation in methods.items():
                if method.lower() not in ("get", "post", "put", "patch", "delete"):
                    continue
                if not isinstance(operation, dict):
                    continue
                endpoints_raw.append((path, method.upper(), operation))

        rng.shuffle(endpoints_raw)
        if n is not None:
            endpoints_raw = endpoints_raw[:n]

        # ── 4. Generar ejemplos ────────────────────────────────────────
        generated = 0
        for path, method, operation in endpoints_raw:
            example = self._api_op_to_example(
                path, method, operation, base_url, api_title, format
            )
            if example:
                self._examples.append(example)
                generated += 1

        print(
            f"[DataDigestor] from_api_spec: {generated} ejemplos generados "
            f"de {len(endpoints_raw)} endpoints ({api_title})"
        )
        return self

    # ── Auxiliar de from_api_spec ──────────────────────────────────────────
    def _api_op_to_example(
        self,
        path:      str,
        method:    str,
        operation: dict,
        base_url:  str,
        api_title: str,
        format:    str,
    ) -> Optional[dict]:
        """
        Convierte un único endpoint OpenAPI en un ejemplo ChatML.
        Devuelve None si el endpoint no tiene suficiente información.
        """
        summary = operation.get("summary") or operation.get("description") or ""
        if not summary:
            return None

        # Construir descripción del endpoint para el tool
        params       = operation.get("parameters", [])
        request_body = operation.get("requestBody", {})
        responses    = operation.get("responses", {})

        # Parámetros de ejemplo
        param_example: dict = {}
        for p in params:
            if not isinstance(p, dict):
                continue
            name    = p.get("name", "param")
            p_in    = p.get("in", "query")
            schema  = p.get("schema", {})
            p_type  = schema.get("type", "string")
            example = p.get("example") or schema.get("example") or _API_TYPE_DEFAULT.get(p_type, "value")
            if p_in in ("query", "path"):
                param_example[name] = example

        # Body de ejemplo
        body_example: dict = {}
        if request_body:
            content = request_body.get("content", {})
            for mime, media in content.items():
                if "json" in mime:
                    schema = media.get("schema", {})
                    body_example = _schema_to_example(schema)
                    break

        # Respuesta de ejemplo
        success_resp = responses.get("200") or responses.get("201") or {}
        resp_desc = success_resp.get("description", "Operación completada con éxito.")

        # Action Input para el ejemplo
        action_input: dict = {
            "method": method,
            "url":    f"{base_url}{path}",
        }
        if param_example:
            action_input["params"] = param_example
        if body_example:
            action_input["body"] = body_example

        # Pregunta del usuario (derivada del summary)
        user_q = _api_summary_to_user_request(summary, method, path, api_title)

        if format == "react":
            assistant_text = (
                f"Thought: {summary}. Llamo al endpoint {method} {path}.\n"
                f"Action: http_get\n"
                f"Action Input: {json.dumps(action_input, ensure_ascii=False)}\n"
                f"Observation: {resp_desc}\n"
                f"Thought: He obtenido la respuesta del endpoint.\n"
                f"Final Answer: {resp_desc}"
            )
            return {
                "messages": [
                    {"role": "system",    "content": self.system_prompt},
                    {"role": "user",      "content": user_q},
                    {"role": "assistant", "content": assistant_text},
                ]
            }

        if format == "function_call":
            # Formato de tool-calling nativo
            tool_desc = {
                "type": "function",
                "function": {
                    "name": _path_to_tool_name(path, method),
                    "description": summary,
                    "parameters": {
                        "type": "object",
                        "properties": {p.get("name", "p"): {"type": p.get("schema", {}).get("type", "string"), "description": p.get("description", "")} for p in params if isinstance(p, dict)},
                    },
                },
            }
            tool_call_content = (
                f"<tool_call>\n"
                f'{json.dumps({"name": _path_to_tool_name(path, method), "arguments": {**param_example, **body_example}}, ensure_ascii=False)}\n'
                f"</tool_call>"
            )
            return {
                "messages": [
                    {"role": "system",    "content": self.system_prompt},
                    {"role": "user",      "content": user_q},
                    {"role": "assistant", "content": tool_call_content},
                    {"role": "tool",      "name": _path_to_tool_name(path, method), "content": resp_desc},
                    {"role": "assistant", "content": resp_desc},
                ],
                "tools": [tool_desc],
            }

        raise ValueError(f"from_api_spec: format desconocido '{format}'. Usa 'react' o 'function_call'.")

    # ── S6.1 — generate_tool_calls ─────────────────────────────────────────

    def generate_tool_calls(
        self,
        tools_list: List[Dict[str, Any]],
        examples: Optional[List[Union[str, Dict[str, Any]]]] = None,
        n_per_tool: int = 5,
        format: str = "react",
        seed: int = 42,
    ) -> "DataDigestor":
        """
        Genera un dataset de function-calling a partir de una lista de herramientas
        y ejemplos opcionales de uso.

        Cada herramienta en ``tools_list`` debe ser un dict con:
            name        (str)  — nombre de la herramienta
            description (str)  — qué hace
            parameters  (dict) — parámetros como JSON Schema (properties + required)
                                 Ej: {"query": {"type": "str"}, "path": {"type": "str"}}

        Parámetros
        ----------
        tools_list : list[dict]
            Lista de herramientas. Mínimo: ``name`` + ``description``.
        examples : list[str | dict] | None
            Ejemplos explícitos del usuario. Cada elemento puede ser:
              - str  : petición libre, el digestor infiere qué herramienta usar
              - dict : {``user``: str, ``tool``: str, ``args``: dict}
                       ejemplo completo con herramienta y args ya especificados
        n_per_tool : int
            Ejemplos sintéticos a generar por herramienta cuando ``examples``
            no se proporciona. Por defecto 5.
        format : str
            ``"react"``          → Thought/Action/Action Input/Observation/Final Answer
            ``"function_call"``  → JSON tool_call estructurado
        seed : int
            Semilla para reproducibilidad.

        Devuelve
        --------
        DataDigestor (para encadenar llamadas)

        Ejemplo
        -------
        >>> d = DataDigestor(task="agente doméstico")
        >>> tools = [
        ...     {"name": "note_save",    "description": "Guarda una nota",
        ...      "parameters": {"title": {"type": "str"}, "body": {"type": "str"}}},
        ...     {"name": "file_organize","description": "Organiza archivos",
        ...      "parameters": {"files": {"type": "list"}, "dest": {"type": "str"}}},
        ... ]
        >>> d.generate_tool_calls(tools, n_per_tool=3).to_jsonl("agent_dataset.jsonl")
        """
        import random as _random

        rng = _random.Random(seed)

        # Construir índice de herramientas por nombre
        tools_by_name: Dict[str, Dict] = {t["name"]: t for t in tools_list}

        # ---------- Ejemplos explícitos -----------------------------------------
        explicit_examples: List[Dict] = []
        if examples:
            for ex in examples:
                if isinstance(ex, str):
                    # Inferencia de herramienta: elegir la más probable por keywords
                    best_tool = self._match_tool(ex, tools_list)
                    if best_tool:
                        args = self._sample_args(best_tool, rng)
                        explicit_examples.append({
                            "user": ex,
                            "tool": best_tool["name"],
                            "args": args,
                        })
                elif isinstance(ex, dict):
                    explicit_examples.append({
                        "user":  ex.get("user", ""),
                        "tool":  ex.get("tool", ""),
                        "args":  ex.get("args", {}),
                    })

        # ---------- Ejemplos sintéticos (n_per_tool × herramienta) ---------------
        synthetic_examples: List[Dict] = []
        for tool in tools_list:
            requests = self._generate_tool_requests(tool, n_per_tool, rng)
            for req in requests:
                args = self._sample_args(tool, rng)
                synthetic_examples.append({
                    "user": req,
                    "tool": tool["name"],
                    "args": args,
                })

        all_examples = explicit_examples + synthetic_examples
        rng.shuffle(all_examples)

        # ---------- Construir sistema y ejemplos ChatML --------------------------
        tools_desc = "\n".join(
            f"- {t['name']}: {t.get('description', '')} "
            f"| params: {list(t.get('parameters', {}).keys())}"
            for t in tools_list
        )
        system_prompt = (
            "Eres un asistente que puede usar herramientas. "
            "Cuando el usuario pida algo que requiera una herramienta, razona y llámala.\n"
            f"Herramientas disponibles:\n{tools_desc}"
        )

        generated = 0
        for ex in all_examples:
            user_msg  = ex["user"]
            tool_name = ex["tool"]
            args_dict = ex["args"]

            tool_info = tools_by_name.get(tool_name, {})
            result_str = f"Herramienta '{tool_name}' ejecutada correctamente."

            if format == "react":
                thought  = f"El usuario quiere que use '{tool_name}'. Voy a llamarla con los parámetros adecuados."
                assistant_text = (
                    f"Thought: {thought}\n"
                    f"Action: {tool_name}\n"
                    f"Action Input: {json.dumps(args_dict, ensure_ascii=False)}\n"
                    f"Observation: {result_str}\n"
                    f"Thought: La herramienta se ejecutó correctamente.\n"
                    f"Final Answer: Listo. {result_str}"
                )
                entry = {
                    "messages": [
                        {"role": "system",    "content": system_prompt},
                        {"role": "user",      "content": user_msg},
                        {"role": "assistant", "content": assistant_text},
                    ]
                }
            else:  # function_call
                tool_call_json = json.dumps(
                    {"tool": tool_name, "args": args_dict},
                    ensure_ascii=False,
                )
                entry = {
                    "messages": [
                        {"role": "system",    "content": system_prompt},
                        {"role": "user",      "content": user_msg},
                        {"role": "assistant", "content": tool_call_json},
                    ]
                }

            self._examples.append(entry)
            generated += 1

        print(f"[DataDigestor] generate_tool_calls: {generated} ejemplos generados "
              f"({len(explicit_examples)} explícitos + {len(synthetic_examples)} sintéticos)")
        return self

    # ── Auxiliares de generate_tool_calls ─────────────────────────────────

    def _match_tool(self, user_text: str, tools_list: List[Dict]) -> Optional[Dict]:
        """Heurística simple: la herramienta cuya descripción comparte más keywords con la petición."""
        user_words = set(re.findall(r'\w+', user_text.lower()))
        best_tool  = None
        best_score = 0
        for tool in tools_list:
            desc_words = set(re.findall(r'\w+', (tool.get("description", "") + " " + tool["name"]).lower()))
            score = len(user_words & desc_words)
            if score > best_score:
                best_score = score
                best_tool  = tool
        return best_tool

    def _sample_args(self, tool: Dict, rng: "random.Random") -> Dict:
        """Genera argumentos de muestra realistas para una herramienta."""
        import random
        params = tool.get("parameters", {})
        args: Dict[str, Any] = {}

        _STRING_SAMPLES: Dict[str, List[str]] = {
            "title":    ["Reunión del lunes", "Lista de compras", "Ideas proyecto"],
            "body":     ["Recordar revisar informe", "Pan, leche, huevos", "Usar LoRA para el prototipo"],
            "query":    ["informe trimestral", "presupuesto 2026", "notas reunión"],
            "path":     ["~/Documentos", "~/Escritorio", "~/Descargas"],
            "dest":     ["~/Documentos/Trabajo", "~/Archivos", "~/Backup"],
            "folder":   ["inbox", "sent", "archive"],
            "name":     ["backup", "proyecto", "datos"],
            "notebook": ["Personal", "Trabajo", "Proyectos"],
            "command":  ["ls -la", "pwd", "echo hola"],
            "url":      ["https://example.com/api", "https://api.github.com"],
        }
        _DEFAULT_STRINGS = ["ejemplo", "valor_prueba", "dato"]

        for param_name, param_schema in params.items():
            ptype = param_schema.get("type", "str")
            if ptype in ("str", "string"):
                candidates = _STRING_SAMPLES.get(param_name, _DEFAULT_STRINGS)
                args[param_name] = rng.choice(candidates)
            elif ptype in ("int", "integer"):
                args[param_name] = rng.randint(1, 30)
            elif ptype in ("float", "number"):
                args[param_name] = round(rng.uniform(0.1, 10.0), 2)
            elif ptype in ("bool", "boolean"):
                args[param_name] = rng.choice([True, False])
            elif ptype in ("list", "array"):
                candidates = _STRING_SAMPLES.get(param_name, _DEFAULT_STRINGS)
                args[param_name] = rng.sample(candidates, min(2, len(candidates)))
            else:
                args[param_name] = "valor_ejemplo"

        return args

    _TOOL_REQUEST_TEMPLATES: Dict[str, List[str]] = {
        "note_save":     ["Guarda una nota sobre {topic}", "Crea una nota titulada '{topic}'",
                          "Apunta esto: {topic}", "Necesito anotar algo sobre {topic}"],
        "file_organize": ["Organiza mis archivos de {topic}", "Mueve los {topic} a su carpeta",
                          "Ordena mis documentos de {topic}", "Pon los {topic} en su sitio"],
        "email_filter":  ["Filtra mis correos de {topic}", "Limpia la bandeja de {topic}",
                          "Archiva los emails de {topic}", "Muestra correos sobre {topic}"],
        "search_files":  ["Busca archivos sobre {topic}", "Encuentra documentos de {topic}",
                          "¿Dónde están los archivos de {topic}?", "Localiza mis {topic}"],
        "calendar_get":  ["¿Qué tengo en el calendario esta semana?", "Muéstrame mis eventos de {topic}",
                          "¿Tengo algo programado para {topic}?", "Consulta mi agenda de {topic}"],
        "process_run":   ["Ejecuta {topic}", "Lanza el proceso {topic}", "Inicia {topic}"],
    }
    _TOPICS = ["trabajo", "reuniones", "facturas", "proyectos", "personal",
               "presupuesto", "viaje", "compras", "informe", "equipo"]

    def _generate_tool_requests(
        self,
        tool: Dict,
        n: int,
        rng: "random.Random",
    ) -> List[str]:
        """Genera n peticiones de usuario sintéticas para una herramienta concreta."""
        import random
        tool_name  = tool["name"]
        templates  = self._TOOL_REQUEST_TEMPLATES.get(
            tool_name,
            [
                f"Usa la herramienta {tool_name} para {{topic}}",
                f"Necesito que llames a {tool_name} sobre {{topic}}",
                f"Ejecuta {tool_name} con {{topic}}",
            ]
        )
        requests = []
        for _ in range(n):
            tmpl  = rng.choice(templates)
            topic = rng.choice(self._TOPICS)
            requests.append(tmpl.format(topic=topic))
        return requests

    def to_jsonl(
        self,
        output_path: Union[str, Path],
        shuffle: bool = True,
        seed: int = 42,
        deduplicate: bool = True,
    ) -> int:
        """
        Exporta todos los ejemplos acumulados a un archivo .jsonl.

        Por defecto, aplica deduplicación automática (exacta + near-dupes)
        antes de exportar. Usa deduplicate=False para desactivarlo.

        Parámetros
        ----------
        output_path : str | Path
        shuffle : bool
            Mezclar los ejemplos antes de exportar. Por defecto True.
        seed : int
            Semilla para la mezcla. Por defecto 42.
        deduplicate : bool
            Ejecutar deduplicación antes de exportar. Por defecto True.

        Devuelve
        --------
        int
            Número de ejemplos exportados.
        """
        if not self._examples:
            print("[DataDigestor] Nada que exportar — no hay ejemplos cargados.")
            return 0

        # ── Deduplicación automática (S2.2) ──────────────────────────
        if deduplicate:
            self.deduplicate()
        # ──────────────────────────────────────────────────────────────

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        examples = list(self._examples)
        if shuffle:
            import random
            rng = random.Random(seed)
            rng.shuffle(examples)

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"[DataDigestor] Exportado: {len(examples)} ejemplos -> {output_path}")
        self.validate(verbose=True)
        return len(examples)

    def get_examples(self) -> List[dict]:
        """Devuelve la lista interna de ejemplos (para inspección)."""
        return list(self._examples)

    def load_jsonl(self, path: Union[str, Path]) -> "DataDigestor":
        """
        Carga ejemplos ya formateados desde un JSONL directamente en _examples.
        Útil para convertir un dataset existente a otro framework sin reprocesar.

        Parámetros
        ----------
        path : str | Path
            Ruta al archivo .jsonl con ejemplos en formato ChatML o Alpaca.

        Devuelve
        --------
        DataDigestor (para encadenar llamadas)
        """
        path = Path(path)
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._examples.append(json.loads(line))
                    count += 1
        print(f"[DataDigestor] Cargados {count} ejemplos desde {path}")
        return self

    # ------------------------------------------------------------------
    # Exportacion universal (G2: Unsloth, LLaMA-Factory, Axolotl)
    # ------------------------------------------------------------------

    def to_unsloth(
        self,
        output_path: Union[str, Path],
        shuffle: bool = True,
        seed: int = 42,
    ) -> int:
        """
        Exporta en formato Alpaca compatible con Unsloth.

        Unsloth acepta datasets Alpaca con campos: instruction, input, output.
        El campo 'input' lleva los datos serializados, 'output' la etiqueta.

        Parámetros
        ----------
        output_path : str | Path
        shuffle : bool
        seed : int

        Devuelve
        --------
        int — número de ejemplos exportados.
        """
        if not self._examples:
            print("[DataDigestor] Nada que exportar.")
            return 0

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        examples = list(self._examples)
        if shuffle:
            import random
            rng = random.Random(seed)
            rng.shuffle(examples)

        alpaca_examples = []
        for ex in examples:
            if "messages" in ex:
                msgs = ex["messages"]
                instruction = next((m["content"] for m in msgs if m["role"] == "system"), "")
                user_text = next((m["content"] for m in msgs if m["role"] == "user"), "")
                assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
                # Separar task del input: task va en instruction, datos en input
                if "\n\n" in user_text:
                    task, inp = user_text.split("\n\n", 1)
                else:
                    task, inp = user_text, ""
                alpaca_examples.append({
                    "instruction": instruction or task,
                    "input": inp,
                    "output": assistant or "",
                })
            else:
                # Ya está en formato Alpaca
                alpaca_examples.append(ex)

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in alpaca_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"[DataDigestor] Unsloth/Alpaca exportado: {len(alpaca_examples)} ejemplos -> {output_path}")
        return len(alpaca_examples)

    def to_llamafactory(
        self,
        output_dir: Union[str, Path],
        dataset_name: str = "mi_dataset",
        shuffle: bool = True,
        seed: int = 42,
    ) -> int:
        """
        Exporta el dataset en formato compatible con LLaMA-Factory.

        Genera:
        - {dataset_name}.json  (formato ShareGPT con conversations)
        - dataset_info.json    (configuracion que LLaMA-Factory necesita)

        Parámetros
        ----------
        output_dir : str | Path
            Carpeta de salida.
        dataset_name : str
            Nombre lógico del dataset (aparece en dataset_info.json).
        shuffle : bool
        seed : int

        Devuelve
        --------
        int — número de ejemplos exportados.
        """
        if not self._examples:
            print("[DataDigestor] Nada que exportar.")
            return 0

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        examples = list(self._examples)
        if shuffle:
            import random
            rng = random.Random(seed)
            rng.shuffle(examples)

        # Convertir a formato ShareGPT (conversations)
        sharegpt_examples = []
        for ex in examples:
            conversations = []
            if "messages" in ex:
                for m in ex["messages"]:
                    role = "human" if m["role"] == "user" else "gpt" if m["role"] == "assistant" else "system"
                    conversations.append({"from": role, "value": m["content"]})
            else:
                conversations.append({"from": "system", "value": ex.get("instruction", "")})
                if ex.get("input"):
                    conversations.append({"from": "human", "value": ex["input"]})
                conversations.append({"from": "gpt", "value": ex.get("output", "")})
            sharegpt_examples.append({"conversations": conversations})

        # Guardar JSON
        json_path = output_dir / f"{dataset_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sharegpt_examples, f, ensure_ascii=False, indent=2)

        # Generar dataset_info.json automático
        dataset_info = {
            dataset_name: {
                "file_name": f"{dataset_name}.json",
                "formatting": "sharegpt",
                "columns": {"messages": "conversations"},
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                    "system_tag": "system",
                },
            }
        }
        info_path = output_dir / "dataset_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

        print(f"[DataDigestor] LLaMA-Factory exportado: {len(sharegpt_examples)} ejemplos")
        print(f"  Dataset: {json_path}")
        print(f"  Config:  {info_path}")
        print(f"  Usar en LLaMA-Factory: copia estos archivos a data/ y añade "
              f"'{dataset_name}' en dataset_info.json")
        return len(sharegpt_examples)

    def to_axolotl(
        self,
        output_dir: Union[str, Path],
        dataset_name: str = "mi_dataset",
        shuffle: bool = True,
        seed: int = 42,
    ) -> int:
        """
        Exporta el dataset en formato compatible con Axolotl.

        Genera:
        - {dataset_name}.jsonl  (formato ShareGPT)
        - axolotl_config.yml    (configuracion basica)

        Parámetros
        ----------
        output_dir : str | Path
        dataset_name : str
        shuffle : bool
        seed : int

        Devuelve
        --------
        int — número de ejemplos exportados.
        """
        if not self._examples:
            print("[DataDigestor] Nada que exportar.")
            return 0

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        examples = list(self._examples)
        if shuffle:
            import random
            rng = random.Random(seed)
            rng.shuffle(examples)

        # Axolotl usa ShareGPT en formato JSONL
        jsonl_path = output_dir / f"{dataset_name}.jsonl"
        count = 0
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for ex in examples:
                if "messages" in ex:
                    conversations = []
                    for m in ex["messages"]:
                        role = "human" if m["role"] == "user" else "gpt" if m["role"] == "assistant" else "system"
                        conversations.append({"from": role, "value": m["content"]})
                else:
                    conversations = [
                        {"from": "system", "value": ex.get("instruction", "")},
                        {"from": "human", "value": ex.get("input", "")},
                        {"from": "gpt", "value": ex.get("output", "")},
                    ]
                f.write(json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n")
                count += 1

        # Generar YAML de configuración
        yaml_content = f"""# Configuracion Axolotl generada por DataDigestor
base_model: Qwen/Qwen2.5-7B-Instruct  # <-- cambia por tu modelo
datasets:
  - path: {jsonl_path}
    type: sharegpt
    conversation: chatml
output_dir: ./output
sequence_len: 2048
lora_r: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - v_proj
num_epochs: 3
batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2e-4
"""
        yaml_path = output_dir / "axolotl_config.yml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        print(f"[DataDigestor] Axolotl exportado: {count} ejemplos")
        print(f"  Dataset: {jsonl_path}")
        print(f"  Config:  {yaml_path}")
        return count

    def validate(
        self,
        max_seq_length: int = 2048,
        verbose: bool = True,
        include_scores: bool = False,
    ) -> Dict[str, Any]:
        """
        Semáforo de calidad del dataset.

        Analiza los ejemplos acumulados y devuelve un informe con:
        - Cantidad total de ejemplos (semáforo: ROJO / AMARILLO / VERDE)
        - Distribución de etiquetas (detecta desbalance > 80%)
        - Longitud media de secuencias vs max_seq_length
        - Riesgo de fuga del prompt (clase dominante > 90%)

        Parámetros
        ----------
        max_seq_length : int
            Longitud máxima de contexto del modelo (en caracteres aprox).
            Se usa para estimar si los ejemplos caben en contexto.
        verbose : bool
            Si True, imprime el informe por pantalla.
        include_scores : bool
            Si True, incluye la clave ``quality_scores`` en el resultado:
            lista de dicts con índice, puntuación 0-100 y desglose
            (length_score, noise_score, format_score).
            Equivale a llamar ``score_examples()`` e incluir el resultado.

        Devuelve
        --------
        dict con claves:
            total, label_counts, label_pct, avg_chars, max_chars,
            semaforo (str: "ROJO" | "AMARILLO" | "VERDE"),
            warnings (list[str]),
            quality_scores (list[dict], solo si include_scores=True)
        """
        if not self._examples:
            print("[DataDigestor] validate(): no hay ejemplos cargados.")
            return {"total": 0, "semaforo": "ROJO", "warnings": ["Sin ejemplos"]}

        total = len(self._examples)
        label_counts: Dict[str, int] = {}
        char_lengths: List[int] = []

        for ex in self._examples:
            # Calcular longitud del ejemplo completo
            if self.output_format == "chatml":
                msgs = ex.get("messages", [])
                # Aplanar contenido multimodal (listas) a string para contabilizar
                _parts: List[str] = []
                for m in msgs:
                    c = m.get("content", "")
                    if isinstance(c, list):
                        for item in c:
                            if isinstance(item, dict):
                                _parts.append(str(item.get("text", item.get("image", ""))))
                            else:
                                _parts.append(str(item))
                    else:
                        _parts.append(str(c))
                full_text = " ".join(_parts)
                assistant_msg = next(
                    (m["content"] for m in msgs if m.get("role") == "assistant"), None
                )
                # El content del assistant puede ser string o lista
                if isinstance(assistant_msg, list):
                    lbl = " ".join(str(x.get("text", "")) for x in assistant_msg if isinstance(x, dict))
                else:
                    lbl = str(assistant_msg) if assistant_msg else "<sin etiqueta>"
            else:
                full_text = ex.get("instruction", "") + " " + ex.get("input", "")
                lbl = ex.get("output") or "<sin etiqueta>"

            char_lengths.append(len(full_text))
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        avg_chars  = sum(char_lengths) / total
        max_chars  = max(char_lengths)
        # Convertir a tokens aproximados (1 token ≈ 4 chars)
        avg_tokens = avg_chars / 4
        max_tokens = max_chars / 4

        label_pct = {lbl: cnt / total * 100 for lbl, cnt in label_counts.items()}
        max_pct   = max(label_pct.values())
        dominant  = max(label_pct, key=label_pct.get)

        warnings: List[str] = []

        # --- Cantidad de ejemplos ---
        if total < 200:
            semaforo_cantidad = "ROJO"
            warnings.append(
                f"Solo {total} ejemplos — INSUFICIENTE para fine-tuning "
                f"(mínimo recomendado: 500)"
            )
        elif total < 500:
            semaforo_cantidad = "AMARILLO"
            warnings.append(
                f"Solo {total} ejemplos — LÍMITE BAJO (recomendado: 500-2000)"
            )
        elif total < 2000:
            semaforo_cantidad = "VERDE"
        else:
            semaforo_cantidad = "VERDE"

        # --- Desbalance de clases ---
        if len(label_counts) > 1 and max_pct > 80:
            warnings.append(
                f"Clase dominante '{dominant}' con {max_pct:.1f}% de los ejemplos "
                f"— DESBALANCE SEVERO (riesgo de sesgo)"
            )
        elif len(label_counts) == 1 and lbl != "<sin etiqueta>":
            warnings.append(
                f"Solo una clase '{dominant}' — el modelo puede aprender a "
                f"responder siempre lo mismo (RIESGO DE FUGA DEL PROMPT)"
            )

        # --- Riesgo de fuga del prompt (etiqueta muy dominante en clasificación) ---
        if len(label_counts) <= 3 and max_pct > 90:
            warnings.append(
                f"'{dominant}' aparece en {max_pct:.1f}% de los ejemplos. "
                f"El modelo puede memoizar esta respuesta — considera añadir "
                f"datos negativos o diversificar el dataset."
            )

        # --- Longitud de secuencias ---
        tokens_est = avg_tokens
        if tokens_est > max_seq_length * 0.9:
            warnings.append(
                f"Longitud media ~{tokens_est:.0f} tokens supera el 90% de "
                f"max_seq_length={max_seq_length} — muchos ejemplos serán truncados"
            )
        elif max_tokens > max_seq_length:
            warnings.append(
                f"Ejemplo más largo ~{max_tokens:.0f} tokens supera "
                f"max_seq_length={max_seq_length} — será truncado"
            )

        # --- Longitud mínima (S2.2) ---
        short_count = sum(1 for c in char_lengths if c / 4 < 10)
        short_pct = short_count / total * 100
        if short_pct > 20:
            warnings.append(
                f"{short_count} ejemplos ({short_pct:.1f}%) tienen <10 tokens. "
                f"Ejemplos demasiado cortos pueden no aportar suficiente señal "
                f"de entrenamiento."
            )

        # --- Consistencia de idioma (S2.2) ---
        try:
            from langdetect import detect_langs
            # Muestrear hasta 100 ejemplos para no ralentizar
            sample_size = min(100, total)
            lang_votes: Dict[str, int] = {}
            for i in range(sample_size):
                msgs = self._examples[i].get("messages", [])
                sample_text = " ".join(
                    str(m.get("content", ""))
                    for m in msgs
                    if not isinstance(m.get("content"), list)
                )
                if len(sample_text) < 15:
                    continue
                try:
                    # langdetect devuelve lista de probabilidades; nos quedamos con la primera
                    detected = detect_langs(sample_text)
                    if detected:
                        top_lang = detected[0].lang
                        lang_votes[top_lang] = lang_votes.get(top_lang, 0) + 1
                except Exception:
                    pass

            if lang_votes:
                majority_lang = max(lang_votes, key=lang_votes.get)
                majority_pct = lang_votes[majority_lang] / sum(lang_votes.values()) * 100
                minority_count = sum(
                    v for k, v in lang_votes.items() if k != majority_lang
                )
                if minority_count > sample_size * 0.05:
                    warnings.append(
                        f"Mezcla de idiomas detectada: {majority_lang} mayoritario "
                        f"({majority_pct:.0f}%), pero hay {minority_count} ejemplos en "
                        f"otros idiomas. Esto puede confundir al modelo si no es multilingüe."
                    )
        except ImportError:
            pass  # langdetect no instalado → no se comprueba idioma

        # --- Homogeneidad de la tarea (misma pregunta en todos los ejemplos) ---
        # Si el 100% de los ejemplos usan exactamente la misma pregunta/task,
        # el modelo corre riesgo de "memorizar" la pregunta y repetirla en
        # cualquier contexto (sesgo del prompt).
        task_texts: Dict[str, int] = {}
        sample_for_task = min(200, total)
        for i in range(sample_for_task):
            ex = self._examples[i]
            if "messages" in ex:
                # Extraer el contenido del user (sin system ni assistant)
                user_msg = next(
                    (m["content"] for m in ex["messages"] if m.get("role") == "user"), ""
                )
                # La task es lo que va ANTES del primer "\n\n" (si existe)
                task_part = str(user_msg).split("\n\n")[0].strip() if user_msg else ""
            else:
                task_part = str(ex.get("instruction", "")).strip()
            task_texts[task_part] = task_texts.get(task_part, 0) + 1

        if task_texts:
            most_common_task = max(task_texts, key=task_texts.get)
            task_dominance = task_texts[most_common_task] / sample_for_task * 100
            if task_dominance > 95 and len(task_texts) == 1:
                warnings.append(
                    f"TAREA HOMOGÉNEA: el 100% de los ejemplos usan exactamente "
                    f"la misma pregunta ('{most_common_task[:80]}...'). "
                    f"El modelo puede aprender a REPETIR la pregunta en vez de solo "
                    f"responder. Mezcla ejemplos con preguntas variadas (5-10% del "
                    f"total) o baja lora_r para mitigarlo."
                )
            elif task_dominance > 95:
                warnings.append(
                    f"Tarea muy homogénea: {task_dominance:.0f}% de los ejemplos "
                    f"usan la misma pregunta. Riesgo de sesgo del prompt — considera "
                    f"variar la redacción de la tarea en algunos ejemplos."
                )

        # --- Near-duplicados (S2.2) ---
        # Solo comprobamos una muestra para no ralentizar en datasets grandes
        near_dupe_count = 0
        dupe_sample = min(200, total)
        if dupe_sample > 1:
            _texts = []
            for i in range(dupe_sample):
                ex = self._examples[i]
                if "messages" in ex:
                    _t = " ".join(
                        str(m.get("content", "")) for m in ex["messages"]
                        if not isinstance(m.get("content"), list)
                    )
                else:
                    _t = f"{ex.get('instruction','')} {ex.get('input','')} {ex.get('output','')}"
                _texts.append(_t)

            for i in range(len(_texts)):
                if not _texts[i]:
                    continue
                for j in range(i + 1, len(_texts)):
                    if not _texts[j]:
                        continue
                    if self._jaccard_similarity(_texts[i], _texts[j]) >= 0.9:
                        near_dupe_count += 1
                        break  # solo contamos uno por ejemplo

        if near_dupe_count > 0:
            warnings.append(
                f"Detectados {near_dupe_count} near-duplicados en muestra de {dupe_sample}. "
                f"Considera llamar a digestor.deduplicate() antes de entrenar — "
                f"los duplicados no aportan diversidad y desperdician capacidad de entrenamiento."
            )

        # Semáforo global: solo warnings CRÍTICOS (datos insuficientes,
        # fuga del prompt, desbalance severo) afectan el color.
        # Warnings informativos (tarea homogénea, near-dupes, longitud)
        # no degradan el semáforo — se muestran como advertencias.
        criticos = [w for w in warnings if any(
            kw in w for kw in (
                "INSUFICIENTE", "FUGA", "memoizar",
                "DESBALANCE SEVERO", "Solo una clase",
            )
        )]
        if semaforo_cantidad == "ROJO" or any("INSUFICIENTE" in w or "FUGA" in w or "memoizar" in w for w in criticos):
            semaforo = "ROJO"
        elif semaforo_cantidad == "AMARILLO" or any("DESBALANCE SEVERO" in w or "Solo una clase" in w for w in criticos):
            semaforo = "AMARILLO"
        else:
            semaforo = "VERDE"

        result = {
            "total":        total,
            "label_counts": label_counts,
            "label_pct":    label_pct,
            "avg_chars":    round(avg_chars, 1),
            "max_chars":    max_chars,
            "avg_tokens_est": round(avg_tokens, 0),
            "max_tokens_est": round(max_tokens, 0),
            "semaforo":     semaforo,
            "warnings":     warnings,
        }

        if include_scores:
            result["quality_scores"] = self.score_examples()

        if verbose:
            _COLORS = {"ROJO": "\033[91m", "AMARILLO": "\033[93m", "VERDE": "\033[92m"}
            _RESET  = "\033[0m"
            color   = _COLORS.get(semaforo, "")
            symbol  = {"ROJO": "X", "AMARILLO": "~", "VERDE": "OK"}.get(semaforo, "?")
            print(f"\n{'='*60}")
            print(f"  SEMAFORO DE CALIDAD: {color}[{symbol} {semaforo}]{_RESET}")
            print(f"{'='*60}")
            print(f"  Total ejemplos    : {total}")
            print(f"  Longitud media    : {avg_chars:.0f} chars  "
                  f"(aprox {avg_tokens:.0f} tokens)")
            print(f"  Ejemplo mas largo : {max_chars} chars  "
                  f"(aprox {max_tokens:.0f} tokens)")
            print(f"  Clases distintas  : {len(label_counts)}")
            if label_counts:
                print("  Distribucion:")
                for lbl_k, pct in sorted(label_pct.items(), key=lambda x: -x[1]):
                    bar = "#" * int(pct / 5)
                    print(f"    {lbl_k[:25]:25s}: {pct:5.1f}%  {bar}")
            if warnings:
                print(f"\n  Advertencias ({len(warnings)}):")
                for w in warnings:
                    print(f"    WARNING: {w}")
            else:
                print(f"\n  {color}No se detectaron problemas.{_RESET}")
            print(f"{'='*60}\n")

        return result

    # ------------------------------------------------------------------
    # Puntuación de calidad por muestra (G4)
    # ------------------------------------------------------------------

    def score_examples(self) -> List[Dict[str, Any]]:
        """
        Calcula una puntuación de calidad 0-100 para cada ejemplo individual.

        Criterios (cada uno pesa 1/3):

        1. **length_score** (0-100): penaliza ejemplos demasiado cortos (<50 chars)
           o exageradamente largos (>8000 chars).  El rango óptimo es 150-4000 chars.

        2. **noise_score** (0-100): ratio de caracteres ruidosos (símbolos
           no alfanuméricos excluyendo puntuación básica) respecto al total.
           Baja el score cuando el porcentaje de ruido supera el 15%.

        3. **format_score** (0-100): comprueba coherencia básica del formato
           ChatML (roles system/user/assistant, contenido no vacío).

        Devuelve
        --------
        list[dict] con claves por ejemplo:
            index (int), score (int 0-100),
            length_score, noise_score, format_score,
            chars (int)
        """
        import re as _re

        scores: List[Dict[str, Any]] = []

        for i, ex in enumerate(self._examples):
            # ── Extraer texto completo ───────────────────────────────────
            if "messages" in ex:
                msgs = ex["messages"]
                full_text = " ".join(
                    str(m.get("content", ""))
                    for m in msgs
                    if not isinstance(m.get("content"), list)
                )
            else:
                full_text = (
                    str(ex.get("instruction", "")) + " " +
                    str(ex.get("input", "")) + " " +
                    str(ex.get("output", ""))
                )
            n_chars = len(full_text)

            # ── 1. length_score ─────────────────────────────────────────
            if n_chars < 50:
                length_score = max(0, int(n_chars / 50 * 40))        # 0-40
            elif n_chars < 150:
                length_score = 40 + int((n_chars - 50) / 100 * 30)   # 40-70
            elif n_chars <= 4000:
                length_score = 100
            elif n_chars <= 8000:
                excess = n_chars - 4000
                length_score = max(60, 100 - int(excess / 4000 * 40))
            else:
                length_score = max(0, 60 - int((n_chars - 8000) / 2000 * 20))

            # ── 2. noise_score ──────────────────────────────────────────
            # Caracteres ruidosos: no alfanumérico, no espacio, no puntuación básica
            noise_chars = len(_re.findall(r'[^\w\s.,;:!?¿¡\-\'"()\[\]{}/@#%&*+=<>]', full_text))
            noise_ratio = noise_chars / max(n_chars, 1)
            if noise_ratio <= 0.05:
                noise_score = 100
            elif noise_ratio <= 0.15:
                noise_score = int(100 - (noise_ratio - 0.05) / 0.10 * 40)  # 60-100
            elif noise_ratio <= 0.30:
                noise_score = int(60 - (noise_ratio - 0.15) / 0.15 * 40)   # 20-60
            else:
                noise_score = max(0, int(20 - noise_ratio * 10))

            # ── 3. format_score ─────────────────────────────────────────
            if "messages" in ex:
                roles = {m.get("role") for m in msgs}
                has_user      = "user"      in roles
                has_assistant = "assistant" in roles
                all_nonempty  = all(
                    bool(m.get("content")) for m in msgs
                )
                format_score = (
                    100 if has_user and has_assistant and all_nonempty
                    else (60 if has_user or has_assistant else 20)
                )
            else:
                has_instruction = bool(ex.get("instruction", "").strip())
                has_output      = bool(ex.get("output", "").strip())
                format_score = (
                    100 if has_instruction and has_output
                    else (60 if has_instruction or has_output else 20)
                )

            overall = int((length_score + noise_score + format_score) / 3)

            scores.append({
                "index":        i,
                "score":        overall,
                "length_score": length_score,
                "noise_score":  noise_score,
                "format_score": format_score,
                "chars":        n_chars,
            })

        return scores

    # ------------------------------------------------------------------
    # Limpieza y balanceo (S2.2 — Semáforo v2)
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """
        Calcula la similitud de Jaccard entre dos textos a nivel de palabras.

        Jaccard = |palabras comunes| / |palabras totales (unión)|

        Un valor ≥ 0.9 indica que dos ejemplos son prácticamente idénticos
        y no aportan diversidad al entrenamiento.
        """
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def deduplicate(self, jaccard_threshold: float = 0.9) -> int:
        """
        Elimina ejemplos duplicados — exactos y near-duplicados.

        Algoritmo
        ---------
        1. Excluye ejemplos VLM (con imágenes) — no se deduplican.
        2. Extrae el texto limpio de cada ejemplo (user + assistant).
        3. Primera pasada: elimina duplicados exactos (mismo texto).
        4. Segunda pasada: elimina near-duplicados (Jaccard ≥ threshold).

        Parámetros
        ----------
        jaccard_threshold : float
            Umbral de similitud Jaccard para near-duplicados. Por defecto 0.9.

        Devuelve
        --------
        int — número de ejemplos eliminados.

        Nota
        ----
        La deduplicación es destructiva — modifica self._examples in-place.
        Se recomienda llamar a este método ANTES de to_jsonl() si se quiere
        mantener el dataset limpio.
        """
        if not self._examples:
            print("[DataDigestor] deduplicate(): no hay ejemplos que deduplicar.")
            return 0

        original_count = len(self._examples)

        # ── Extraer texto limpio de cada ejemplo ────────────────────
        def _extract_text(ex: dict) -> str:
            """Extrae el texto combinado de user + assistant de un ejemplo."""
            if "messages" in ex:
                # ChatML
                texts = []
                for m in ex["messages"]:
                    content = m.get("content", "")
                    if isinstance(content, list):
                        # VLM: contenido multimodal → saltar
                        return ""
                    texts.append(str(content))
                return " ".join(texts)
            elif "instruction" in ex:
                # Alpaca
                return f"{ex.get('instruction','')} {ex.get('input','')} {ex.get('output','')}"
            elif "text" in ex:
                # Completion (nivel conocimiento crudo)
                return str(ex["text"])
            return ""

        texts = [_extract_text(ex) for ex in self._examples]

        removed_exact = 0
        removed_near = 0
        keep_indices: List[int] = []

        # ── Primera pasada: duplicados exactos ──────────────────────
        seen: Dict[str, int] = {}   # texto → índice del primer ejemplo
        for i, t in enumerate(texts):
            if not t:
                # Ejemplos VLM o vacíos → mantener siempre
                keep_indices.append(i)
                continue
            if t in seen:
                removed_exact += 1
            else:
                seen[t] = i
                keep_indices.append(i)

        # ── Segunda pasada: near-duplicados (solo entre los que sobrevivieron) ─
        survivors = [self._examples[i] for i in keep_indices]
        survivor_texts = [texts[i] for i in keep_indices]

        final_keep: List[int] = []
        # Para cada superviviente, comprobar contra los ya aceptados
        for i, (ex, t) in enumerate(zip(survivors, survivor_texts)):
            if not t:
                final_keep.append(i)
                continue
            is_dupe = False
            for j in final_keep:
                other_t = survivor_texts[j]
                if not other_t:
                    continue
                if self._jaccard_similarity(t, other_t) >= jaccard_threshold:
                    is_dupe = True
                    removed_near += 1
                    break
            if not is_dupe:
                final_keep.append(i)

        # ── Reconstruir examples ────────────────────────────────────
        self._examples = [survivors[i] for i in final_keep]
        total_removed = removed_exact + removed_near
        final_count = len(self._examples)

        if total_removed > 0:
            print(
                f"[DataDigestor] Deduplicación: {original_count} -> {final_count} "
                f"(-{total_removed}: {removed_exact} exactos, {removed_near} near-dupes)"
            )
        else:
            print(f"[DataDigestor] Deduplicación: no se encontraron duplicados "
                  f"({original_count} ejemplos)")

        return total_removed

    def rebalance(self, strategy: str = "oversample") -> int:
        """
        Rebalancea las clases del dataset para corregir desbalance severo (>80%).

        Estrategias
        -----------
        - "oversample": duplica aleatoriamente ejemplos de clases minoritarias
          hasta que la clase más pequeña tenga al menos un 25% del total.

        Parámetros
        ----------
        strategy : str
            Estrategia de rebalanceo. Solo "oversample" por ahora.

        Devuelve
        --------
        int — número de ejemplos añadidos (0 si no se necesitó rebalanceo).
        """
        if not self._examples:
            print("[DataDigestor] rebalance(): no hay ejemplos.")
            return 0

        import random
        rng = random.Random(42)

        # ── Contar clases ───────────────────────────────────────────
        label_to_examples: Dict[str, List[dict]] = {}
        for ex in self._examples:
            lbl = self._extract_label(ex)
            label_to_examples.setdefault(lbl, []).append(ex)

        if len(label_to_examples) < 2:
            print("[DataDigestor] rebalance(): solo hay una clase — no se puede rebalancear.")
            return 0

        total = len(self._examples)
        max_pct = max(len(v) for v in label_to_examples.values()) / total * 100

        if max_pct <= 80:
            print(f"[DataDigestor] rebalance(): clases balanceadas (máx {max_pct:.1f}%) — no necesario.")
            return 0

        # ── Oversample de clases minoritarias ────────────────────────
        target_min_pct = 0.25  # la clase más pequeña debe ser ≥25%
        target_count = int(total * target_min_pct)
        added = 0

        for lbl, examples in label_to_examples.items():
            if len(examples) < target_count:
                needed = target_count - len(examples)
                # Duplicar aleatoriamente con reemplazo
                new_examples = [dict(ex) for ex in rng.choices(examples, k=needed)]
                self._examples.extend(new_examples)
                added += needed
                label_to_examples[lbl].extend(new_examples)

        if added > 0:
            print(
                f"[DataDigestor] Rebalanceo ({strategy}): +{added} ejemplos sinteticos "
                f"-> total: {len(self._examples)}"
            )

        return added

    # ------------------------------------------------------------------
    # Aumentación sintética (G1)
    # ------------------------------------------------------------------

    def augment(
        self,
        strategy: str = "template_swap",
        n_augmented: Optional[int] = None,
    ) -> "DataDigestor":
        """
        Aumenta el dataset generando ejemplos sintéticos — útil cuando el
        semáforo es ROJO (<200 ejemplos).

        Estrategias
        -----------
        - "template_swap": crea copias de cada ejemplo cambiando únicamente
          la formulación de la tarea (rephrase de la pregunta / instrucción).
          El texto de entrada y la etiqueta no cambian. Ideal para datasets
          de clasificación con pocas filas.
        - "paraphrase": aplica sustitución de sinónimos (tabla EN+ES de ~90
          pares) sobre el texto de entrada. La etiqueta no cambia. 100%
          offline — sin LLM externo.

        Parámetros
        ----------
        strategy : str
            "template_swap" (por defecto) | "paraphrase"
        n_augmented : int, opcional
            Número de ejemplos nuevos a añadir. Si None, aumenta hasta
            alcanzar 200 ejemplos (umbral VERDE mínimo). Si el dataset ya
            tiene ≥200 ejemplos, no hace nada (pasa n_augmented=N para
            forzarlo igualmente).

        Devuelve
        --------
        DataDigestor (self) para encadenamiento fluido.

        Ejemplo
        -------
        >>> d = DataDigestor("¿Es este contrato abusivo? Responde SÍ o NO.")
        >>> d.from_csv("contratos.csv").augment("template_swap").to_jsonl("out.jsonl")
        """
        if not self._examples:
            print("[DataDigestor] augment(): no hay ejemplos base para aumentar.")
            return self

        if strategy not in ("template_swap", "paraphrase"):
            raise ValueError(
                f"[DataDigestor] augment(): strategy desconocida: {strategy!r}. "
                f"Opciones: 'template_swap', 'paraphrase'"
            )

        import random
        rng = random.Random(42)

        original_count = len(self._examples)
        target = n_augmented if n_augmented is not None else max(0, 200 - original_count)

        if target <= 0:
            print(
                f"[DataDigestor] augment(): dataset ya tiene {original_count} ejemplos "
                f"(>=200) — no es necesario aumentar. "
                f"Pasa n_augmented=N para forzarlo."
            )
            return self

        new_examples: List[dict] = []

        if strategy == "template_swap":
            # Ciclar sobre los ejemplos base generando nuevas variantes de task.
            # _build_example → _task_variation() elige una variante aleatoria
            # por lo que sucesivas llamadas producen formulaciones distintas.
            pool = list(self._examples)
            i = 0
            while len(new_examples) < target:
                ex = pool[i % len(pool)]
                text = self._extract_input_text(ex)
                label: Optional[str] = self._extract_label(ex)
                if label == "<sin etiqueta>":
                    label = None
                new_examples.append(self._build_example(text, label))
                i += 1

        else:  # paraphrase
            pool = list(self._examples)
            i = 0
            max_iterations = len(pool) * 5  # salvaguarda contra bucle infinito
            iterations = 0
            while len(new_examples) < target and iterations < max_iterations:
                ex = pool[i % len(pool)]
                text = self._extract_input_text(ex)
                label = self._extract_label(ex)
                if label == "<sin etiqueta>":
                    label = None
                paraphrased = self._apply_synonyms(text, rng)
                if paraphrased != text:
                    new_examples.append(self._build_example(paraphrased, label))
                i += 1
                iterations += 1

            if not new_examples:
                print(
                    "[DataDigestor] augment(paraphrase): ningún texto cambió con "
                    "los sinónimos disponibles — prueba strategy='template_swap'."
                )
                return self

        self._examples.extend(new_examples)
        print(
            f"[DataDigestor] augment({strategy!r}): "
            f"+{len(new_examples)} ejemplos sinteticos "
            f"({original_count} -> {len(self._examples)})"
        )
        return self

    # ------------------------------------------------------------------
    # Helpers de extracción (compartidos por validate / deduplicate / rebalance / augment)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_label(ex: dict) -> str:
        """Extrae la etiqueta de un ejemplo en cualquier formato."""
        if "messages" in ex:
            for m in ex["messages"]:
                if m.get("role") == "assistant":
                    content = m.get("content", "")
                    if isinstance(content, str):
                        return content.strip()
            return "<sin etiqueta>"
        elif "output" in ex:
            return str(ex.get("output", "")).strip() or "<sin etiqueta>"
        return "<sin etiqueta>"

    @staticmethod
    def _extract_input_text(ex: dict) -> str:
        """
        Extrae el texto de entrada de un ejemplo (inverso de _build_example).

        Para ChatML: devuelve la parte DESPUÉS del primer \\n\\n en el mensaje
        user (donde _build_chatml almacena los datos tras la task).
        Para Alpaca: devuelve el campo ``input``.
        """
        if "messages" in ex:
            user_msg = next(
                (m.get("content", "") for m in ex["messages"] if m.get("role") == "user"),
                "",
            )
            if isinstance(user_msg, list):
                # Contenido multimodal (VLM) — devolver como está (sin modificar)
                return str(user_msg)
            if "\n\n" in str(user_msg):
                return str(user_msg).split("\n\n", 1)[1]
            return str(user_msg)
        return str(ex.get("input", ""))

    @staticmethod
    def _apply_synonyms(text: str, rng: "random.Random") -> str:  # type: ignore[name-defined]
        """
        Aplica sustitución de sinónimos al texto usando _SYNONYM_TABLE.

        Recorre las palabras del texto y reemplaza ~30% de las que tienen
        sinónimo disponible (para no alterar demasiado el texto original).
        La sustitución es sensible a mayúsculas/minúsculas: conserva la
        capitalización original de la palabra.

        Parámetros
        ----------
        text : str
            Texto de entrada.
        rng : random.Random
            Generador de números aleatorios (para reproducibilidad).

        Devuelve
        --------
        str — texto con sustituciones aplicadas.
        """
        import re
        # Tokenizar preservando espacios y puntuación
        tokens = re.findall(r"\w+|\W+", text)
        result = []
        for token in tokens:
            lower = token.lower()
            if lower in _SYNONYM_TABLE and rng.random() < 0.3:
                synonym = _SYNONYM_TABLE[lower]
                # Conservar capitalización original
                if token[0].isupper():
                    synonym = synonym[0].upper() + synonym[1:]
                result.append(synonym)
            else:
                result.append(token)
        return "".join(result)

    def clear(self) -> "DataDigestor":
        """Vacía el acumulador de ejemplos."""
        self._examples.clear()
        return self

    def from_excel(
        self,
        path: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        text_cols: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ) -> "DataDigestor":
        """
        Carga un archivo Excel (.xlsx / .xls) y genera ejemplos de entrenamiento.
        Requiere: pip install openpyxl

        Parámetros
        ----------
        path : str | Path
            Ruta al archivo Excel.
        sheet_name : str | int
            Nombre o índice de la hoja a cargar. Por defecto la primera (0).
        text_cols : list[str], opcional
            Columnas a incluir como texto de entrada.
            Si es None se usan todas menos label_col.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("[DataDigestor] 'pandas' no instalado. pip install pandas")
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError(
                "[DataDigestor] 'openpyxl' no instalado. "
                "Ejecuta: pip install openpyxl"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Archivo no encontrado: {path}")

        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        print(f"[DataDigestor] Excel cargado: {len(df)} filas, {len(df.columns)} columnas")
        print(f"  Hoja: {sheet_name}  |  Columnas: {list(df.columns)}")

        # ── Auto-enriquecimiento de dominio ─────────────────────────
        sample_size = min(100, len(df))
        _scols = text_cols if text_cols else [c for c in df.columns if c != self.label_col]
        sample_texts = [
            self._serialize_row(row, _scols)
            for _, row in df.head(sample_size).iterrows()
        ]
        self._auto_enrich([t for t in sample_texts if t.strip()])
        # ──────────────────────────────────────────────────────────────

        if self.label_col and self.label_col not in df.columns:
            raise ValueError(
                f"[DataDigestor] label_col={self.label_col!r} no encontrada. "
                f"Columnas: {list(df.columns)}"
            )

        if text_cols is None:
            text_cols = [c for c in df.columns if c != self.label_col]

        added = skipped = 0
        for _, row in df.iterrows():
            if self.label_col:
                raw_label = row[self.label_col]
                if self.skip_nulls and pd.isna(raw_label):
                    skipped += 1
                    continue
                label = self._resolve_label(raw_label)
            else:
                label = None

            text = self._serialize_row(row, text_cols)
            if not text.strip():
                skipped += 1
                continue

            self._examples.append(self._build_example(text, label))
            added += 1

        print(f"  Ejemplos añadidos: {added}  |  Omitidos: {skipped}")
        return self

    def from_image(
        self,
        path: Union[str, Path],
        label: Optional[str] = None,
        lang: str = "es",
    ) -> "DataDigestor":
        """
        Extrae texto de una imagen con OCR y genera un ejemplo.
        Requiere: pip install easyocr

        Parámetros
        ----------
        path : str | Path
            Ruta a la imagen (.png, .jpg, .jpeg, .webp, .bmp).
        label : str, opcional
            Etiqueta del ejemplo. Si es None, modo extracción.
        lang : str
            Idioma para OCR. Ej: "es" (español), "en" (inglés), "es+en" (ambos).
            Ver lista completa en: https://www.jaided.ai/easyocr/
        """
        try:
            import easyocr
        except ImportError:
            raise ImportError(
                "[DataDigestor] 'easyocr' no instalado. "
                "Ejecuta: pip install easyocr"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Imagen no encontrada: {path}")

        langs = [l.strip() for l in lang.replace("+", ",").split(",")]
        reader = easyocr.Reader(langs, gpu=False, verbose=False)
        results = reader.readtext(str(path), detail=0)
        text = _clean_text(" ".join(results))

        if not text.strip():
            print(f"  [WARN] OCR no extrajo texto de {path.name}")
            return self

        resolved_label = self._resolve_label(label) if label is not None else None
        self._examples.append(self._build_example(f"[{path.name}] {text}", resolved_label))
        print(f"[DataDigestor] Imagen procesada: {path.name} -> {len(text)} chars")
        return self

    def from_images_folder(
        self,
        folder: Union[str, Path],
        label_from_subfolder: bool = True,
        lang: str = "es",
    ) -> "DataDigestor":
        """
        Procesa todas las imágenes de una carpeta con OCR.

        Si label_from_subfolder=True, la etiqueta se toma del nombre
        de la subcarpeta (estructura tipo ImageFolder):
            carpeta/
              SPAM/imagen1.jpg
              HAM/imagen2.jpg

        Si label_from_subfolder=False, todas las imágenes se procesan
        en modo extracción (sin etiqueta).

        Requiere: pip install easyocr
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"[DataDigestor] No es una carpeta: {folder}")

        _img_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
        images = list(folder.rglob("*"))
        images = [f for f in images if f.suffix.lower() in _img_exts]

        if not images:
            print(f"[DataDigestor] No se encontraron imágenes en {folder}")
            return self

        print(f"[DataDigestor] Procesando {len(images)} imágenes con OCR (lang={lang})...")
        for img_path in sorted(images):
            lbl = img_path.parent.name if label_from_subfolder else None
            try:
                self.from_image(img_path, label=lbl, lang=lang)
            except Exception as e:
                print(f"  [WARN] Error en {img_path.name}: {e}")

        return self

    def from_images_folder_vlm(
        self,
        folder: Union[str, Path],
        question: str = "Describe detalladamente el contenido de esta imagen.",
        label_from_subfolder: bool = True,
        answer_template: Optional[str] = None,
    ) -> "DataDigestor":
        """
        Genera un dataset VLM multimodal conservando las rutas de imagen.

        A diferencia de from_images_folder() (que hace OCR y extrae texto),
        este método genera mensajes ChatML con referencias a las imágenes
        para entrenar Vision-Language Models (VLMs) con LoRA.

        Cada ejemplo tiene el formato:
          {
            "messages": [
              {
                "role": "user",
                "content": [
                  {"type": "image", "image": "/ruta/absoluta/imagen.jpg"},
                  {"type": "text",  "text": "<question>"}
                ]
              },
              {"role": "assistant", "content": "<label o answer_template>"}
            ]
          }

        Parámetros
        ----------
        folder : str | Path
            Carpeta de imágenes. Estructura tipo ImageFolder:
              folder/
                CLASE_A/imagen1.jpg
                CLASE_B/imagen2.jpg
        question : str
            Pregunta que se hará al VLM sobre cada imagen.
            Ej: "¿Es este meme sexista? Responde YES o NO."
        label_from_subfolder : bool
            Si True, la respuesta del asistente es el nombre de la subcarpeta.
            Si False, la respuesta es answer_template (o texto vacío).
        answer_template : str, opcional
            Plantilla para la respuesta cuando label_from_subfolder=False.
            Ej: "Esta imagen muestra contenido relacionado con {filename}."

        Nota
        ----
        Las rutas se guardan como absolutas para que VLMTrainer pueda
        encontrar las imágenes desde cualquier directorio de trabajo.
        """
        folder = Path(folder).resolve()
        if not folder.is_dir():
            raise NotADirectoryError(f"[DataDigestor] No es una carpeta: {folder}")

        _img_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
        images = sorted([f for f in folder.rglob("*") if f.suffix.lower() in _img_exts])

        if not images:
            print(f"[DataDigestor] No se encontraron imágenes en {folder}")
            return self

        added = 0
        print(f"[DataDigestor] Generando dataset VLM: {len(images)} imágenes...")
        for img_path in images:
            label = img_path.parent.name if label_from_subfolder else None

            if label is None and answer_template:
                answer = answer_template.format(
                    filename=img_path.stem,
                    folder=img_path.parent.name,
                )
            elif label is not None:
                answer = label
            else:
                answer = ""

            # Aplicar label_map si está configurado
            if answer and self.label_map and answer in self.label_map:
                answer = str(self.label_map[answer])

            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(img_path)},
                            {"type": "text",  "text": question},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ]
            }
            self._examples.append(example)
            added += 1

        print(f"[DataDigestor] VLM dataset: {added} ejemplos generados")
        return self

    def from_pdf_tables(
        self,
        path: Union[str, Path],
        table_to_rows: bool = True,
    ) -> "DataDigestor":
        """
        Extrae tablas de un PDF como ejemplos estructurados.
        Requiere: pip install pdfplumber

        Cada fila de cada tabla se convierte en un ejemplo del mismo modo
        que from_csv — usando label_col y text_cols configurados en el Digestor.

        Parámetros
        ----------
        path : str | Path
            Ruta al PDF con tablas.
        table_to_rows : bool
            Si True, cada fila = un ejemplo (recomendado para datasets tabulares).
            Si False, cada tabla = un ejemplo de texto serializado completo.
        """
        try:
            import pdfplumber
            import pandas as pd
        except ImportError:
            raise ImportError(
                "[DataDigestor] 'pdfplumber' no instalado. "
                "Ejecuta: pip install pdfplumber"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Archivo no encontrado: {path}")

        added = skipped = tables_found = 0

        # ── Auto-enriquecimiento de dominio ─────────────────────────
        # Muestreamos las primeras páginas como texto plano para detectar dominio
        _sample_texts: List[str] = []
        with pdfplumber.open(str(path)) as _smp_pdf:
            for _sp, _page in enumerate(_smp_pdf.pages):
                if _sp >= 3:
                    break
                _txt = (_page.extract_text() or "").strip()
                if _txt:
                    _sample_texts.append(_txt)
        self._auto_enrich([t for t in _sample_texts if len(t) > 20])
        # ──────────────────────────────────────────────────────────────

        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for tbl in tables:
                    if not tbl or len(tbl) < 2:
                        continue
                    tables_found += 1
                    # Primera fila = cabecera
                    headers = [str(h).strip() if h else f"col_{i}"
                               for i, h in enumerate(tbl[0])]
                    df = pd.DataFrame(tbl[1:], columns=headers)

                    if table_to_rows:
                        # Procesar como CSV
                        text_cols = [c for c in df.columns if c != self.label_col]
                        for _, row in df.iterrows():
                            if self.label_col and self.label_col in df.columns:
                                raw_label = row[self.label_col]
                                if self.skip_nulls and (raw_label is None or str(raw_label).strip() == ""):
                                    skipped += 1
                                    continue
                                label = self._resolve_label(raw_label)
                            else:
                                label = None
                            text = self._serialize_row(row, text_cols)
                            if not text.strip():
                                skipped += 1
                                continue
                            self._examples.append(self._build_example(
                                f"[Pág.{page_num}] {text}", label
                            ))
                            added += 1
                    else:
                        # Tabla completa como un solo ejemplo
                        text = df.to_string(index=False)
                        self._examples.append(self._build_example(
                            f"[Pág.{page_num} Tabla]\n{text}", label=None
                        ))
                        added += 1

        print(f"[DataDigestor] PDF tablas: {tables_found} tablas, "
              f"{added} ejemplos añadidos, {skipped} omitidos")
        return self

    def from_text_chunks(
        self,
        path: Union[str, Path],
        chunk_size: int = 512,
        overlap: int = 64,
        encoding: str = "utf-8",
    ) -> "DataDigestor":
        """
        Lee un documento largo (TXT, MD, PDF) y lo divide en chunks solapados.
        Útil para entrenar modelos en documentación técnica, contratos, libros, etc.

        Cada chunk se convierte en un ejemplo de extracción (sin etiqueta).
        El modelo aprende a "completar" o "resumir" ese fragmento según la tarea.

        Parámetros
        ----------
        path : str | Path
            Ruta al documento (.txt, .md, .pdf).
        chunk_size : int
            Tamaño de cada chunk en palabras. Por defecto 512.
        overlap : int
            Solapamiento en palabras entre chunks consecutivos. Por defecto 64.
            Evita que el contexto se corte justo en medio de una idea.
        encoding : str
            Codificación para archivos de texto.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Archivo no encontrado: {path}")

        ext = path.suffix.lower()

        # Extraer texto según el tipo de archivo
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                full_text = "\n".join(
                    (page.extract_text() or "") for page in reader.pages
                )
            except ImportError:
                raise ImportError("[DataDigestor] pip install pypdf para chunking de PDFs")
        else:
            full_text = path.read_text(encoding=encoding)

        full_text = _clean_text(full_text)
        words = full_text.split()

        # ── Auto-enriquecimiento de dominio ─────────────────────────
        # Usamos las primeras 500 palabras como muestra
        sample = " ".join(words[:500])
        self._auto_enrich([sample] if sample.strip() else [])
        # ──────────────────────────────────────────────────────────────

        total_words = len(words)

        if total_words == 0:
            print(f"[DataDigestor] Chunking: documento vacío — {path.name}")
            return self

        chunks = []
        step = max(1, chunk_size - overlap)
        for start in range(0, total_words, step):
            chunk_words = words[start: start + chunk_size]
            if len(chunk_words) < max(10, overlap):   # descartar chunks demasiado cortos
                break
            chunks.append(" ".join(chunk_words))

        for i, chunk in enumerate(chunks):
            self._examples.append(
                self._build_example(f"[Fragmento {i+1}/{len(chunks)}] {chunk}", label=None)
            )

        print(f"[DataDigestor] Chunking: {path.name} -> "
              f"{total_words} palabras -> {len(chunks)} chunks "
              f"(size={chunk_size}, overlap={overlap})")
        return self

    def from_conversations(
        self,
        path: Union[str, Path],
        user_field: str = "human",
        assistant_field: str = "gpt",
        encoding: str = "utf-8",
    ) -> "DataDigestor":
        """
        Carga un dataset de conversaciones en formato ShareGPT / OpenAI.

        Formatos soportados:
        1. ShareGPT: lista de {"conversations": [{"from": "human", "value": "..."}, ...]}
        2. OpenAI:   lista de {"messages": [{"role": "user", "content": "..."}, ...]}

        Parámetros
        ----------
        path : str | Path
            Ruta al archivo JSON o JSONL con conversaciones.
        user_field : str
            Nombre del rol humano en formato ShareGPT (default: "human").
        assistant_field : str
            Nombre del rol asistente en formato ShareGPT (default: "gpt").
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Archivo no encontrado: {path}")

        raw = path.read_text(encoding=encoding).strip()
        records = json.loads(raw) if raw.startswith("[") else \
                  [json.loads(l) for l in raw.splitlines() if l.strip()]

        # ── Auto-enriquecimiento de dominio ─────────────────────────
        sample_texts: List[str] = []
        for rec in records[:50]:
            if "messages" in rec:
                for m in rec["messages"]:
                    sample_texts.append(str(m.get("content", "")))
            elif "conversations" in rec:
                for turn in rec["conversations"]:
                    sample_texts.append(str(turn.get("value", "")))
        self._auto_enrich([t for t in sample_texts if t.strip()])
        # ──────────────────────────────────────────────────────────────

        added = skipped = 0
        for record in records:
            # Detectar formato automáticamente
            if "messages" in record:
                # Formato OpenAI — ya en formato ChatML, añadir directamente
                msgs = record["messages"]
                if not any(m.get("role") == "assistant" for m in msgs):
                    skipped += 1
                    continue
                # Asegurar que tiene system prompt
                if not any(m.get("role") == "system" for m in msgs):
                    msgs = [{"role": "system", "content": self.system_prompt}] + msgs
                self._examples.append({"messages": msgs})
                added += 1

            elif "conversations" in record:
                # Formato ShareGPT → convertir a ChatML
                msgs = [{"role": "system", "content": self.system_prompt}]
                for turn in record["conversations"]:
                    role_raw = turn.get("from", "")
                    content  = str(turn.get("value", "")).strip()
                    if not content:
                        continue
                    if role_raw == user_field:
                        msgs.append({"role": "user",      "content": content})
                    elif role_raw == assistant_field:
                        msgs.append({"role": "assistant", "content": content})
                    # ignorar otros roles (system, tool, etc.)
                if len(msgs) < 3:   # system + al menos user + assistant
                    skipped += 1
                    continue
                self._examples.append({"messages": msgs})
                added += 1
            else:
                skipped += 1

        print(f"[DataDigestor] Conversaciones cargadas: {added} ejemplos | {skipped} omitidos")
        return self

    def from_markdown_dialogue(
        self,
        path: Union[str, Path],
        strip_identity: bool = True,
        skip_refusals: bool = True,
        min_turn_chars: int = 2,
        encoding: str = "utf-8",
    ) -> "DataDigestor":
        """
        Destila transcripciones markdown de charlas con IAs → ChatML (modo distill).

        El flujo estrella de la reconversión: exportas una charla con Claude/
        Gemini/ChatGPT (o un session.md) y se convierte en dataset SFT. Detecta
        los marcadores de rol habituales (## Human / **User:** / Assistant: /
        You: / Claude: / ChatGPT: / Gemini:) y construye pares user→assistant.

        Todo DETERMINISTA (regex + reglas): funciona offline, sin LLM.

        Higiene (calidad > cantidad):
          - strip_identity: quita líneas de identidad del modelo frontera
            ("As an AI…", "Soy Claude…") que enseñarían identidad equivocada.
          - skip_refusals: descarta turnos del asistente que son un rechazo.
          - min_turn_chars: solo descarta turnos casi vacíos (default 2). La
            longitud NO es señal de calidad: una respuesta concisa correcta
            ("Son 4.") es dato de primera. La higiene real es semántica.
        Solo se emiten pares user→assistant limpios y en orden; los turnos
        huérfanos o descartados no rompen el ejemplo.

        Parámetros
        ----------
        path : str | Path
            Archivo .md/.txt/.markdown o CARPETA (procesa todos, recursivo).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Ruta no encontrada: {path}")

        if path.is_dir():
            files = sorted(
                p for p in path.rglob("*")
                if p.suffix.lower() in (".md", ".markdown", ".txt")
            )
        else:
            files = [path]
        if not files:
            print(f"[DataDigestor] Sin archivos .md/.txt en {path}")
            return self

        added = skipped = dropped_turns = 0

        for f in files:
            try:
                text = f.read_text(encoding=encoding)
            except Exception as exc:
                print(f"  [!] No se pudo leer {f.name}: {exc}")
                skipped += 1
                continue

            turns = _parse_md_dialogue(text)
            if not turns:
                skipped += 1
                continue

            # Higiene por turno → (rol, contenido|None); None = descartado
            clean: List = []
            for role, content in turns:
                if strip_identity:
                    content = _strip_identity(content)
                if role == "assistant" and skip_refusals and _is_refusal(content):
                    dropped_turns += 1
                    clean.append((role, None))
                    continue
                if len(content.strip()) < min_turn_chars:
                    clean.append((role, None))
                    continue
                clean.append((role, content.strip()))

            # Emparejar user→assistant limpios y en orden. Un turno descartado
            # (None) invalida el par pendiente → nunca se emite un par a medias.
            messages = [{"role": "system", "content": self.system_prompt}]
            pending_user = None
            for role, content in clean:
                if content is None:
                    pending_user = None
                    continue
                if role == "user":
                    pending_user = content
                elif role == "assistant" and pending_user is not None:
                    messages.append({"role": "user", "content": pending_user})
                    messages.append({"role": "assistant", "content": content})
                    pending_user = None

            if len(messages) >= 3:   # system + al menos un par
                self._examples.append({"messages": messages})
                added += 1
            else:
                skipped += 1

        print(
            f"[DataDigestor] Destilación markdown: {added} conversaciones "
            f"({skipped} omitidas, {dropped_turns} turnos descartados por "
            f"higiene) de {len(files)} archivo(s)."
        )
        return self

    def from_document_knowledge(
        self,
        path: Union[str, Path],
        level: str = "auto",
        chunk_chars: int = 1200,
        pairs_per_chunk: int = 2,
        encoding: str = "utf-8",
        judge_url: Optional[str] = None,
        judge_model: Optional[str] = None,
    ) -> "DataDigestor":
        """
        Convierte un documento en bruto (legal, README, apuntes) en dataset.

        Niveles de calidad (ver constantes del modo conocimiento):
          - "completion" : texto crudo → {"text": chunk} (continued-pretraining).
          - "template"   : Q&A por plantilla (standalone, determinista).
          - "llm"        : un LLM redacta Q&A naturales (mejor calidad).
          - "auto"       : usa "llm" si hay endpoint; si no, "template" con aviso.

        STANDALONE: completion/template no tocan la red. "llm"/"auto" degradan a
        "template" con aviso claro si el endpoint no responde (Opción A: el
        embudo nunca se atasca). Solo maneja .txt/.md (documentos de texto);
        PDF/DOCX: extrae texto antes con from_pdf/from_docx o pásalos como .txt.

        Parámetros
        ----------
        path : str | Path
            Archivo .txt/.md/.markdown o carpeta (recursivo).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Ruta no encontrada: {path}")
        if level not in ("completion", "template", "llm", "auto"):
            raise ValueError("level debe ser completion|template|llm|auto, "
                             f"got: {level!r}")

        if path.is_dir():
            files = sorted(
                p for p in path.rglob("*")
                if p.suffix.lower() in (".txt", ".md", ".markdown")
            )
        else:
            files = [path]
        if not files:
            print(f"[DataDigestor] Sin documentos .txt/.md en {path}")
            return self

        # Resolver nivel efectivo (Opción A: degradar con aviso)
        eff = level
        if level in ("llm", "auto"):
            if _llm_reachable(judge_url):
                eff = "llm"
            else:
                eff = "template"
                if level == "llm":
                    print("[DataDigestor] AVISO: nivel 'llm' pedido pero el "
                          "endpoint no responde → degradando a plantilla "
                          "(nivel 2, standalone). Arranca el serve para calidad "
                          "máxima.")
                else:
                    print("[DataDigestor] Sin endpoint LLM → conocimiento por "
                          "plantilla (nivel 2, standalone). Con el serve "
                          "arrancado, 'auto' sube a nivel 3 (Q&A generado).")
        print(f"[DataDigestor] Modo conocimiento — nivel efectivo: '{eff}'.")

        added = 0
        for f in files:
            try:
                text = _clean_text(f.read_text(encoding=encoding))
            except Exception as exc:
                print(f"  [!] No se pudo leer {f.name}: {exc}")
                continue
            for chunk in _chunk_text(text, chunk_chars):
                if eff == "completion":
                    self._examples.append({"text": chunk})
                    added += 1
                elif eff == "template":
                    q = f"¿Qué explica el documento sobre «{_chunk_topic(chunk)}»?"
                    self._examples.append(self._knowledge_chatml(q, chunk))
                    added += 1
                else:  # llm
                    qa = []
                    try:
                        qa = self._llm_qa_pairs(chunk, pairs_per_chunk,
                                                judge_url, judge_model)
                    except Exception as exc:
                        print(f"  [!] Generación LLM falló en un chunk ({exc}); "
                              "plantilla para ese fragmento.")
                    if not qa:   # sin pares válidos → plantilla como red de seguridad
                        qa = [(f"¿Qué explica el documento sobre "
                               f"«{_chunk_topic(chunk)}»?", chunk)]
                    for q, a in qa:
                        self._examples.append(self._knowledge_chatml(q, a))
                        added += 1

        print(f"[DataDigestor] Conocimiento: {added} ejemplos de "
              f"{len(files)} documento(s).")
        return self

    def _knowledge_chatml(self, question: str, answer: str) -> dict:
        """Ejemplo ChatML para el modo conocimiento (system + user + assistant)."""
        return {"messages": [
            {"role": "system",    "content": self.system_prompt},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]}

    def _llm_qa_pairs(self, chunk: str, n: int,
                      url: Optional[str], model: Optional[str]) -> List:
        """Genera hasta n pares (pregunta, respuesta) de un chunk con el LLM.
        Devuelve [] si la respuesta no es parseable (el llamante degrada)."""
        sys = (
            "Eres un generador de datos de entrenamiento SFT. A partir del "
            "PASAJE, redacta preguntas naturales que un usuario real haría y "
            "cuya respuesta esté CONTENIDA en el pasaje. Cada respuesta debe "
            "ser precisa y autocontenida (NO digas 'según el pasaje' ni "
            "'el documento dice'). Calidad sobre cantidad: mejor 1 par "
            "excelente que 3 forzados.\n"
            "Devuelve SOLO un objeto JSON válido, sin texto alrededor:\n"
            '{"pairs": [{"q": "<pregunta>", "a": "<respuesta>"}]}'
        )
        user = f"Genera hasta {n} pares.\n\nPASAJE:\n{chunk}"
        raw = _llm_chat(
            [{"role": "system", "content": sys},
             {"role": "user", "content": user}],
            url=url, model=model,
        )
        data = _extract_json_obj(raw)
        out: List = []
        if data:
            for p in (data.get("pairs") or [])[:n]:
                q = str(p.get("q", "")).strip()
                a = str(p.get("a", "")).strip()
                if q and a:
                    out.append((q, a))
        return out

    def from_docx(
        self,
        path: Union[str, Path],
    ) -> "DataDigestor":
        """
        Carga un archivo .docx (Microsoft Word).
        Cada parrafo no vacio se convierte en un ejemplo individual.

        Requiere: pip install python-docx
        """
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "[DataDigestor] from_docx() requiere python-docx. "
                "Instalalo con: pip install python-docx"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Archivo no encontrado: {path}")

        doc = Document(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        if not paragraphs:
            print(f"[DataDigestor] DOCX sin texto extraible: {path.name}")
            return self

        # ── Auto-enriquecimiento ─────────────────────────────────
        sample_texts = paragraphs[:min(100, len(paragraphs))]
        self._auto_enrich(sample_texts)
        # ──────────────────────────────────────────────────────────

        added = skipped = 0
        for i, p in enumerate(paragraphs):
            if len(p) < 10:
                skipped += 1
                continue
            self._examples.append(
                self._build_example(f"[Párrafo {i+1}/{len(paragraphs)}] {p}", label=None)
            )
            added += 1

        print(f"[DataDigestor] DOCX cargado: {added} ejemplos | {skipped} omitidos (cortos)")
        return self

    def from_html(
        self,
        path: Union[str, Path],
        text_selector: Optional[str] = None,
    ) -> "DataDigestor":
        """
        Carga un archivo .html o .htm extrayendo solo el texto visible.

        Por defecto extrae el texto de <body>. Si se especifica text_selector
        (CSS), extrae solo de los elementos que coincidan.

        Requiere: pip install beautifulsoup4 lxml

        Parámetros
        ----------
        path : str | Path
        text_selector : str, opcional
            Selector CSS para el contenedor de texto. Ej: "article", "main", ".content".
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "[DataDigestor] from_html() requiere beautifulsoup4 + lxml. "
                "Instalalo con: pip install beautifulsoup4 lxml"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Archivo no encontrado: {path}")

        html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")

        # Eliminar scripts, estilos y metadatos
        for tag in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        # Extraer texto del selector o de body entero
        if text_selector:
            parts = soup.select(text_selector)
            if not parts:
                print(f"[DataDigestor] Selector '{text_selector}' no encontró elementos. Usando body completo.")
                parts = [soup.body] if soup.body else []
        else:
            parts = [soup.body] if soup.body else []

        raw_texts: List[str] = []
        for part in parts:
            text = part.get_text(separator="\n", strip=True)
            # Partir en bloques por doble salto de línea
            raw_texts.extend(b.strip() for b in text.split("\n\n") if len(b.strip()) > 30)

        if not raw_texts:
            print(f"[DataDigestor] HTML sin texto extraible: {path.name}")
            return self

        # ── Auto-enriquecimiento ─────────────────────────────────
        sample_texts = raw_texts[:min(100, len(raw_texts))]
        self._auto_enrich(sample_texts)
        # ──────────────────────────────────────────────────────────

        added = skipped = 0
        total = len(raw_texts)
        for i, block in enumerate(raw_texts):
            if len(block) < 20:
                skipped += 1
                continue
            self._examples.append(
                self._build_example(f"[Bloque {i+1}/{total}] {block[:2000]}", label=None)
            )
            added += 1

        print(f"[DataDigestor] HTML cargado: {added} ejemplos | {skipped} omitidos")
        return self

    def from_audio(
        self,
        path: Union[str, Path],
        chunk_minutes: int = 5,
        language: Optional[str] = None,
    ) -> "DataDigestor":
        """
        Transcribe un archivo de audio (MP3, WAV, M4A, OGG, etc.)
        usando faster-whisper y lo divide en fragmentos.

        Requiere: pip install faster-whisper

        Parámetros
        ----------
        path : str | Path
            Ruta al archivo de audio.
        chunk_minutes : int
            Minutos por fragmento de transcripcion (default: 5).
            Cada fragmento se convierte en un ejemplo individual.
        language : str, opcional
            Codigo de idioma (ej: 'es', 'en'). Si None, auto-detecta.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "[DataDigestor] from_audio() requiere faster-whisper. "
                "Instalalo con: pip install faster-whisper"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Archivo no encontrado: {path}")

        print(f"[DataDigestor] Transcribiendo audio: {path.name}...")

        # Cargar modelo small (buen balance velocidad/precision en CPU)
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            str(path),
            language=language,
            beam_size=5,
            vad_filter=True,       # filtrar silencio
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        detected_lang = info.language
        print(f"[DataDigestor] Idioma detectado: {detected_lang} (prob: {info.language_probability:.2f})")

        # Agrupar segmentos en chunks de chunk_minutes
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_time = 0.0
        chunk_seconds = chunk_minutes * 60

        for seg in segments:
            current_chunk.append(seg.text.strip())
            current_time = seg.end
            if current_time >= chunk_seconds and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_time = 0.0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        if not chunks:
            print(f"[DataDigestor] Audio sin texto transcrito: {path.name}")
            return self

        # ── Auto-enriquecimiento ─────────────────────────────────
        sample_texts = chunks[:min(10, len(chunks))]
        self._auto_enrich(sample_texts)
        # ──────────────────────────────────────────────────────────

        added = 0
        for i, chunk in enumerate(chunks):
            if len(chunk) < 30:
                continue
            self._examples.append(
                self._build_example(
                    f"[Transcripcion audio {i+1}/{len(chunks)}, idioma={detected_lang}] {chunk}",
                    label=None,
                )
            )
            added += 1

        print(f"[DataDigestor] Audio transcrito: {added} fragmentos "
              f"({len(chunks)*chunk_minutes} min aprox, idioma={detected_lang})")
        return self

    def from_video(
        self,
        path: Union[str, Path],
        chunk_minutes: int = 5,
        language: Optional[str] = None,
    ) -> "DataDigestor":
        """
        Extrae el audio de un archivo de video y lo transcribe.

        Requiere: pip install faster-whisper
        Requiere ffmpeg en el PATH del sistema (para extraer audio).

        Parámetros
        ----------
        path : str | Path
            Ruta al archivo de video (MP4, MKV, AVI, MOV, WebM, etc.).
        chunk_minutes : int
            Minutos por fragmento de transcripcion.
        language : str, opcional
            Codigo de idioma.
        """
        import subprocess

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[DataDigestor] Archivo no encontrado: {path}")

        # Verificar ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError(
                "[DataDigestor] from_video() requiere ffmpeg en el PATH.\n"
                "  Windows: winget install Gyan.FFmpeg\n"
                "  Linux:   sudo apt install ffmpeg\n"
                "  Mac:     brew install ffmpeg"
            )

        # Extraer audio a WAV temporal
        import tempfile
        wav_path = Path(tempfile.mktemp(suffix=".wav"))
        print(f"[DataDigestor] Extrayendo audio de video: {path.name}...")

        result = subprocess.run([
            "ffmpeg", "-i", str(path),
            "-vn",                # sin video
            "-acodec", "pcm_s16le",  # WAV 16-bit
            "-ar", "16000",       # 16kHz (suficiente para voz)
            "-ac", "1",           # mono
            "-y",                 # sobrescribir
            str(wav_path),
        ], capture_output=True, text=True)

        if not wav_path.exists() or wav_path.stat().st_size < 1000:
            raise RuntimeError(
                f"[DataDigestor] No se pudo extraer audio del video: {path.name}\n"
                f"  ffmpeg stderr: {result.stderr[:300]}"
            )

        # Delegar transcripcion a from_audio
        try:
            return self.from_audio(str(wav_path), chunk_minutes=chunk_minutes, language=language)
        finally:
            # Limpiar WAV temporal
            if wav_path.exists():
                wav_path.unlink()

    # ------------------------------------------------------------------
    # S9 — Personalización de usuario
    # ------------------------------------------------------------------

    def from_user_profile(
        self,
        profile_path: Union[str, Path],
        n_per_tool: int = 5,
        format: str = "react",
        seed: int = 42,
    ) -> "DataDigestor":
        """
        Genera ejemplos de entrenamiento personalizados a partir de un perfil
        de usuario (``user_profile.json``).

        El perfil describe las rutas, contactos y tareas habituales del usuario.
        Los ejemplos resultantes usan esos datos reales en vez de valores
        genéricos, produciendo un adapter personal mucho más preciso.

        Parámetros
        ----------
        profile_path : str | Path
            Ruta a ``user_profile.json``.
        n_per_tool : int
            Ejemplos personalizados a generar por herramienta. Por defecto 5.
        format : str
            ``"react"`` (Thought/Action/…/Final Answer) o ``"function_call"``
            (JSON puro). Debe coincidir con el formato del adapter base.
        seed : int
            Semilla para reproducibilidad.

        Estructura esperada de ``user_profile.json``
        ---------------------------------------------
        .. code-block:: json

            {
              "name":         "Felipe",
              "work_dirs":    ["~/Desktop/Proyecto_V3", "~/Documentos/Trabajo"],
              "notes_folder": "~/Notas",
              "contacts":     {"jefe": "jefe@empresa.com", "equipo": "equipo@empresa.com"},
              "language":     "es",
              "common_tasks": ["organizar las descargas", "guardar notas de reuniones"]
            }

        Devuelve
        --------
        DataDigestor (para encadenar llamadas)

        Ejemplo
        -------
        >>> d = DataDigestor(task="agente personal")
        >>> d.from_user_profile("user_profile.json", n_per_tool=8).to_jsonl("personal.jsonl")
        """
        import random as _random

        profile_path = Path(profile_path)
        if not profile_path.exists():
            raise FileNotFoundError(
                f"[DataDigestor] Perfil no encontrado: {profile_path}"
            )

        with open(profile_path, encoding="utf-8") as _f:
            profile = json.load(_f)

        name         = profile.get("name", "Usuario")
        work_dirs    = profile.get("work_dirs", ["~/Documentos"])
        notes_folder = profile.get("notes_folder", "~/Notas")
        contacts     = profile.get("contacts", {})   # {"alias": "email@..."}
        language     = profile.get("language", "es")
        common_tasks = profile.get("common_tasks", [])
        contact_names = list(contacts.keys()) or (
            ["cliente", "jefe", "equipo"] if language == "es"
            else ["client", "boss", "team"]
        )

        rng = _random.Random(seed)
        added = 0

        # ── Plantillas bilingüe por herramienta ─────────────────────
        _T: Dict[str, Dict[str, List[str]]] = {
            "note_save": {
                "es": [
                    "Guarda una nota titulada '{title}' con el texto '{body}'",
                    "Apunta esto en mis notas: {body}",
                    "Crea una nota sobre '{title}' en {folder}",
                    "Necesito anotar: {body}",
                    "Añade a mis notas: '{title}' — {body}",
                ],
                "en": [
                    "Save a note titled '{title}' with content '{body}'",
                    "Add a note to my notes: {body}",
                    "Create a note about '{title}' in {folder}",
                    "Note this down: {body}",
                    "Add to my notes: '{title}' — {body}",
                ],
            },
            "file_organize": {
                "es": [
                    "Organiza los archivos de {dir}",
                    "Ordena mis documentos en {dir}",
                    "Mueve los archivos descargados a {dir}",
                    "Limpia la carpeta {dir} y organiza su contenido",
                    "Agrupa por tipo los archivos de {dir}",
                ],
                "en": [
                    "Organize files in {dir}",
                    "Sort my documents in {dir}",
                    "Move downloaded files to {dir}",
                    "Clean up {dir} and organize its contents",
                    "Group files by type in {dir}",
                ],
            },
            "search_files": {
                "es": [
                    "Busca archivos sobre '{query}' en {dir}",
                    "¿Dónde están los archivos de '{query}'?",
                    "Encuentra documentos relacionados con '{query}' en {dir}",
                    "Localiza '{query}' en mis carpetas de {dir}",
                    "Busca cualquier archivo que mencione '{query}'",
                ],
                "en": [
                    "Search for files about '{query}' in {dir}",
                    "Where are the '{query}' files?",
                    "Find documents related to '{query}' in {dir}",
                    "Locate '{query}' in my {dir} folders",
                    "Search any file mentioning '{query}'",
                ],
            },
            "email_filter": {
                "es": [
                    "Filtra los correos de {contact}",
                    "Archiva los emails de {contact} del último mes",
                    "Muéstrame los correos de {contact}",
                    "Limpia la bandeja de entrada de mensajes de {contact}",
                    "¿Tengo correos pendientes de {contact}?",
                ],
                "en": [
                    "Filter emails from {contact}",
                    "Archive last month's emails from {contact}",
                    "Show me emails from {contact}",
                    "Clean inbox messages from {contact}",
                    "Do I have pending emails from {contact}?",
                ],
            },
            "calendar_get": {
                "es": [
                    "¿Qué tengo en el calendario esta semana?",
                    "Muéstrame mis eventos de esta semana",
                    "¿Tengo algo programado para mañana?",
                    "Consulta mi agenda del próximo mes",
                    "¿Cuándo es mi próxima reunión?",
                ],
                "en": [
                    "What's on my calendar this week?",
                    "Show me my events this week",
                    "Do I have anything scheduled for tomorrow?",
                    "Check my agenda for next month",
                    "When is my next meeting?",
                ],
            },
        }

        _TOPICS = (
            common_tasks
            or (["proyecto", "reunión", "informe", "presupuesto", "viaje"]
                if language == "es"
                else ["project", "meeting", "report", "budget", "travel"])
        )

        lang_key = "es" if language == "es" else "en"

        def _build_react(tool_name: str, args: Dict, user_msg: str) -> None:
            nonlocal added
            result_str = (
                f"Herramienta '{tool_name}' ejecutada correctamente."
                if language == "es"
                else f"Tool '{tool_name}' executed successfully."
            )
            if language == "es":
                thought1 = (
                    f"El usuario quiere que use '{tool_name}'. "
                    "Voy a llamarla con los parámetros adecuados."
                )
                thought2 = "La herramienta se ejecutó correctamente."
                final = f"Listo. {result_str}"
            else:
                thought1 = (
                    f"The user wants me to use '{tool_name}'. "
                    "I'll call it with the right parameters."
                )
                thought2 = "The tool executed successfully."
                final = f"Done. {result_str}"

            assistant_text = (
                f"Thought: {thought1}\n"
                f"Action: {tool_name}\n"
                f"Action Input: {json.dumps(args, ensure_ascii=False)}\n"
                f"Observation: {result_str}\n"
                f"Thought: {thought2}\n"
                f"Final Answer: {final}"
            )
            self._examples.append({
                "messages": [
                    {"role": "system",    "content": (
                        f"Eres el asistente personal de {name}. "
                        "Usa las herramientas disponibles cuando sea necesario."
                    )},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": assistant_text},
                ]
            })
            added += 1

        def _build_fc(tool_name: str, args: Dict, user_msg: str) -> None:
            nonlocal added
            tool_call_json = json.dumps(
                {"tool": tool_name, "args": args}, ensure_ascii=False
            )
            self._examples.append({
                "messages": [
                    {"role": "system",    "content": (
                        f"Eres el asistente personal de {name}. "
                        "Responde con JSON cuando uses una herramienta."
                    )},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": tool_call_json},
                ]
            })
            added += 1

        _build = _build_react if format == "react" else _build_fc

        # ── note_save ────────────────────────────────────────────────
        tmpls = _T["note_save"][lang_key]
        for _ in range(n_per_tool):
            topic = rng.choice(_TOPICS)
            title = topic[:40]
            body  = f"Recordatorio: {topic}" if language == "es" else f"Reminder: {topic}"
            tmpl  = rng.choice(tmpls)
            user_msg = tmpl.format(title=title, body=body, folder=notes_folder)
            args = {
                "title":    title,
                "body":     body,
                "notebook": Path(notes_folder).name,
            }
            _build("note_save", args, user_msg)

        # ── file_organize ────────────────────────────────────────────
        tmpls = _T["file_organize"][lang_key]
        for _ in range(n_per_tool):
            work_dir = rng.choice(work_dirs)
            tmpl     = rng.choice(tmpls)
            user_msg = tmpl.format(dir=work_dir)
            args = {"dest": work_dir, "dry_run": True}
            _build("file_organize", args, user_msg)

        # ── search_files ─────────────────────────────────────────────
        tmpls = _T["search_files"][lang_key]
        for _ in range(n_per_tool):
            work_dir = rng.choice(work_dirs)
            query    = rng.choice(_TOPICS)
            tmpl     = rng.choice(tmpls)
            user_msg = tmpl.format(query=query, dir=work_dir)
            args = {"query": query, "path": work_dir}
            _build("search_files", args, user_msg)

        # ── email_filter ─────────────────────────────────────────────
        tmpls = _T["email_filter"][lang_key]
        for _ in range(n_per_tool):
            contact  = rng.choice(contact_names)
            tmpl     = rng.choice(tmpls)
            user_msg = tmpl.format(contact=contact)
            args = {"folder": "inbox", "mode": "archive", "dry_run": True}
            _build("email_filter", args, user_msg)

        # ── calendar_get ─────────────────────────────────────────────
        tmpls = _T["calendar_get"][lang_key]
        for _ in range(n_per_tool):
            tmpl     = rng.choice(tmpls)
            user_msg = tmpl
            args = {}
            _build("calendar_get", args, user_msg)

        # ── common_tasks libres ──────────────────────────────────────
        for task in common_tasks:
            # Generar un ejemplo libre que refleje exactamente la tarea del usuario
            _build(
                "note_save",
                {"title": task[:40], "body": task, "notebook": Path(notes_folder).name},
                task,
            )

        print(
            f"[DataDigestor] from_user_profile: {added} ejemplos personalizados "
            f"para '{name}' (n_per_tool={n_per_tool}, format={format})"
        )
        return self

    # ------------------------------------------------------------------
    # Enriquecimiento de dominio (S2.1-A / S2.1-B)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_domain(sample_texts: List[str]) -> Dict[str, Any]:
        """
        Detecta el dominio de un conjunto de textos por heurística de keywords.

        Algoritmo
        ---------
        1. Concatena todos los textos de muestra en un solo string (lowercase).
        2. Para cada dominio (excluyendo sub-dominios _*), cuenta matches.
        3. El dominio con más matches gana.
        4. Si el máximo es < 2 matches → "general".
        5. Calcula nivel de confianza (0-100%) basado en la diferencia
           entre el ganador y el segundo lugar.

        Parámetros
        ----------
        sample_texts : list[str]
            Muestra de textos a analizar (ej: primeras 100 filas del dataset).

        Devuelve
        --------
        dict con claves:
            "domain": str — nombre del dominio principal
            "confidence": float — 0-100% de confianza
            "sub_domain": str — sub-dominio detectado (ej: "patient", "clinical")
            "scores": dict — puntuaciones de todos los dominios
        """
        if not sample_texts:
            return {"domain": "general", "confidence": 0.0,
                    "sub_domain": None, "scores": {}}

        combined = " ".join(sample_texts).lower()

        # Solo dominios principales (excluir sub-dominios _*)
        main_domains = {k: v for k, v in DOMAIN_KEYWORDS.items()
                        if not k.startswith("_")}
        scores: Dict[str, int] = {}

        for domain, keywords in main_domains.items():
            score = 0
            for kw in keywords:
                pattern = re.escape(kw.lower())
                matches = re.findall(pattern, combined)
                score += len(matches)
            scores[domain] = score

        if not scores:
            return {"domain": "general", "confidence": 0.0,
                    "sub_domain": None, "scores": scores}

        # Ordenar por puntuación descendente
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        best_domain, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0

        # Umbral mínimo: al menos 2 keywords detectadas
        if best_score < 2:
            return {"domain": "general", "confidence": 0.0,
                    "sub_domain": None, "scores": scores}

        # Calcular confianza: diferencia entre 1º y 2º, normalizada
        total = sum(scores.values()) or 1
        gap = (best_score - second_score) / max(best_score, 1)
        coverage = best_score / max(total, 1)
        confidence = min(100.0, round((gap * 0.6 + coverage * 0.4) * 100, 1))

        # ── Detectar sub-dominio ─────────────────────────────────────
        sub_domain = DataDigestor._detect_sub_domain(combined, best_domain)

        return {
            "domain": best_domain,
            "confidence": confidence,
            "sub_domain": sub_domain,
            "scores": scores,
        }

    @staticmethod
    def _detect_sub_domain(combined_text: str, main_domain: str) -> Optional[str]:
        """
        Dentro de un dominio principal, detecta el sub-dominio más específico.

        Ej: para "medical", detecta si es "patient" (síntomas cotidianos)
        o "clinical" (jerga médica profesional).
        """
        sub_keys = [k for k in DOMAIN_KEYWORDS if k.startswith(f"_{main_domain}_")]
        if not sub_keys:
            return None

        best_sub = None
        best_score = 0
        for sk in sub_keys:
            score = 0
            for kw in DOMAIN_KEYWORDS[sk]:
                pattern = re.escape(kw.lower())
                score += len(re.findall(pattern, combined_text))
            if score > best_score:
                best_score = score
                # Extraer nombre limpio: "_medical_patient" → "patient"
                best_sub = sk.replace(f"_{main_domain}_", "")

        return best_sub if best_score >= 2 else None

    def enrich_with_domain(self, domain: str, sub_domain: Optional[str] = None) -> "DataDigestor":
        """
        Enriquece el system prompt con contexto del dominio especificado.

        El system prompt original se conserva — el contexto de dominio se
        prepende para que el modelo active sus circuitos neuronales
        especializados durante el entrenamiento.

        Parámetros
        ----------
        domain : str
            Nombre del dominio principal.
        sub_domain : str, opcional
            Sub-dominio para un system prompt mas especifico.
            Ej: "patient" o "clinical" para "medical".

        Devuelve
        --------
        self — para encadenamiento fluido.
        """
        if domain == "auto":
            domain = self._detected_domain or "general"

        if domain not in DOMAIN_SYSTEM_PROMPTS:
            print(f"[DataDigestor] Dominio desconocido '{domain}'. Usando 'general'.")
            domain = "general"

        # Intentar usar prompt de sub-dominio si existe
        prompt_key = domain
        if sub_domain:
            sub_key = f"_{domain}_{sub_domain}"
            if sub_key in DOMAIN_SYSTEM_PROMPTS:
                prompt_key = sub_key

        domain_context = DOMAIN_SYSTEM_PROMPTS[prompt_key]
        original_system = self.system_prompt

        if original_system != _CHATML_SYSTEM and domain != "general":
            # Si el usuario ya puso un system prompt custom, lo respetamos
            self.system_prompt = f"{domain_context}\n\n{original_system}"
        elif domain != "general":
            # Reemplazar el default generico por el especializado de dominio
            self.system_prompt = domain_context
        else:
            # Dominio general: mantener el default o el custom del usuario
            self.system_prompt = original_system

        self._detected_domain = domain
        self._detected_sub_domain = sub_domain
        self._enrichment_done = True

        ctx_type = f"{domain}/{sub_domain}" if sub_domain else domain
        print(f"[DataDigestor] Dominio detectado: {ctx_type}")
        print(f"  System prompt enriquecido ({len(self.system_prompt)} chars)")

        return self

    def _auto_enrich(self, sample_texts: List[str]) -> None:
        """
        Detecta el dominio automáticamente y enriquece el system prompt.

        Solo se ejecuta si:
        - auto_enrich=True (defecto)
        - No se ha ejecutado ya (idempotente vía _enrichment_done)
        - El usuario no forzó un dominio manualmente

        Parámetros
        ----------
        sample_texts : list[str]
            Muestra de textos del dataset para la detección de dominio.
        """
        if not self.auto_enrich or self._enrichment_done:
            return

        if self._forced_domain:
            self.enrich_with_domain(self._forced_domain)
            return

        result = self._detect_domain(sample_texts)
        domain = result["domain"]
        confidence = result["confidence"]
        sub_domain = result.get("sub_domain")

        # Guardar info de confianza para consulta pública
        self._domain_confidence = confidence

        if domain == "general":
            print(f"[DataDigestor] Dominio: general (confianza baja, "
                  f"max score < umbral)")
            self._detected_domain = "general"
            self._enrichment_done = True
            return

        # Mostrar candidatos top-3 para transparencia
        ranked = sorted(result["scores"].items(), key=lambda x: -x[1])[:3]
        candidates = ", ".join(
            f"{d}={s}" for d, s in ranked if s > 0 and not d.startswith("_")
        )
        sub_info = f" ({sub_domain})" if sub_domain else ""
        print(f"[DataDigestor] Dominio: {domain}{sub_info} "
              f"(confianza: {confidence:.0f}%)")
        if candidates:
            print(f"  Candidatos: {candidates}")

        self.enrich_with_domain(domain, sub_domain=sub_domain)

    # ------------------------------------------------------------------
    # Model awareness (S2.1-C)
    # ------------------------------------------------------------------

    def _probe_model(self) -> None:
        """
        Sondea el modelo objetivo usando ModelAnalyzer para extraer
        metadatos relevantes sin descargar pesos.

        Rellena:
        - _model_family: str (ej: "qwen", "llama", "mistral")
        - _model_supports_system: bool
        - _model_max_seq_length: int
        """
        try:
            from motor.analyzer import ModelAnalyzer
            analyzer = ModelAnalyzer(self.model_id)
            info = analyzer.analyze()
            self._model_family = info.get("family", "unknown")
            self._model_supports_system = info.get("supports_system_prompt", True)
            self._model_max_seq_length = info.get("max_position_embeddings", 2048)
            print(
                f"[DataDigestor] Modelo sondeado: {self._model_family} | "
                f"system_prompt={'✓' if self._model_supports_system else '✗'} | "
                f"ctx={self._model_max_seq_length}"
            )
        except Exception as e:
            print(f"[DataDigestor] No se pudo sondear el modelo ({e}). "
                  f"Usando defaults (system=✓, ctx=2048).")

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _resolve_label(self, raw: Any) -> str:
        """Convierte un valor raw al string de etiqueta limpio."""
        import math
        # NaN / None
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            return self.null_placeholder.upper()
        # Usar label_map si existe
        if raw in self.label_map:
            return str(self.label_map[raw])
        # Intentar también con el valor como int (para claves int en un dict leído de CSV)
        try:
            if int(raw) in self.label_map:
                return str(self.label_map[int(raw)])
        except (ValueError, TypeError):
            pass
        # Por defecto: convertir a string mayúsculas
        return str(raw).strip().upper()

    def _serialize_row(self, row: "pd.Series", text_cols: List[str]) -> str:
        """
        Convierte una fila de DataFrame en una cadena de texto legible.
        Ej: "Clase=3, Nombre=Braund Mr. Owen, Sexo=male, Edad=22, Tarifa=7.25"
        """
        import pandas as pd
        parts = []
        for col in text_cols:
            val = row.get(col)
            if pd.isna(val):
                val = self.null_placeholder
            else:
                # Redondear floats con muchos decimales
                if isinstance(val, float):
                    val = f"{val:.4g}"
                else:
                    val = str(val).strip()
            if val:
                parts.append(f"{col}={val}")
        return ", ".join(parts)

    def _build_example(self, text: str, label: Optional[str]) -> dict:
        """Construye el dict de ejemplo según el formato de salida."""
        if self.output_format == "chatml":
            return self._build_chatml(text, label)
        return self._build_alpaca(text, label)

    def _task_variation(self) -> str:
        """
        Devuelve una variacion aleatoria de la tarea para evitar que el modelo
        memorice la pregunta exacta (sesgo del prompt).

        El 50% de las veces devuelve la tarea original; el otro 50%
        selecciona entre reformulaciones que preservan el significado.
        """
        import random
        task = self.task
        if not task:
            return ""   # modos distill/knowledge/vlm no enmarcan con tarea
        if len(self._examples) > 0:
            # Quitar puntuacion final para construir variantes limpias
            core = task.rstrip(" .")
            variants = [
                task,                                           # original
                f"Por favor, {core[0].lower()}{core[1:]}.",    # cortesia
                f"{core}. Da solo la respuesta.",               # instruccion breve
                task,                                           # original (mas peso)
                f"Pregunta: {core}?",                           # formato QA
                task,                                           # original (mas peso)
            ]
            return random.choice(variants)
        return task

    def _build_chatml(self, text: str, label: Optional[str]) -> dict:
        """
        Formato ChatML adaptado al modelo objetivo (model-aware).

        - Si el modelo soporta system prompt (Qwen, Llama 3, Gemma, Phi):
          {"messages": [{"role":"system",...}, {"role":"user",...}, ...]}

        - Si el modelo NO soporta system prompt (Mistral, Llama 2 antiguo):
          El contexto de sistema se inyecta en el mensaje user como prefijo.
        """
        _tv = self._task_variation()
        user_content = f"{_tv}\n\n{text}" if _tv else text
        messages = []

        if self._model_supports_system:
            messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user",   "content": user_content})
        else:
            # Modelos sin system prompt: inyectar contexto en user
            prefixed_user = (
                f"[Contexto: {self.system_prompt}]\n\n{user_content}"
            )
            messages.append({"role": "user", "content": prefixed_user})

        if label is not None:
            messages.append({"role": "assistant", "content": label})
        return {"messages": messages}

    def _build_alpaca(self, text: str, label: Optional[str]) -> dict:
        """
        Formato Alpaca:
        {
          "instruction": "<task>",
          "input":       "<text>",
          "output":      "<label>"   ← vacío si no hay etiqueta
        }
        """
        return {
            "instruction": self._task_variation(),
            "input":        text,
            "output":       label if label is not None else "",
        }

    def _print_label_distribution(self) -> None:
        """Imprime la distribución de etiquetas para detectar desbalance."""
        if not self._examples:
            return
        counts: Dict[str, int] = {}
        for ex in self._examples:
            if self.output_format == "chatml":
                msgs = ex.get("messages", [])
                assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
                lbl = assistant or "<sin etiqueta>"
            else:
                lbl = ex.get("output") or "<sin etiqueta>"
            counts[lbl] = counts.get(lbl, 0) + 1
        total = sum(counts.values())
        print("  Distribución de etiquetas:")
        for lbl, cnt in sorted(counts.items()):
            pct = cnt / total * 100
            print(f"    {lbl:20s}: {cnt:5d} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Limpieza básica de texto extraído de PDF u otras fuentes."""
    # Colapsar espacios múltiples y líneas en blanco repetidas
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_file_type(path: Union[str, Path]) -> str:
    """
    Detecta el tipo de archivo por extensión.
    Devuelve: "csv" | "excel" | "json" | "jsonl" | "txt" | "pdf" | "image" | "unknown"
    """
    ext = Path(path).suffix.lower()
    _map = {
        ".csv":   "csv",
        ".xlsx":  "excel",
        ".xls":   "excel",
        ".json":  "json",
        ".jsonl": "jsonl",
        ".txt":   "txt",
        ".md":    "txt",
        ".pdf":   "pdf",
        ".docx":  "docx",
        ".html":  "html",
        ".htm":   "html",
        ".mp3":   "audio",
        ".wav":   "audio",
        ".m4a":   "audio",
        ".ogg":   "audio",
        ".flac":  "audio",
        ".mp4":   "video",
        ".mkv":   "video",
        ".avi":   "video",
        ".mov":   "video",
        ".webm":  "video",
        ".png":   "image",
        ".jpg":   "image",
        ".jpeg":  "image",
        ".webp":  "image",
        ".bmp":   "image",
        ".tiff":  "image",
        ".gif":   "image",
    }
    return _map.get(ext, "unknown")
